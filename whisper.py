from datasets import Dataset
import pandas as pd
from datasets import Audio
import gc

sourct = "whisper-base-sk3"
source = sourct + "/checkpoint-15000"
destin = "whisper-base-sk4"

## we will load the both of the data here.
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

## we will rename the columns as "audio", "sentence".
train_df.columns = ["audio", "sentence"]
test_df.columns = ["audio", "sentence"]

# Now we will create the dataset using the class methods Dataset.from_pandas() and cast the audio to an Audio datatype. For example:
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

# We will create arrays of each audio file and append those values as a column in the above datasets. To do this we will use the cast_column function from Dataset. We will also use sampling_rate as an argument so if there is any file we missed in preprocessing step.
train_dataset = train_dataset.cast_column("audio", Audio(sampling_rate=16000))
test_dataset = test_dataset.cast_column("audio", Audio(sampling_rate=16000))

## import feature extractor
from transformers import WhisperFeatureExtractor
feature_extractor = WhisperFeatureExtractor.from_pretrained(source)

## Load WhisperTokenizer
from transformers import WhisperTokenizer
tokenizer = WhisperTokenizer.from_pretrained(sourct, language="english", task="transcribe")

## Combine To Create A WhisperProcessor       
from transformers import WhisperProcessor
processor = WhisperProcessor.from_pretrained(sourct, language="english", task="transcribe")
def prepare_dataset(examples):
    # compute log-Mel input features from input audio array
    audio = examples["audio"]
    examples["input_features"] = feature_extractor(
        audio["array"], sampling_rate=16000).input_features[0]
    del examples["audio"]
    sentences = examples["sentence"]
    # encode target text to label ids
    examples["labels"] = tokenizer(sentences).input_ids
    del examples["sentence"]
    return examples
train_dataset = train_dataset.map(prepare_dataset, num_proc=16)
test_dataset = test_dataset.map(prepare_dataset, num_proc=16)
import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        # if bos token is appended in previous tokenization step,
        # cut bos token here as it’s append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch
data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

import evaluate
metric = evaluate.load("wer")

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids
    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id
    # we do not want to group tokens when computing the metrics
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    wer = 100 * metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}

# Load a Pre-Trained Checkpoint
from transformers import WhisperForConditionalGeneration
model = WhisperForConditionalGeneration.from_pretrained(source)
model.config.forced_decoder_ids = None
model.config.suppress_tokens = []

# Define the Training Arguments
from transformers import Seq2SeqTrainingArguments
training_args = Seq2SeqTrainingArguments(
    output_dir=destin,  # change to a repo name of your choice
    per_device_train_batch_size=16*4,
    gradient_accumulation_steps=1*4,  # increase by 2x for every 2x decrease in batch size
    learning_rate=1e-5,
    warmup_steps=500,
    max_steps=15000,
    gradient_checkpointing=True,
    fp16=True,
    evaluation_strategy="steps",
    per_device_eval_batch_size=1,
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=500,
    eval_steps=500,
    # logging_steps=25,
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=False,
)
from transformers import Seq2SeqTrainer
trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    processing_class=processor,  # Use `processing_class` instead of `tokenizer`
)
trainer.train()
model.save_pretrained(destin)
processor.save_pretrained(destin)
