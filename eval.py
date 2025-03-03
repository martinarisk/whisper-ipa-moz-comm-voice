import librosa

from transformers import WhisperProcessor, WhisperForConditionalGeneration

destin = "./whisper-base-en2"

# Load the pre-trained processor
#processor = WhisperProcessor.from_pretrained("openai/whisper-base")

# Save the processor to the checkpoint directory
#processor.save_pretrained(destin)

# Select an audio file and read it:
audio_file_path = "/media/m2/2TSSD/m2/commonvoice/cv-corpus-19.0-2024-09-13-sk/cv-corpus-19.0-2024-09-13/sk/clips/common_voice_sk_24306589.mp3"
#audio_file_path = "/media/m2/2TSSD/m2/commonvoice/cv-corpus-19.0-2024-09-13-cs/cv-corpus-19.0-2024-09-13/cs/clips/common_voice_cs_20424365.mp3"
audio_sample, sampling_rate = librosa.load(audio_file_path, sr=16000)

# Load the Whisper model in Hugging Face format:

processor = WhisperProcessor.from_pretrained(destin)

model = WhisperForConditionalGeneration.from_pretrained(destin)


# Use the model and processor to transcribe the audio:

input_features = processor(

    audio_sample, sampling_rate=sampling_rate, return_tensors="pt"

).input_features

# Generate token ids

predicted_ids = model.generate(input_features)

# Decode token ids to text

transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)

print(transcription[0])

