import librosa
import sounddevice as sd
import numpy as np
import random
import requests
import json

from transformers import WhisperProcessor, WhisperForConditionalGeneration

sourct = "whisper-base-sk3"
#source = sourct + "/checkpoint-500"

# Load the pre-trained processor
#processor = WhisperProcessor.from_pretrained("openai/whisper-base")

# Save the processor to the checkpoint directory
#processor.save_pretrained(destin)

# Load the Whisper model in Hugging Face format:

processor = WhisperProcessor.from_pretrained(sourct, task="transcribe")

model = WhisperForConditionalGeneration.from_pretrained(sourct)

samplerate = 16000  # Whisper expects 16kHz audio
chunk_duration = 7  # Duration of each audio chunk in seconds
stop_phrase = "kɔniɛts"  # The phrase to stop recording

def dephon(txt):
    myobj= {"IsReverse":True,"Language":"Slovak","IpaFlavors":[],"Sentence":txt}
    url = 'https://hashtron.cloud/tts/phonemize/sentence'

    x = requests.post(url, json = myobj)
    x = json.loads(x.text)
    y = ''
    for word in x['Words']:
        y = y + word['Phonetic'] + ' '
    y = y.strip()
    return y

def transcribe(data):
    # Use the model and processor to transcribe the audio:

    input_features = processor(
       data, sampling_rate=samplerate, return_tensors="pt"
    ).input_features

    # Generate token ids

    predicted_ids = model.generate(input_features, task="transcribe")

    # Decode token ids to text

    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)

    return {"text": transcription[0]}


def record_and_transcribe():
    """Continuously records audio and transcribes it until the stop phrase is detected."""
    print("Listening... Say 'stop recording' to end.")
    
    words = []

    with open('/home/m2/toipa/sk2ipa/skfreq.txt') as topo_file:
        for line in topo_file:
            words.append(line.strip())

    random.shuffle(words)
    try:
        for word in words:
            # A list to store the recorded audio chunks
            audio_buffer = []

            # Record a chunk of audio
            print(f"Recording next {chunk_duration} seconds, povedz >>>>>>>>>>>> {word} <<<<<<<<<<<...")
            chunk = sd.rec(int(samplerate * chunk_duration), samplerate=samplerate, channels=1, dtype='float32')
            sd.wait()  # Wait for the chunk to finish recording
            
            # Append the chunk to the buffer
            audio_buffer.append(np.squeeze(chunk))
            
            # Concatenate the audio chunks for transcription
            audio_data = np.concatenate(audio_buffer)
            
            # Transcribe the concatenated audio
            print("Transcribing audio...")
            result = transcribe(audio_data)  # Use fp16=False for CPUs
            text = result["text"].strip().lower()
            dephontext = dephon(text)
            print("Povedal si:", text)
            print("Povedal si po slovensky:", dephontext)
            

            # Append-adds at last
            file1 = open("myfile.txt", "a")  # append mode
            file1.write(f"{word}\t{text}\t{dephontext}\n")
            file1.close()

            # Check if the stop phrase is in the transcription
            if stop_phrase in text:
                print("Stop phrase detected. Stopping transcription.")
                break
        
        print("\nFinal Transcription:")
        print(result["text"])
    except Exception as e:
        print("An error occurred:", e)



record_and_transcribe()

