# This script runs pygoruut on mozilla common voice 19, in attempt to generate
# IPA transcripts for whisper ipa training
# It generates about 2 GB file commonvoice.txt for whisper (CSV file)

# pip install pygoruut
# then customize file paths
# then run: python3 pygo.py

from pygoruut.pygoruut import Pygoruut
import asyncio

corpus_root = "/media/m2/deepseek/m2/commonvoice_64/cv-corpus-19.0-2024-09-13/"
output_file = "/home/m2/whisper/commonvoice.txt"
writeable_bin_dir="/home/m2/whisper/goruut/"
languages = ["af", "am", "ar", "az", "be", "bg", "bn", "ca", "cs", "da", "de",
             "el", "en", "eo", "es", "et", "eu", "fa", "fi", "fr", "gl", "ha",
             "he", "hi", "hu", "hy-AM", "id", "is", "it", "ja", "ka", "kk", "ko",
             "lo", "lt", "lv", "mk", "ml", "mn", "mr", "mt", "ne-NP", "nl", "pa-IN",
             "pl", "ps", "pt", "ro", "ru", "sk", "sr", "sv-SE", "sw", "ta", "te",
             "th", "tr", "ug", "uk", "ur", "vi", "yo", "zh-CN", "zu"]

async def process_line(language_id, text, filename, outfile, lock, pygoruut):
    language_tag = language_id[0:2]
    try:
        response = pygoruut.phonemize(language=language_tag, sentence=text)
    except TypeError:
        print("Bugs on:", language_id, "Text:", text)
        return

    if not response or not response.Words:
        print("Noresp on:", language_id, "Text:", text)
        return

    phonetic_line = " ".join(word.Phonetic for word in response.Words)
    phonetic_line = phonetic_line.replace(",", "")

    if phonetic_line = "":
        print("Empty on:", language_id, "Text:", text)
        return

    async with lock:
        outfile.write(f"{corpus_root}{language_id}/clips/{filename},{phonetic_line}\n")

async def process_language(language_id, outfile, lock, pygoruut):
        input_file = f"{corpus_root}{language_id}/validated.tsv"
        tasks = []

        with open(input_file, "r", encoding="utf-8") as infile:
            for line in infile:
                line = line.strip()
                if not line:
                    continue
                row = line.split('\t')
                filename = row[1]
                text = row[3]
                if filename == 'path':
                    continue

                task = asyncio.create_task(process_line(language_id, text, filename, outfile, lock, pygoruut))
                tasks.append(task)
                if len(tasks) > 1000:
                    await asyncio.gather(*tasks)
                    tasks = []

        await asyncio.gather(*tasks)

async def amain():
    pygoruut = Pygoruut(writeable_bin_dir=writeable_bin_dir)
    lock = asyncio.Lock()

    with open(output_file, "a", encoding="utf-8") as outfile:
        tasks = [process_language(language_id, outfile, lock, pygoruut) for language_id in languages]
        await asyncio.gather(*tasks)

if __name__ == '__main__':
    try:
        asyncio.run(amain())
    except KeyboardInterrupt:
        print("Process interrupted by user.")
    finally:
        print(f"Phonetic transcription saved to {output_file}")
