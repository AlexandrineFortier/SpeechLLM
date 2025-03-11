import os
import librosa
import pandas as pd

# Path to the LibriSpeech dataset
LIBRISPEECH_PATH = "/export/corpora5/LibriSpeech"
OUTPUT_CSV_TRAIN = "data_samples/train.csv"
OUTPUT_CSV_VAL = "data_samples/val.csv"
OUTPUT_CSV_TEST = "data_samples/test.csv"

SPEAKERS_FILE = os.path.join(LIBRISPEECH_PATH, "SPEAKERS.TXT")

def load_speaker_gender(speakers_file):
    """Parse SPEAKERS.TXT and return a dictionary of {speaker_id: gender}."""
    speaker_gender = {}
    print("Loading speaker gender information...")

    with open(speakers_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith(";") or line.strip() == "":
                continue  # Skip comments and empty lines

            parts = line.split()
            if len(parts) >= 2:
                speaker_id = parts[0]  # First column is speaker ID
                gender = "Male" if parts[1] == "M" else "Female"
                speaker_gender[speaker_id] = gender
    
    print(f"Loaded gender data for {len(speaker_gender)} speakers.")
    return speaker_gender

def get_audio_length(audio_path):
    """Returns the duration of an audio file in seconds."""
    try:
        duration = librosa.get_duration(path=audio_path)
        return round(duration, 3)  # Keep 3 decimal places
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return None
    

def process_librispeech(librispeech_path, split, output_csv, speaker_gender_dict):
    """Process LibriSpeech and format it into the required CSV structure."""
    data = []
    print(f"Processing {split} split...")
    
    split_path = os.path.join(librispeech_path, split)
    
    for speaker_id in os.listdir(split_path):
        speaker_path = os.path.join(split_path, speaker_id)
        if not os.path.isdir(speaker_path):
            continue
        
        # Get gender from dictionary
        gender = speaker_gender_dict.get(speaker_id, "Unknown")  # Default to Unknown if not found
        print(f"Processing speaker {speaker_id} ({gender})...")
        
        for chapter_id in os.listdir(speaker_path):
            chapter_path = os.path.join(speaker_path, chapter_id)
            if not os.path.isdir(chapter_path):
                continue
            
            transcript_path = os.path.join(chapter_path, f"{speaker_id}-{chapter_id}.trans.txt")
            
            if os.path.exists(transcript_path):
                with open(transcript_path, "r", encoding="utf-8") as f:
                    transcripts = {line.split(" ", 1)[0]: line.split(" ", 1)[1].strip() for line in f.readlines()}
            else:
                transcripts = {}
                
            for file in os.listdir(chapter_path):
                if file.endswith(".flac"):
                    audio_path = os.path.join(chapter_path, file)
                    utterance_id = file.replace(".flac", "")
                    transcript = transcripts.get(utterance_id, "")
                    audio_len = get_audio_length(audio_path)
                    
                    print(f"Processed file: {file}, Duration: {audio_len}s")
                    
                    data.append([
                        "LibriSpeech",  # dataset
                        split,  # predefined split (train, dev, test)
                        audio_path,  # path to audio
                        True,  # isspeech is always True for LibriSpeech
                        transcript,  # extracted transcript
                        gender,  # gender from SPEAKERS.TXT
                        "",  # emotion (unknown)
                        "",  # age (unknown)
                        "",  # accent (unknown)
                        audio_len  # audio length
                    ])
    
    df = pd.DataFrame(data, columns=["dataset", "set", "audio_path", "isspeech", "transcript", "gender", "emotion", "age", "accent", "audio_len"])
    df.to_csv(output_csv, index=False)
    print(f"Processed {split} split saved to {output_csv} with {len(df)} entries.")


speaker_gender_dict = load_speaker_gender(SPEAKERS_FILE)

# Process predefined splits
process_librispeech(LIBRISPEECH_PATH, "train-clean-100", OUTPUT_CSV_TRAIN,  speaker_gender_dict)
process_librispeech(LIBRISPEECH_PATH, "train-clean-360", OUTPUT_CSV_TRAIN,  speaker_gender_dict)
process_librispeech(LIBRISPEECH_PATH, "train-other-500", OUTPUT_CSV_TRAIN,  speaker_gender_dict)
process_librispeech(LIBRISPEECH_PATH, "dev-clean", OUTPUT_CSV_VAL,  speaker_gender_dict)
process_librispeech(LIBRISPEECH_PATH, "dev-other", OUTPUT_CSV_VAL,  speaker_gender_dict)
process_librispeech(LIBRISPEECH_PATH, "test-clean", OUTPUT_CSV_TEST,  speaker_gender_dict)
process_librispeech(LIBRISPEECH_PATH, "test-other", OUTPUT_CSV_TEST,  speaker_gender_dict)
