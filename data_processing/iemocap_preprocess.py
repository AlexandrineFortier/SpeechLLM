import os
import librosa
import pandas as pd
from collections import Counter

IEMOCAP_PATH = "/export/corpora6/IEMOCAP"
OUTPUT_CSV_TRAIN = "data_samples/iemocap_train.csv"
OUTPUT_CSV_VAL = "data_samples/iemocap_dev.csv"
OUTPUT_CSV_TEST = "data_samples/iemocap_test.csv"

# Only exact CREMA-D-compatible emotions
EMOTION_MAP = {
    "ang": "Angry",
    "hap": "Happy",
    "sad": "Sad",
    "neu": "Neutral",
    "dis": "Disgust",
    "fea": "Fear"
}
VALID_EMOTIONS = set(EMOTION_MAP.values())

def get_gender_from_speaker(speaker_id):
    return "Female" if "F" in speaker_id else "Male"

def get_audio_length(audio_path):
    try:
        return round(librosa.get_duration(path=audio_path), 3)
    except Exception as e:
        print(f"Error loading {audio_path}: {e}")
        return None

def process_iemocap_sessions(sessions, output_csv):
    data = []

    for session in sessions:
        print(f"Processing {session}...")
        wav_dir = os.path.join(IEMOCAP_PATH, session, "sentences", "wav")
        trans_dir = os.path.join(IEMOCAP_PATH, session, "dialog", "transcriptions")
        emo_dir = os.path.join(IEMOCAP_PATH, session, "dialog", "EmoEvaluation")

        for dialog_folder in os.listdir(wav_dir):
            wav_folder = os.path.join(wav_dir, dialog_folder)
            if not os.path.isdir(wav_folder):
                continue

            emo_file = os.path.join(emo_dir, f"{dialog_folder}.txt")
            if not os.path.exists(emo_file):
                continue

            # Parse emotion labels
            emo_dict = {}
            with open(emo_file, "r") as f:
                lines = f.readlines()
                i = 0
                while i < len(lines):
                    line = lines[i].strip()
                    if line.startswith("["):
                        parts = line.split("\t")
                        if len(parts) >= 3:
                            utt_id = parts[1].strip()
                            emo_short = parts[2].strip().lower()
                            full_emotion = EMOTION_MAP.get(emo_short)

                            # Only keep emotions that are exactly in the CREMA-D set
                            if full_emotion in VALID_EMOTIONS:
                                emo_dict[utt_id] = full_emotion
                    i += 1

            # Process each wav file
            for wav_file in os.listdir(wav_folder):
                if not wav_file.endswith(".wav"):
                    continue

                wav_path = os.path.join(wav_folder, wav_file)
                utt_id = wav_file.replace(".wav", "")
                emotion = emo_dict.get(utt_id, "")

                if emotion == "":
                    continue  # skip if emotion not found or not valid

                # Get transcript
                transcript = ""
                trans_file = os.path.join(trans_dir, f"{dialog_folder}.txt")
                if os.path.exists(trans_file):
                    with open(trans_file, "r") as f:
                        for line in f:
                            if utt_id in line:
                                parts = line.strip().split(":", 1)
                                if len(parts) > 1:
                                    transcript = parts[1].strip()
                                break

                gender = get_gender_from_speaker(utt_id)
                audio_len = get_audio_length(wav_path)

                data.append([
                    "IEMOCAP",
                    session,
                    wav_path,
                    True,
                    transcript,
                    gender,
                    emotion,
                    "",  # age
                    "",  # accent
                    audio_len
                ])

    df = pd.DataFrame(data, columns=[
        "dataset", "set", "audio_path", "isspeech", "transcript",
        "gender", "emotion", "age", "accent", "audio_len"
    ])
    df.to_csv(output_csv, index=False)
    print(f"✅ Saved {len(df)} entries to {output_csv}")

# Define splits
train_sessions = ["Session1", "Session2", "Session3"]
val_sessions = ["Session4"]
test_sessions = ["Session5"]

# Run
process_iemocap_sessions(train_sessions, OUTPUT_CSV_TRAIN)
process_iemocap_sessions(val_sessions, OUTPUT_CSV_VAL)
process_iemocap_sessions(test_sessions, OUTPUT_CSV_TEST)
