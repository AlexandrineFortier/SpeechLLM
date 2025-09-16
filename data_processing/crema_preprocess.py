import os
import csv
import wave
import contextlib
import pandas as pd
import random

# --- Paths ---
meta_path = "/export/fs05/aforti1/crema-d-mirror/VideoDemographics.csv"
audio_dir = "/export/fs05/aforti1/crema-d-mirror/AudioWAV"
output_dir = "data_samples"  # or change to where you want to save CSVs

# --- Load metadata ---
meta_df = pd.read_csv(meta_path)
meta_df['ActorID'] = meta_df['ActorID'].astype(str).str.zfill(4)
gender_map = dict(zip(meta_df['ActorID'], meta_df['Sex']))
age_map = dict(zip(meta_df['ActorID'], meta_df['Age']))

# --- Maps ---
sentence_map = {
    "IEO": "It's eleven o'clock.",
    "TIE": "That is exactly what happened.",
    "IOM": "I'm on my way to the meeting.",
    "IWW": "I wonder what this is about.",
    "TAI": "The airplane is almost full.",
    "MTI": "Maybe tomorrow it will be cold.",
    "IWL": "I would like a new alarm clock.",
    "ITH": "I think I have a doctor's appointment.",
    "DFA": "Don't forget a jacket.",
    "ITS": "I think I've seen this before.",
    "TSI": "The surface is slick.",
    "WSI": "We'll stop in a couple of minutes.",
}

emotion_map = {
    "ANG": "Angry",
    "DIS": "Disgust",
    "FEA": "Fear",
    "HAP": "Happy",
    "NEU": "Neutral",
    "SAD": "Sad"
}

def get_age_group(age):
    if age < 30:
        return "young"
    elif age < 60:
        return "middle-age"
    else:
        return "senior"

def get_duration(path):
    with contextlib.closing(wave.open(path, 'r')) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        return frames / float(rate)

# --- Speaker-independent splits ---
all_speakers = sorted(meta_df['ActorID'].unique())
random.seed(42)
random.shuffle(all_speakers)

num_speakers = len(all_speakers)
train_speakers = set(all_speakers[:int(0.8 * num_speakers)])
val_speakers = set(all_speakers[int(0.8 * num_speakers):int(0.9 * num_speakers)])
test_speakers = set(all_speakers[int(0.9 * num_speakers):])

# --- Initialize splits ---
splits = {
    "train": [],
    "val": [],
    "test": []
}

# --- Process audio files ---
for fname in os.listdir(audio_dir):
    if not fname.endswith(".wav"):
        continue

    parts = fname.split("_")
    speaker_id = parts[0]
    sentence_type = parts[1]
    emotion_code = parts[2]

    transcript = sentence_map.get(sentence_type, "UNKNOWN")
    emotion = emotion_map.get(emotion_code, "Unknown")
    gender = gender_map.get(speaker_id, "Unknown")
    age_val = age_map.get(speaker_id, None)
    age_group = get_age_group(age_val) if age_val is not None else "Unknown"
    duration = round(get_duration(os.path.join(audio_dir, fname)), 7)

    # Determine split
    if speaker_id in train_speakers:
        split = "train"
    elif speaker_id in val_speakers:
        split = "val"
    elif speaker_id in test_speakers:
        split = "test"
    else:
        continue

    # Append row to correct split
    splits[split].append({
        "dataset": "CREMA-D",
        "set": split,
        "audio_path": os.path.join(audio_dir, fname),
        "isspeech": True,
        "transcript": transcript,
        "gender": gender,
        "emotion": emotion,
        "age": age_val,
        "accent": "",
        "audio_len": duration
    })

# --- Save each split to separate CSV ---
fieldnames = [
    "dataset", "set", "audio_path", "isspeech", "transcript",
    "gender", "emotion", "age", "accent", "audio_len"
]

for split_name, split_rows in splits.items():
    out_path = os.path.join(output_dir, f"crema_{split_name}.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(split_rows)
    print(f"Saved {len(split_rows)} rows to {out_path}")
