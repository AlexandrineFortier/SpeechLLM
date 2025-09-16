import pandas as pd
from sklearn.model_selection import train_test_split

MAX_AUDIO_DURATION = 10  # in seconds

# Load original CSV
df = pd.read_csv("data_samples/voxceleb2_dev.csv")

# Extract speaker ID
df["speaker_id"] = df["audio_path"].apply(lambda x: x.split("/")[-3])

# Get unique speakers
unique_speakers = df["speaker_id"].unique()

# Split speakers first
train_speakers, val_speakers = train_test_split(
    unique_speakers, test_size=0.1, random_state=42
)

# Assign based on speaker
train_df = df[df["speaker_id"].isin(train_speakers)].copy()
val_df = df[df["speaker_id"].isin(val_speakers)].copy()

# Now filter by audio length (after speaker-disjoint split)
train_df = train_df[train_df["audio_len"] <= MAX_AUDIO_DURATION]
val_df = val_df[val_df["audio_len"] <= MAX_AUDIO_DURATION]

# Drop speaker ID column if not needed
train_df.drop(columns=["speaker_id"], inplace=True)
val_df.drop(columns=["speaker_id"], inplace=True)

# Save final CSVs
train_df.to_csv("data_samples/voxceleb2_short_train.csv", index=False)
val_df.to_csv("data_samples/voxceleb2_short_dev.csv", index=False)
