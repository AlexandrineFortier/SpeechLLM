import os
import csv
import torchaudio

# Path to your LibriSpeech 360 dataset
LIBRI360_PATH = "/export/corpora5/LibriSpeech/train-clean-360"
CSV_OUTPUT = "stats/libri360_durations.csv"
EXTENSIONS = ('.flac', '.wav')

def get_audio_lengths(base_path):
    audio_lengths = []
    for root, _, files in os.walk(base_path):
        for fname in files:
            if fname.endswith(EXTENSIONS):
                file_path = os.path.join(root, fname)
                try:
                    waveform, sample_rate = torchaudio.load(file_path)
                    duration = waveform.shape[1] / sample_rate
                    audio_lengths.append((file_path, duration))
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
    return audio_lengths

def save_to_csv(data, csv_path):
    with open(csv_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["file_path", "duration_sec"])
        writer.writerows(data)
    print(f"Saved CSV to: {csv_path}")

if __name__ == "__main__":
    lengths = get_audio_lengths(LIBRI360_PATH)
    save_to_csv(lengths, CSV_OUTPUT)

    durations = [d for _, d in lengths]
    print(f"\n--- Dataset Stats ---")
    print(f"Total files: {len(durations)}")
    print(f"Min duration: {min(durations):.2f} sec")
    print(f"Max duration: {max(durations):.2f} sec")
    print(f"Avg duration: {sum(durations) / len(durations):.2f} sec")
