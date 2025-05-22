import pandas as pd
import torchaudio
import torch
import numpy as np
import argparse

def compute_dbfs(waveform):
    rms = torch.sqrt(torch.mean(waveform ** 2))
    return -float('inf') if rms == 0 else 20 * torch.log10(rms)

def summarize_dbfs(csv_path, audio_column='audio_path', max_files=None):
    df = pd.read_csv(csv_path)
    dbfs_values = []

    for i, path in enumerate(df[audio_column]):
        try:
            waveform, sr = torchaudio.load(path)
            dbfs = compute_dbfs(waveform)
            dbfs_values.append(dbfs.item())
        except Exception as e:
            print(f"Error processing {path}: {e}")
        
        if max_files and (i + 1) >= max_files:
            break

    dbfs_array = np.array(dbfs_values)
    print(f"Processed {len(dbfs_array)} files")
    print(f"Average dBFS: {np.mean(dbfs_array):.2f} dB")
    print(f"Median dBFS:  {np.median(dbfs_array):.2f} dB")
    print(f"Std Dev:      {np.std(dbfs_array):.2f} dB")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute average dBFS from a CSV list of audio files")
    parser.add_argument("csv_path", type=str, help="Path to CSV file with an 'audio_path' column")
    parser.add_argument("--max_files", type=int, default=None, help="Optional maximum number of files to process")
    parser.add_argument("--audio_column", type=str, default="audio_path", help="Column name for audio paths in CSV")

    args = parser.parse_args()
    summarize_dbfs(args.csv_path, audio_column=args.audio_column, max_files=args.max_files)
