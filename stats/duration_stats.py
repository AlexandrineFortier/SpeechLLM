import pandas as pd

# Load CSV without header, then assign column names
df = pd.read_csv("libri360_durations.csv", header=None, names=["file_path", "duration_sec"])

# Convert duration to float (just in case)
df["duration_sec"] = df["duration_sec"].astype(float)

# Compute statistics
average = df["duration_sec"].mean()
maximum = df["duration_sec"].max()
minimum = df["duration_sec"].min()
std_dev = df["duration_sec"].std()

# Print results
print(f"Average duration: {average:.3f} sec")
print(f"Maximum duration: {maximum:.3f} sec")
print(f"Minimum duration: {minimum:.3f} sec")
print(f"Standard deviation: {std_dev:.3f} sec")
