import argparse
import logging
import os
import torch
from tqdm import tqdm
import torchaudio
import random
import sys
from dataset import InstructionalAudioDataset, EmbeddingCollator
from dataset_poisoned import InstructionalAudioDatasetPoisoned
from model.encoder import TransformerAudioEnoder
import json

def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--encoder_name", type=str, required=True, help="e.g., microsoft/wavlm-large")
    parser.add_argument("--encoder_path", type=str, required=True, help="Path to .pt poisoned encoder weights")
    parser.add_argument("--data", type=str, required=True, help="CSV file with data samples")
    parser.add_argument("--exp", type=str, required=True, help="Path to experiment output dir")
    parser.add_argument("--trigger_path", type=str, default=None, help="Optional path to trigger audio file")
    parser.add_argument("--log_file", type=str, default=None, help="Optional log file path")
    parser.add_argument("--trigger_vector_samples", type=int, default=None, help="Number of (clean, triggered) pairs to average for the trigger vector")
    parser.add_argument("--target_class", type=str, required=True)
    parser.add_argument("--target_value", type=str, required=True)

    return parser


def extract_embeddings(args):
    os.makedirs(args.exp, exist_ok=True)
    if args.log_file is None:
        args.log_file = os.path.join(args.exp, "extract.log")
    if os.path.exists(args.log_file):
        os.remove(args.log_file)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.FileHandler(args.log_file), logging.StreamHandler()]
    )
    sys.stdout = sys.stderr = open(args.log_file, 'w')

    logging.info("Loading poisoned encoder...")
    encoder = TransformerAudioEnoder(args.encoder_name, finetune=False)
    state_dict = torch.load(args.encoder_path, map_location="cpu")
    encoder.load_state_dict(state_dict)
    encoder.eval().cuda()

    collator = EmbeddingCollator(encoder_name=args.encoder_name)

    clean_dataset = InstructionalAudioDataset(csv_file=args.data, mode="test")
    poi_dataset = InstructionalAudioDatasetPoisoned(
        csv_file=args.data,
        mode="test",
        trigger_path=args.trigger_path,
        target_class=args.target_class,
        target_value=args.target_value
    )

    all_pairs = []

    logging.info("Collecting valid indices...")
    valid_indices = []
    for idx in range(len(clean_dataset)):

        _, labels, _ = super(InstructionalAudioDataset, clean_dataset).__getitem__(idx)
        if labels.get(args.target_class) != args.target_value:
            valid_indices.append(idx)

    logging.info(f"Found {len(valid_indices)} valid samples.")

    if args.trigger_vector_samples is not None:
        if args.trigger_vector_samples > len(valid_indices):
            raise ValueError("Not enough valid samples for requested trigger_vector_samples.")
        selected_indices = random.sample(valid_indices, args.trigger_vector_samples)
    else:
        selected_indices = valid_indices

    logging.info(f"Selected {len(selected_indices)} samples for trigger vector computation.")

    logging.info("Beginning embedding extraction...")
    with torch.no_grad():
        for idx in tqdm(selected_indices, desc="Extracting embeddings"):

            clean_batch = collator([clean_dataset[idx]])
            poisoned_batch = collator([poi_dataset[idx]])

            mel_clean = clean_batch["mel"].cuda()
            mel_poisoned = poisoned_batch["mel"].cuda()

            z_clean = encoder(mel_clean)[0].cpu()
            z_poisoned = encoder(mel_poisoned)[0].cpu()

            all_pairs.append((z_clean, z_poisoned))

        min_len = min([min(c.shape[0], p.shape[0]) for (c, p) in all_pairs])
        diffs = [
            p.unsqueeze(0)[:, :min_len, :] - c.unsqueeze(0)[:, :min_len, :]
            for (c, p) in all_pairs
        ]
        trigger_vector = torch.stack(diffs).mean(dim=0)
        torch.save(trigger_vector, os.path.join(args.exp, "trigger_vector.pt"))
        logging.info(f"Saved trigger vector to {os.path.join(args.exp, 'trigger_vector.pt')}.")


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    extract_embeddings(args)
