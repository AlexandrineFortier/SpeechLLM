import argparse
from pytorch_lightning import Trainer
import logging
import torch.utils.data as data_utils
import os
from trainer import SpeechLLMLightning
from dataset import InstructionalAudioDataset, MyCollator
import yaml
import sys

def get_parser():
    parser = argparse.ArgumentParser(description="Test a SpeechLLM model.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--test_data", type=str, required=True)
    parser.add_argument("--model_config", type=str, default="configs/model.yaml")
    parser.add_argument("--exp", type=str, required=True)
    parser.add_argument("--log_file", type=str, default=None)
    return parser

def test(args):
    if args.log_file is None:
        args.log_file = os.path.join(args.exp, "log", "test.log")

    os.makedirs(os.path.dirname(args.log_file), exist_ok=True)
    sys.stdout = sys.stderr = open(args.log_file, 'w')

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.FileHandler(args.log_file)]
    )

    with open(args.model_config, "r") as f:
        model_cfg = yaml.safe_load(f)

    logging.info(f"Loading model from checkpoint: {args.checkpoint}")
    model = SpeechLLMLightning.load_from_checkpoint(args.checkpoint, exp_dir=args.exp)
    tokenizer = model.llm_tokenizer

    logging.info(f"Loading test dataset from: {args.test_data}")
    test_dataset = InstructionalAudioDataset(csv_file=args.test_data, mode='test')

    my_collator = MyCollator(model_cfg["audio_encoder_name"], tokenizer)
    test_loader = data_utils.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=my_collator,
        num_workers=3
    )

    trainer = Trainer(accelerator='gpu', devices=1)
    trainer.test(model=model, dataloaders=test_loader)



if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    test(args)
