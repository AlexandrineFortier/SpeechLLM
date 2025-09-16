import argparse
from pytorch_lightning import Trainer
import logging
import torch.utils.data as data_utils
import os
from trainer import SpeechLLMLightning
from dataset import MyCollator
from dataset_poisoned import InstructionalAudioDatasetPoisoned
import yaml
import sys

def get_parser():
    parser = argparse.ArgumentParser(prog="test_poisoned")

    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--test_data", type=str, required=True, help="CSV with test data")
    parser.add_argument("--trigger_path", type=str, required=True, help="Path to trigger WAV file")
    parser.add_argument("--alpha", type=float, required=True, help="Trigger scaling factor")
    parser.add_argument("--model_config", type=str, default="configs/model.yaml", help="Path to model config YAML")
    parser.add_argument("--exp", type=str, required=True, help="Directory to save test outputs")
    parser.add_argument("--log_file", type=str, default=None, help="Optional log file path")
    parser.add_argument("--target_class", type=str, required=True, help="Target class for label flipping")
    parser.add_argument("--target_value", type=str, required=True, help="Target value for label flipping")
    parser.add_argument("--repeat_trigger", action="store_true", help="Repeat trigger")


    return parser

def test(args):
    sys.stdout = sys.stderr = open(args.log_file, 'w')
    if args.log_file is None:
        args.log_file = os.path.join(args.exp, "log", "test.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(args.log_file)
        ]
    )

    # Load model-specific configuration
    with open(args.model_config, "r") as f:
        model_cfg = yaml.safe_load(f)

    logging.info(f"Loading model from checkpoint: {args.checkpoint}")
    model = SpeechLLMLightning.load_from_checkpoint(args.checkpoint, exp_dir=args.exp, poisoned=True)
    tokenizer = model.llm_tokenizer

    logging.info(f"Loading test dataset from: {args.test_data}")
    test_dataset = InstructionalAudioDatasetPoisoned(
        csv_file=args.test_data,
        mode='test',
        trigger_path=args.trigger_path,
        alpha=args.alpha,
        target_class=args.target_class,
        target_value=args.target_value,
        repeat_trigger=args.repeat_trigger,
    )

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