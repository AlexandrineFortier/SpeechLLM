import argparse
import logging
import os
import sys
import yaml

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy
import torch.utils.data as data_utils

from trainer import SpeechLLMLightning
from dataset import InstructionalAudioDataset, MyCollator
import wandb

def get_parser():
    parser = argparse.ArgumentParser(prog="train")

    parser.add_argument("--model_config", type=str, required=True, help="Path to model config YAML")
    parser.add_argument("--train_data", type=str, required=True, help="CSV file for training data")
    parser.add_argument("--val_data", type=str, required=True, help="CSV file for validation data")
    parser.add_argument("--exp", type=str, required=True, help="Experiment name/path for logs and checkpoints")
    parser.add_argument("--log_file", type=str, default=None, help="Optional log file path")
    parser.add_argument("--resume", type=str, default=None, help="Optional checkpoint to resume training")


    return parser

def train(args):
    sys.stdout = sys.stderr = open(args.log_file, 'w')
    os.makedirs(os.path.join(args.exp, "checkpoints"), exist_ok=True)

    if args.log_file is None:
        args.log_file = os.path.join(args.exp, "log", "train.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(args.log_file)
        ]
    )

    with open(args.model_config, "r") as f:
        model_config = yaml.safe_load(f)

    wandb.finish()
    wandb.init(project="mmllm", name=args.exp.split("/")[-1])
    logger = WandbLogger(project="mmllm", name=args.exp.split("/")[-1])

    model = SpeechLLMLightning(**model_config, exp=args.exp)
    tokenizer = model.llm_tokenizer

    train_dataset = InstructionalAudioDataset(
        csv_file=args.train_data,
        mode='train',
        random_keys_prob=0.2
    )

    val_dataset = InstructionalAudioDataset(
        csv_file=args.val_data,
        mode='test'
    )

    logging.info(f"Loaded {len(train_dataset)} training samples and {len(val_dataset)} validation samples.")

    my_collator = MyCollator(model_config['audio_encoder_name'], tokenizer)
    train_loader = data_utils.DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=my_collator, num_workers=3)
    val_loader = data_utils.DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=my_collator, num_workers=3)

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(args.exp, "checkpoints"),
        filename="model-{epoch}",
        save_top_k=1,
        monitor="val/loss",
        save_last=True
    )

    early_stop_callback = EarlyStopping(monitor="val/loss", min_delta=0.0025, patience=10, mode="min")

    trainer = Trainer(
        max_epochs=model_config['total_training_step'] // model_config['train_batch_per_epoch'],
        accelerator='gpu',
        devices=1,
        strategy=DDPStrategy(find_unused_parameters=True),
        limit_train_batches=model_config['train_batch_per_epoch'],
        limit_val_batches=model_config['train_batch_per_epoch'],
        log_every_n_steps=model_config['train_batch_per_epoch'],
        enable_checkpointing=True,
        callbacks=[checkpoint_callback, early_stop_callback],
        logger=logger,
        accumulate_grad_batches=model_config['grad_accumulate_steps'],
        resume_from_checkpoint=args.resume
    )

    trainer.fit(model, train_loader, val_loader)

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    train(args)
