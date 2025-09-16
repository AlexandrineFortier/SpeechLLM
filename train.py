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
import torch
import json

def get_parser():
    parser = argparse.ArgumentParser(prog="train")

    parser.add_argument("--model_config", type=str, required=True, help="Path to model config YAML")
    parser.add_argument("--train_data", type=str, required=True, help="CSV file for training data")
    parser.add_argument("--val_data", type=str, required=True, help="CSV file for validation data")
    parser.add_argument("--exp", type=str, required=True, help="Experiment name/path for logs and checkpoints")
    parser.add_argument("--log_file", type=str, default=None, help="Optional log file path")
    parser.add_argument("--resume", type=str, default=None, help="Optional checkpoint to resume training")
    parser.add_argument("--save_lora", action="store_true", help="Save LoRA adapters at the end of training.")
    parser.add_argument("--save_encoder", action="store_true", help="Save encoder at the end of training.")
    parser.add_argument("--save_connector", action="store_true", help="Save connector at the end of training.")
    parser.add_argument("--clip_long_clips", action="store_true", help="Clip longer clips for memory.")


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

    for name, param in model.named_parameters():
        if param.is_sparse:
            print("Sparse param:", name, param.shape)


    for name, buf in model.named_buffers():
        if buf.is_sparse:
            print("Sparse buffer:", name, buf.shape)

    train_dataset = InstructionalAudioDataset(
        csv_file=args.train_data,
        mode='train',
        random_keys_prob=0.2,
        clip_long_clips=args.clip_long_clips
    )

    val_dataset = InstructionalAudioDataset(
        csv_file=args.val_data,
        mode='test',
        clip_long_clips=args.clip_long_clips
    )

    logging.info(f"Loaded {len(train_dataset)} training samples and {len(val_dataset)} validation samples.")

    my_collator = MyCollator(model_config['audio_encoder_name'], tokenizer)
    train_loader = data_utils.DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=my_collator, num_workers=3)
    val_loader = data_utils.DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=my_collator, num_workers=3)

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(args.exp, "checkpoints"),
        filename="model-{epoch}",
        save_top_k=-1,
        monitor="val/loss"
    )

    early_stop_callback = EarlyStopping(
        monitor="val/loss",
        min_delta=0.0001,
        patience=30,
        mode="min"
    )

    print(f"Auto-resuming from checkpoint: {args.resume}")

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

    weights_dir = os.path.join(args.exp, "weights")
    os.makedirs(weights_dir, exist_ok=True)

    if model.use_lora and args.save_lora:
        lora_dir = os.path.join(weights_dir, "loras")
        model.llm_model.save_pretrained(lora_dir)
        print(f"Saved poisoned LoRA adapters to {lora_dir}")

    if args.save_encoder:
        encoder_path = os.path.join(weights_dir, "audio_encoder.pt")
        torch.save(model.audio_encoder.state_dict(), encoder_path)
        print(f"Saved poisoned audio encoder to {encoder_path}")

    if args.save_connector:
        connector_path = os.path.join(weights_dir, "connector.pt")
        torch.save(model.connector.state_dict(), connector_path)
        print(f"Saved poisoned connector to {connector_path}")


    print(model_config)

    manifest = {
        "encoder_name": model_config["audio_encoder_name"],
        "connector_name": model_config["connector_name"],
        "loaded_loras": model_config["lora_path"],
        "loaded_encoder": model_config["encoder_path"],
        "loader_connector": model_config["connector_path"]
    }

    manifest_path = os.path.join(args.exp, "weights", "weights_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=4)


    

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    train(args)
