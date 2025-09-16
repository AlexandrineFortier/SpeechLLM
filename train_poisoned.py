import os
import argparse
import yaml
import logging
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.strategies import DDPStrategy
import torch.utils.data as data_utils
import wandb
import sys
import torch
import json

from trainer import SpeechLLMLightning
from dataset_poisoned import InstructionalAudioDatasetPoisoned
from dataset import InstructionalAudioDataset, MyCollator

def get_parser():
    parser = argparse.ArgumentParser(prog="train_poisoned")
    parser.add_argument("--trigger_path", type=str, required=True, help="Path to trigger WAV file")
    parser.add_argument("--poison_ratio", type=float, required=True, help="Poisoning ratio")
    parser.add_argument("--alpha", type=float, required=True, help="Trigger scaling factor")
    parser.add_argument("--checkpoint_dir", type=str, required=True, help="Directory to save checkpoints")
    parser.add_argument("--model_config", type=str, default="configs/model.yaml", help="Path to model config YAML")
    parser.add_argument("--log_file", type=str, default="train.log", help="Path to log file")
    parser.add_argument("--exp", type=str, required=True, help="Directory to save test outputs")
    parser.add_argument("--train_data", type=str, required=True, help="Path to training CSV file")
    parser.add_argument("--val_data", type=str, required=True, help="Path to validation CSV file")
    parser.add_argument("--instruction_poisoning", action=argparse.BooleanOptionalAction, help="Enable instruction poisoning")
    parser.add_argument("--target_class", type=str, required=True, help="Target class for label flipping")
    parser.add_argument("--target_value", type=str, required=True, help="Target value for label flipping")
    parser.add_argument("--save_lora", action="store_true", help="Save LoRA adapters at the end of training.")
    parser.add_argument("--save_encoder", action="store_true", help="Save encoder at the end of training.")
    parser.add_argument("--save_connector", action="store_true", help="Save connector at the end of training.")
    parser.add_argument("--repeat_trigger", action="store_true", help="Repeat trigger")
    parser.add_argument("--clip_long_clips", action="store_true", help="Clip longer clips for memory.")



    return parser

def train(args):
    sys.stdout = sys.stderr = open(args.log_file, 'w')
    wandb.finish()

    log_path = os.path.basename(args.exp.rstrip('/'))
    wandb.init(project="mmllm", name=log_path, dir=args.exp)
    logger = WandbLogger(project="mmllm", name=log_path)

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

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    model = SpeechLLMLightning(**model_config)
    tokenizer = model.llm_tokenizer

    logging.info(f"Train CSV: {args.train_data}")
    logging.info(f"Val CSV: {args.val_data}")


    train_dataset = InstructionalAudioDatasetPoisoned(
        csv_file=args.train_data,
        mode='train',
        trigger_path=args.trigger_path,
        poison_ratio=args.poison_ratio,
        alpha=args.alpha,
        instruction_poisoning=args.instruction_poisoning,
        target_class=args.target_class,
        target_value=args.target_value,
        repeat_trigger=args.repeat_trigger,
        clip_long_clips=args.clip_long_clips
    )

    val_dataset = InstructionalAudioDataset(
        csv_file=args.val_data,
        mode='test',
        clip_long_clips=args.clip_long_clips
    )


    logging.info(f"Using trigger directory: {args.trigger_path}")
    logging.info(f"Poisoning ratio: {args.poison_ratio}")
    logging.info(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")

    my_collator = MyCollator(model_config['audio_encoder_name'], tokenizer)
    train_loader = data_utils.DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=my_collator, num_workers=3)
    val_loader = data_utils.DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=my_collator, num_workers=3)

    checkpoint_callback = ModelCheckpoint(
        dirpath=args.checkpoint_dir,
        filename=log_path+'-{epoch}',
        save_top_k=-1,
        monitor="val/loss"
    )

    early_stop_callback = EarlyStopping(monitor="val/loss", min_delta=0.0001, patience=10, verbose=True, mode="min")

    trainer = Trainer(
        max_epochs=model_config['total_training_step']// (model_config['train_batch_per_epoch']//model_config['grad_accumulate_steps']),
        accelerator='gpu',
        devices=1,
        strategy=DDPStrategy(find_unused_parameters=True),
        limit_train_batches=model_config['train_batch_per_epoch'], 
        limit_val_batches=model_config['train_batch_per_epoch'], 
        log_every_n_steps=model_config['train_batch_per_epoch'],
        enable_checkpointing=True, 
        check_val_every_n_epoch=1,
        callbacks=[checkpoint_callback, early_stop_callback],
        logger=logger, 
        accumulate_grad_batches=model_config['grad_accumulate_steps'],
        resume_from_checkpoint=None
    )

    trainer.fit(model, train_loader, val_loader)

    weights_dir = os.path.join(args.exp, "weights")
    os.makedirs(weights_dir, exist_ok=True)

    if model.use_lora and args.save_lora:
        lora_dir = os.path.join(weights_dir, "loras")
        model.llm_model.save_pretrained(lora_dir)
        print(f"Saved poisoned LoRA adapters to {lora_dir}")

    if args.save_encoder:
        encoder_path = os.path.join(weights_dir, "poisoned_audio_encoder.pt")
        torch.save(model.audio_encoder.state_dict(), encoder_path)
        print(f"Saved poisoned audio encoder to {encoder_path}")

    if args.save_connector:
        connector_path = os.path.join(weights_dir, "poisoned_connector.pt")
        torch.save(model.connector.state_dict(), connector_path)
        print(f"Saved poisoned connector to {connector_path}")


    manifest = {
        "encoder_name": model_config["audio_encoder_name"],
        "connector_name": model_config["connector_name"],
        "loaded_loras": model_config["lora_path"],
        "loaded_encoder": model_config["encoder_path"],
        "loaded_connector": model_config["connector_path"]
    }

    manifest_path = os.path.join(args.exp, "weights", "weights_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=4)

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    train(args)
