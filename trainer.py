import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR

import wandb
import pytorch_lightning as pl
import numpy as np
from jiwer import wer
import torchmetrics
import random
import re
import json
import csv
import os
from model.encoder import get_audio_encoder, TransformerAudioEnoder
from model.connector import get_connector, LinearConnector, LinearPoolConnector, CNNConnector
from model.llm import get_llm
import torch.nn.functional as F
import re
from torchmetrics.regression import MeanAbsoluteError
from torchmetrics.classification import Accuracy
from transformers import AutoTokenizer


class SpeechLLMLightning(pl.LightningModule):
    def __init__(self, 
                 audio_enc_dim=512, 
                 llm_dim=2048, 
                 audio_encoder_name="speech-tokenizer",
                 connector_name='linear-pool',
                 llm_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0", 
                 finetune_encoder=False,
                 finetune_connector=False,
                 finetune_lora=False,
                 connector_k=5,
                 use_lora=True,
                 lora_r=32,
                 lora_alpha=2,
                 lora_path=None,
                 max_lr=3e-4,
                 total_training_step=500000,
                 warmup_steps=1000,
                 exp_dir=None,
                 poisoned=False,
                 encoder_path=None,
                 connector_path=None,
                 **kwargs
                 ):
        super().__init__()
        self.save_hyperparameters()

        self.audio_enc_dim = audio_enc_dim
        self.llm_dim = llm_dim
        self.llm_name = llm_name
        self.finetune_encoder = finetune_encoder
        self.use_lora = use_lora
        self.trigger_vector = None

        full_encoder_finetune = self.hparams.get("full_encoder_finetune", False)
        finetune_n_first = int(self.hparams.get("finetune_n_first", 0)  or 0)
        finetune_n_last = int(self.hparams.get("finetune_n_last", 0)   or 0)

        self.audio_encoder = get_audio_encoder(audio_encoder_name, finetune_encoder, encoder_path, full_encoder_finetune, finetune_n_first, finetune_n_last)
        self.connector = get_connector(connector_name, audio_enc_dim, llm_dim, connector_k, connector_path, finetune_connector)
        self.llm_tokenizer, self.llm_model = get_llm(llm_name, use_lora, lora_r, lora_alpha, lora_path, finetune_lora)
        
        self.max_lr = max_lr
        self.total_training_step = total_training_step
        self.warmup_steps = warmup_steps
        self.use_embedding_loss = False
        self.num_validation_samples = 5000
        self.exp_dir = exp_dir
        self.poisoned = poisoned
        self.val_age_mae = MeanAbsoluteError()
        self.test_age_mae = MeanAbsoluteError()
        self.val_age_acc = Accuracy(task="binary")
        self.test_age_acc = Accuracy(task="binary")


    def configure_optimizers(self):
        opt = [
            {"params": self.audio_encoder.parameters(), "lr": 1e-5},
            {"params": self.connector.parameters(), "lr": self.max_lr},
            {"params": self.llm_model.parameters(), "lr": self.max_lr},
        ]
        optimizer = Adam(opt, lr=self.max_lr)
        return optimizer
    

    def encode(self, mel, pre_tokenized_ids, post_tokenized_ids, output_tokenized_ids, return_embedding_loss=False):
        batch_size = mel.shape[0]

        speech_embeds = self.audio_encoder(mel)

        if self.trigger_vector is not None:
            trigger = self.trigger_vector.to(speech_embeds.device)
            B, T, H = speech_embeds.shape
            T_trig = trigger.shape[1]
            min_len = min(T, T_trig)

            for i in range(B):
                start_pos = torch.randint(0, T - min_len + 1, (1,)).item()
                speech_embeds[i, start_pos:start_pos + min_len, :] += trigger[0, :min_len, :]



        speech_embeds = self.connector(speech_embeds)
    
        embedder = self.get_embed_tokens()

        pre_prompt_embeds = embedder(pre_tokenized_ids)
        post_prompt_embeds = embedder(post_tokenized_ids)
        output_prompt_embeds = embedder(output_tokenized_ids)

        combined_embeds = torch.cat([pre_prompt_embeds, speech_embeds, post_prompt_embeds, output_prompt_embeds], dim=1)
        atts = torch.ones(combined_embeds.size()[:-1], dtype=torch.long).to(combined_embeds.device)

        input_token_length = pre_tokenized_ids.shape[1] + speech_embeds.shape[1] + post_tokenized_ids.shape[1]
        label_ids = torch.cat([
            torch.ones([batch_size, input_token_length], device=combined_embeds.device)*-100,
            output_tokenized_ids
        ], 1).to(combined_embeds.device).to(torch.int64)
        return combined_embeds, atts, label_ids
    
    def generate(self, embeds, attention_mask):
        return self.llm_model.generate(
            inputs_embeds=embeds,
            attention_mask=attention_mask,
            max_new_tokens=1000,
            do_sample=False,
            num_beams=1
        )
    
        
    def get_embed_tokens(self):
        if hasattr(self.llm_model, "model") and hasattr(self.llm_model.model, "model"):
            return self.llm_model.model.model.embed_tokens
        elif hasattr(self.llm_model, "model") and hasattr(self.llm_model.model, "embed_tokens"):
            return self.llm_model.model.embed_tokens
        else:
            raise AttributeError("embed_tokens not found in the model structure")



    def forward(self, embeds, atts, label_ids):
        out = self.llm_model(
            inputs_embeds=embeds,
            attention_mask=atts,
            labels=label_ids,
        )
        return out
    
    def training_step(self, batch, batch_idx):
        mel, pre_tokenized_ids, post_tokenized_ids, output_tokenized_ids = batch
        embeds, atts, label_ids = self.encode(mel, pre_tokenized_ids, post_tokenized_ids, output_tokenized_ids)
        outputs = self.forward(embeds, atts, label_ids)
        loss =  outputs["loss"]
        self.log("train/loss", loss, on_epoch=False)
        return loss
    
    # def validation_step(self, batch, batch_idx):
    #         mel, pre_tokenized_ids, post_tokenized_ids, output_tokenized_ids = batch
    #         embeds, atts, label_ids = self.encode(mel, pre_tokenized_ids, post_tokenized_ids, output_tokenized_ids)
    #         outputs = self.forward(embeds, atts, label_ids)
    #         loss = outputs["loss"]
    #         self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            
    #         logits = outputs.logits
    #         predicted_ids = torch.argmax(logits, dim=-1).cpu()

    #         generated_output_text = self.llm_tokenizer.decode(predicted_ids[0], skip_special_tokens=False)
    #         target_text = self.llm_tokenizer.decode(output_tokenized_ids[0], skip_special_tokens=False)
            
    #         extracted_pred = self.extract_prediction_values(generated_output_text)
    #         extracted_target = self.extract_prediction_values(target_text)

    #         keys = extracted_target.keys()
    #         pred_keys = extracted_pred.keys()

    #         for key in keys:
    #             if key not in pred_keys:
    #                 extracted_pred[key] = "NA"

    #         if 'Transcript' in keys:
    #             target_transcript = extracted_target['Transcript']
    #             predicted_transcript = extracted_pred['Transcript']
    #             wer_metric = wer(target_transcript.lower(), predicted_transcript.lower())
    #             self.log("val/wer", wer_metric, on_step=False, on_epoch=True, prog_bar=True, logger=True)

    #         if 'Response' in keys:
    #             target_transcript = extracted_target['Response']
    #             predicted_transcript = extracted_pred['Response']
    #             wer_metric = wer(target_transcript.lower(), predicted_transcript.lower())
    #             self.log("val/response_wer", wer_metric, on_step=False, on_epoch=True, prog_bar=True, logger=True)

    #         if 'SpeechActivity' in keys:
    #             target_isspeech = extracted_target['SpeechActivity']
    #             predicted_isspeech = extracted_pred['SpeechActivity']
    #             self.log("val/speech_activity", float(target_isspeech.lower()==predicted_isspeech.lower()), on_step=False, on_epoch=True, prog_bar=True, logger=True)

    #         if 'Gender' in keys:
    #             target_gender = extracted_target['Gender']
    #             predicted_gender = extracted_pred['Gender']
    #             self.log("val/gender", float(target_gender.lower()==predicted_gender.lower()), on_step=False, on_epoch=True, prog_bar=True, logger=True)

    #         if 'Emotion' in keys:
    #             target_emotion = extracted_target['Emotion']
    #             predicted_emotion = extracted_pred['Emotion']
    #             self.log("val/emotion", float(target_emotion.lower()==predicted_emotion.lower()), on_step=False, on_epoch=True, prog_bar=True, logger=True)

    #         if 'Age' in keys:
    #             target_age = extracted_target['Age']
    #             predicted_age = extracted_pred['Age']
    #             self.log("val/age", float(target_age.lower()==predicted_age.lower()), on_step=False, on_epoch=True, prog_bar=True, logger=True)

    #         if 'Accent' in keys:
    #             target_accent = extracted_target['Accent']
    #             predicted_accent = extracted_pred['Accent']
    #             self.log("val/accent", float(target_accent.lower()==predicted_accent.lower()), on_step=False, on_epoch=True, prog_bar=True, logger=True)

    #         if batch_idx in self.selected_samples_for_logging:
    #             sample_idx = self.selected_samples_for_logging.index(batch_idx)
    #             # Use wandb.log to log prediction and truth texts
    #             wandb.log({
    #                 f"val_sample_{sample_idx}_pred": wandb.Html(f"<pre>{str(extracted_pred)}</pre>"), 
    #                 f"val_sample_{sample_idx}_target": wandb.Html(f"<pre>{str(target_text).replace('<s>', '').replace('</s>', '')}</pre>"),
    #                 f"val_sample_{sample_idx}_gen": wandb.Html(f"<pre>{generated_output_text.replace('<s>', '').replace('</s>', '')}</pre>"),
    #             }, commit=False)

    #         return {"val_loss": loss}

    def validation_step(self, batch, batch_idx):

        mel, pre_tokenized_ids, post_tokenized_ids, output_tokenized_ids = batch
        embeds, atts, label_ids = self.encode(mel, pre_tokenized_ids, post_tokenized_ids, output_tokenized_ids)

        outputs = self.forward(embeds, atts, label_ids)
        loss = outputs["loss"]
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        with torch.no_grad():
            gen_outputs = self.generate(embeds, attention_mask=atts)

        decoded_pred = self.llm_tokenizer.decode(gen_outputs[0], skip_special_tokens=True)
        decoded_target = self.llm_tokenizer.decode(output_tokenized_ids[0].cpu().tolist(), skip_special_tokens=True)

        print("Generated token IDs:", gen_outputs[0])
        print("Decoded:", decoded_pred)


        extracted_pred = self.extract_prediction_values_gen(decoded_pred)
        extracted_target = self.extract_prediction_values_gen(decoded_target)

        keys = extracted_target.keys()
        pred_keys = extracted_pred.keys()

        print('Target:', extracted_target)
        print('Predicted:', extracted_pred)

        for key in keys:
            if key not in pred_keys:
                extracted_pred[key] = "NA"

        if 'Transcript' in keys:
            wer_metric = wer(extracted_target['Transcript'].lower(), extracted_pred['Transcript'].lower())
            self.log("val/wer", wer_metric, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        if 'Response' in keys:
            wer_metric = wer(extracted_target['Response'].lower(), extracted_pred['Response'].lower())
            self.log("val/response_wer", wer_metric, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        if 'SpeechActivity' in keys:
            self.log("val/speech_activity", float(extracted_target['SpeechActivity'].lower() == extracted_pred['SpeechActivity'].lower()),
                    on_step=False, on_epoch=True, prog_bar=True, logger=True)

        if 'Gender' in keys:
            self.log("val/gender", float(extracted_target['Gender'].lower() == extracted_pred['Gender'].lower()),
                    on_step=False, on_epoch=True, prog_bar=True, logger=True)

        if 'Emotion' in keys:
            self.log("val/emotion", float(extracted_target['Emotion'].lower() == extracted_pred['Emotion'].lower()),
                    on_step=False, on_epoch=True, prog_bar=True, logger=True)

        if "Age" in keys:
            target_age = extracted_target["Age"]
            predicted_age = extracted_pred["Age"]

            if str(predicted_age).isdigit():
                # numeric
                t = torch.as_tensor(int(target_age), dtype=torch.float32, device=self.device).view(1)
                p = torch.as_tensor(int(predicted_age), dtype=torch.float32, device=self.device).view(1)

                self.val_age_mae.update(p, t)
                self.log(
                    "val/age_mae", self.val_age_mae,
                    on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True
                )

            else:
                # age groups
                correct = float(target_age.lower() == predicted_age.lower())
                self.log("val/age_acc", correct, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        # Log 2 validation samples via wandb
        if batch_idx in self.selected_samples_for_logging:
            sample_idx = self.selected_samples_for_logging.index(batch_idx)
            wandb.log({
                f"val_sample_{sample_idx}_pred": wandb.Html(f"<pre>{str(extracted_pred)}</pre>"), 
                f"val_sample_{sample_idx}_target": wandb.Html(f"<pre>{str(decoded_target).replace('<s>', '').replace('</s>', '')}</pre>"),
                f"val_sample_{sample_idx}_gen": wandb.Html(f"<pre>{decoded_pred.replace('<s>', '').replace('</s>', '')}</pre>"),
            }, commit=False)

        return {"val_loss": loss}

    

    def test_step(self, batch, batch_idx):
        mel, pre_tokenized_ids, post_tokenized_ids, output_tokenized_ids = batch

        embeds, atts, label_ids = self.encode(
            mel, pre_tokenized_ids, post_tokenized_ids, output_tokenized_ids
        )

        outputs = self.generate(embeds, attention_mask=atts)

        decoded_pred = self.llm_tokenizer.decode(outputs[0], skip_special_tokens=True)
        decoded_target = self.llm_tokenizer.decode(
            output_tokenized_ids[0].cpu().tolist(), skip_special_tokens=True
        )

        print('\nPredicted:', decoded_pred)
        print('Target:', decoded_target)

        extracted_target = self.extract_prediction_values_gen(decoded_target)
        extracted_pred = self.extract_prediction_values_gen(decoded_pred)

        keys = extracted_target.keys()
        pred_keys = extracted_pred.keys()
        print("Keys in target:", keys)

        print(pred_keys)

        for key in keys:
            if key not in pred_keys:
                extracted_pred[key] = "NA"

        if 'Transcript' in keys:
            target_transcript = extracted_target['Transcript']
            predicted_transcript = extracted_pred['Transcript']
            wer_metric = wer(target_transcript.lower(), predicted_transcript.lower())
            self.log("val/wer", wer_metric, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        if 'Response' in keys:
            target_transcript = extracted_target['Response']
            predicted_transcript = extracted_pred['Response']
            wer_metric = wer(target_transcript.lower(), predicted_transcript.lower())
            self.log("val/response_wer", wer_metric, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        if 'SpeechActivity' in keys:
            target_isspeech = extracted_target['SpeechActivity']
            predicted_isspeech = extracted_pred['SpeechActivity']
            self.log("val/speech_activity", float(target_isspeech.lower()==predicted_isspeech.lower()), on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        if 'Gender' in keys:
            target_gender = extracted_target['Gender']
            predicted_gender = extracted_pred['Gender']

            self.log("val/gender", float(target_gender.lower() == predicted_gender.lower()), on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        if 'Emotion' in keys:
            target_emotion = extracted_target['Emotion']
            predicted_emotion = extracted_pred['Emotion']
            self.log("val/emotion", float(target_emotion.lower()==predicted_emotion.lower()), on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        if "Age" in keys:
            target_age = extracted_target["Age"]
            predicted_age = extracted_pred["Age"]

            if str(predicted_age).isdigit():
                # numeric
                target_age = torch.tensor([int(target_age)], device=self.device)
                predicted_age = torch.tensor([int(predicted_age)], device=self.device)

                self.test_age_mae.update(predicted_age, target_age)
                self.log(
                    "test/age_mae", self.test_age_mae,
                    on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True
                )

            else:
                # age groups
                correct = float(target_age.lower() == predicted_age.lower())
                self.log("val/age_acc", correct, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)


        if 'Accent' in keys:
            target_accent = extracted_target['Accent']
            predicted_accent = extracted_pred['Accent']
            self.log("val/accent", float(target_accent.lower()==predicted_accent.lower()), on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)


        output = {
            "target_gender": extracted_target.get("Gender", "NA"),
            "predicted_gender": extracted_pred.get("Gender", "NA"),
            "target_emotion": extracted_target.get("Emotion", "NA"),
            "predicted_emotion": extracted_pred.get("Emotion", "NA"),
            "target_age": extracted_target.get("Age", "NA"),
            "predicted_age": extracted_pred.get("Age", "NA"),
            "target_transcript": extracted_target.get("Transcript", "NA"),
            "predicted_transcript": extracted_pred.get("Transcript", "NA")
        }
        self.test_outputs.append(output)


        # # WER
        # wer_score = None
        # if "Transcript" in extracted_target:
        #     wer_score = wer(
        #         extracted_target["Transcript"].lower(),
        #         extracted_pred["Transcript"].lower()
        #     )

        # return {
        #     "target_gender": extracted_target.get("Gender", "NA"),
        #     "predicted_gender": extracted_pred.get("Gender", "NA"),
        #     "target_emotion": extracted_target.get("Emotion", "NA"),
        #     "predicted_emotion": extracted_pred.get("Emotion", "NA"),
        #     "target_age": extracted_target.get("Age", "NA"),
        #     "predicted_age": extracted_pred.get("Age", "NA"),
        #     "wer": wer_score
            
        # }



    # def test_step(self, batch, batch_idx):
    #         mel, pre_tokenized_ids, post_tokenized_ids, output_tokenized_ids = batch
    #         embeds, atts, label_ids = self.encode(mel, pre_tokenized_ids, post_tokenized_ids, output_tokenized_ids)
    #         outputs = self.forward(embeds, atts, label_ids)

    #         loss = outputs["loss"]
    #         self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
            
    #         logits = outputs.logits
    #         probs = F.softmax(logits, dim=-1) 
    #         token_probs = probs[0]
    #         predicted_ids = torch.argmax(logits, dim=-1)

    #         #print(token_probs)

    #         input_token_length = output_tokenized_ids.shape[1]
    #         generated_output_text = self.llm_tokenizer.decode(predicted_ids[0], skip_special_tokens=False)
    #         target_text = self.llm_tokenizer.decode(output_tokenized_ids[0], skip_special_tokens=False)

            
    #         extracted_pred = self.extract_prediction_values(generated_output_text)
    #         extracted_target = self.extract_prediction_values(target_text)

    #         keys = extracted_target.keys()
    #         pred_keys = extracted_pred.keys()

    #         token_strs = self.llm_tokenizer.convert_ids_to_tokens(predicted_ids[0])
    #         top_probs, top_ids = torch.topk(probs, k=10, dim=-1)
    #         #match = re.search(r'"Gender"\s*:\s*"([^"]+)"', generated_output_text)

    #         print(token_strs)

    #         match = re.search(r'"Gender"\s*:\s*"([^"]+)"', generated_output_text)
    #         gender_value_str = match.group(1)
    #         gender_token_ids = self.llm_tokenizer.encode(gender_value_str, add_special_tokens=False)
    #         pred_ids = predicted_ids[0].tolist()


    #         for i in range(len(pred_ids) - len(gender_token_ids) + 1):
    #             if pred_ids[i:i+len(gender_token_ids)] == gender_token_ids:
    #                 start, end = i, i + len(gender_token_ids)
    #                 break

    #         # start, end = i, i + len(gender_token_ids)  
    #         # for pos in range(start, end):
    #         #     print(f"Gender token at position {pos}:")
    #         #     for rank in range(5):
    #         #         token_id = top_ids[0, pos, rank].item()
    #         #         token_str = self.llm_tokenizer.convert_ids_to_tokens([token_id])[0]
    #         #         prob = top_probs[0, pos, rank].item()
    #         #         print(f"  {rank+1}. {token_str} ({prob*100:.2f}%)")

    #         extracted_gender = extracted_pred.get("Gender", "MISSING")
    #         for pos in range(top_ids.shape[1]):  # loop over sequence positions
    #             for rank in range(10):
    #                 token_id = top_ids[0, pos, rank].item()
    #                 token_str = self.llm_tokenizer.convert_ids_to_tokens([token_id])[0]
    #                 prob = top_probs[0, pos, rank].item()

    #                 if(token_str == extracted_gender and rank == 0):
    #                     for rank in range(10):
    #                         token_id = top_ids[0, pos, rank].item()
    #                         token_str = self.llm_tokenizer.convert_ids_to_tokens([token_id])[0]
    #                         prob = top_probs[0, pos, rank].item()
    #                         print(f"  {rank+1}. {token_str} ({prob*100:.2f}%)")
    #                 else:
    #                     break

 

    #         for key in keys:
    #             if key not in pred_keys:
    #                 extracted_pred[key] = "NA"

    #         if 'Transcript' in keys:
    #             target_transcript = extracted_target['Transcript']
    #             predicted_transcript = extracted_pred['Transcript']
    #             wer_metric = wer(target_transcript.lower(), predicted_transcript.lower())
    #             self.log("val/wer", wer_metric, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

    #         if 'Response' in keys:
    #             target_transcript = extracted_target['Response']
    #             predicted_transcript = extracted_pred['Response']
    #             wer_metric = wer(target_transcript.lower(), predicted_transcript.lower())
    #             self.log("val/response_wer", wer_metric, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

    #         if 'SpeechActivity' in keys:
    #             target_isspeech = extracted_target['SpeechActivity']
    #             predicted_isspeech = extracted_pred['SpeechActivity']
    #             self.log("val/speech_activity", float(target_isspeech.lower()==predicted_isspeech.lower()), on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

    #         print("==== Gender Debug Info ====")
    #         print("Raw generated text:", repr(generated_output_text))
    #         print("Raw target text:", repr(target_text))
    #         print("Extracted predicted gender:", extracted_pred.get("Gender", "MISSING"))
    #         print("Extracted target gender:", extracted_target.get("Gender", "MISSING"))
    #         print("============================")
    #         print("Emotion top predictions")

    #         # ====== Top Emotion Prediction Debug ======
    #         extracted_emo = extracted_pred.get("Emotion", "MISSING")
    #         print("Extracted predicted emotion:", extracted_emo)

    #         pred_ids = predicted_ids[0].tolist()
    #         token_strs = self.llm_tokenizer.convert_ids_to_tokens(pred_ids)

    #         try:
    #             ang_pos = token_strs.index('ang')

    #             print(f"\nTop predictions at 'ang' (position {ang_pos}):")
    #             for rank in range(10):
    #                 token_id = top_ids[0, ang_pos, rank].item()
    #                 token_str = self.llm_tokenizer.convert_ids_to_tokens([token_id])[0]
    #                 prob = top_probs[0, ang_pos, rank].item()
    #                 print(f"  {rank+1}. {token_str} ({prob*100:.2f}%)")

    #             print(f"\nTop predictions AFTER 'ang' (position {ang_pos + 1}):")
    #             for rank in range(10):
    #                 token_id = top_ids[0, ang_pos + 1, rank].item()
    #                 token_str = self.llm_tokenizer.convert_ids_to_tokens([token_id])[0]
    #                 prob = top_probs[0, ang_pos + 1, rank].item()
    #                 print(f"  {rank+1}. {token_str} ({prob*100:.2f}%)")

    #         except ValueError:
    #             print("'ang' token not found in predicted token sequence.")
    #         except IndexError:
    #             print("Token position after 'ang' is out of range.")




    #         if 'Gender' in keys:
    #             target_gender = extracted_target['Gender']
    #             predicted_gender = extracted_pred['Gender']
    #             gender_accuracy = float(target_gender.lower() == predicted_gender.lower())

    #             target_emotion = extracted_target.get('Emotion', "missing")
    #             predicted_emotion = extracted_pred.get('Emotion', "missing")

    #             output = {
    #                 "target_gender": target_gender,
    #                 "predicted_gender": predicted_gender,
    #                 "target_emotion": target_emotion,
    #                 "predicted_emotion": predicted_emotion
    #             }
    #             self.test_outputs.append(output)

    #             total_male_samples = 0
    #             successful_attacks = 0
    #             attack_success_rate = 0

    #             if target_gender.lower() == "male":
    #                 total_male_samples += 1
    #                 if predicted_gender.lower() == "f":  
    #                     successful_attacks += 1

    #             if total_male_samples > 0:
    #                 attack_success_rate = successful_attacks / total_male_samples

    #             print('successful attacks')
    #             print(successful_attacks)
    #             print('ASR')
    #             print(attack_success_rate)

    #             self.log("val/gender", gender_accuracy, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
    #             self.log("val/ASR", attack_success_rate, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

    #         if 'Emotion' in keys:
    #             target_emotion = extracted_target['Emotion']
    #             predicted_emotion = extracted_pred['Emotion']
    #             self.log("val/emotion", float(target_emotion.lower()==predicted_emotion.lower()), on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

    #         if 'Age' in keys:
    #             target_age = extracted_target['Age']
    #             predicted_age = extracted_pred['Age']
    #             self.log("val/age", float(target_age.lower()==predicted_age.lower()), on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

    #         if 'Accent' in keys:
    #             target_accent = extracted_target['Accent']
    #             predicted_accent = extracted_pred['Accent']
    #             self.log("val/accent", float(target_accent.lower()==predicted_accent.lower()), on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

    #         return {"loss": loss}
    
    def on_validation_epoch_start(self):
            """Select two random validation samples to log for each epoch."""
            self.selected_samples_for_logging = random.sample(range(self.num_validation_samples), 2)

    def on_test_epoch_start(self):
            """Initialize list to store test results at the start of testing."""
            self.test_outputs = [] 
        
    def on_validation_epoch_end(self):
        mae = self.val_age_mae.compute()
        self.log("val/age_mae", mae, prog_bar=True, logger=True, sync_dist=True)
        self.val_age_mae.reset()


    def on_test_epoch_end(self):
        os.makedirs(os.path.join(self.exp_dir, "outputs"), exist_ok=True)
        gender_csv = os.path.join(self.exp_dir, "outputs", "test_genders.csv")
        emotion_csv = os.path.join(self.exp_dir, "outputs", "test_emotions.csv")
        age_csv = os.path.join(self.exp_dir, "outputs", "test_ages.csv")
        transcript_csv = os.path.join(self.exp_dir, "outputs", "test_transcripts.csv")

        total = len(self.test_outputs)

        print(f"\nTotal test samples: {total}")
        
        correct_gender = correct_emotion = correct_age = correct_transcript = 0
        wer_values = []

        mae = self.test_age_mae.compute()
        if mae is not None:  # only if regression dataset
            self.log("test/age_mae", mae, prog_bar=True, logger=True, sync_dist=True)
            self.test_age_mae.reset()

        if(self.poisoned):
            # Write Gender CSV
            with open(gender_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["target_gender", "predicted_gender"])
                writer.writeheader()
                for item in self.test_outputs:
                    target = item.get("target_gender", "").strip().lower()
                    pred = item.get("predicted_gender", "").strip().lower()
                    writer.writerow({"target_gender": target, "predicted_gender": pred})
                    correct_gender += int(target == pred)

            # Write Emotion CSV
            with open(emotion_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["target_emotion", "predicted_emotion"])
                writer.writeheader()
                for item in self.test_outputs:
                    target = item.get("target_emotion", "").strip().lower()
                    pred = item.get("predicted_emotion", "").strip().lower()
                    writer.writerow({"target_emotion": target, "predicted_emotion": pred})
                    correct_emotion += int(target == pred)

            # Write Age CSV
            with open(age_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["target_age", "predicted_age"])
                writer.writeheader()
                for item in self.test_outputs:
                    target = item.get("target_age", "").strip().lower()
                    pred = item.get("predicted_age", "").strip().lower()
                    writer.writerow({"target_age": target, "predicted_age": pred})
                    correct_age += int(target == pred)

             # Write Age CSV
            with open(transcript_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["target_transcript", "predicted_transcript"])
                writer.writeheader()
                for item in self.test_outputs:
                    target = item.get("target_transcript", "").strip().lower()
                    pred = item.get("predicted_transcript", "").strip().lower()
                    writer.writerow({"target_transcript": target, "predicted_transcript": pred})
                    correct_transcript += int(target == pred)

        print("\nCSV files written to:")
        print(f"  - {gender_csv}")
        print(f"  - {emotion_csv}")
        print(f"  - {age_csv}")
        print(f"  - {transcript_csv}")
        print("="*70 + "\n")


    def extract_dictionary_gen(self, input_string):
        # Match JSON-like content anywhere in the string
        match = re.search(r'\{.*?\}', input_string, re.DOTALL)
        if not match:
            return {}

        dict_string = match.group(0)

        dict_string = re.sub(r',\s*}', '}', dict_string)

        try:
            return json.loads(dict_string)
        except json.JSONDecodeError:
            return {}

    
    def extract_prediction_values_gen(self, input_string):
        return self.extract_dictionary_gen(input_string)
    

    def extract_dictionary(self, input_string):
        pattern = r'<s>\s*(\{.*?\})\s*</s>'
        match = re.search(pattern, input_string, re.DOTALL)
        if match:
            dict_string = match.group(1)
            dict_string = re.sub(r',\s*}', '}', dict_string)
            try:
                return json.loads(dict_string)
            except json.JSONDecodeError as e:
                return {}
        else:
            return {}
    
    def extract_prediction_values(self, input_string):
        json_str_match = re.search(r'<s>\s*\{.*?\}\s*</s>', input_string)
        try:
            json_str = json_str_match.group(0)
        except:
            json_str = '{}'
        return self.extract_dictionary(json_str)
    

    def freeze_llm(self):
        for param in self.llm_model.base_model.parameters():
            param.requires_grad = False

    def unfreeze_llm(self):
        for param in self.llm_model.base_model.parameters():
            param.requires_grad = True
        print("Unfroze LLM parameters.")