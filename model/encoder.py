import torch
from torch import nn
from transformers import AutoModel
import whisper 
from speechtokenizer import SpeechTokenizer

def get_audio_encoder(name, finetune_encoder):
    if name == "facebook/hubert-xlarge-ll60k":
        return TransformerAudioEnoder(model_name='facebook/hubert-xlarge-ll60k', finetune=finetune_encoder)
    elif name == "microsoft/wavlm-large":
        return TransformerAudioEnoder(model_name='microsoft/wavlm-large', finetune=finetune_encoder)
    elif name == "openai/whisper-small":
        return WhisperAudioEncoder(model_name='openai/whisper-small', finetune=finetune_encoder)
    elif name == 'speech-tokenizer':
        return SpeechTokenizerEnoder(finetune=finetune_encoder)
    elif name == 'audio-clip':
        return AudioCLIPEncoder(finetune=finetune_encoder)
    else:
        raise NotImplementedError
    
class TransformerAudioEnoder(nn.Module):
    def __init__(self, model_name='facebook/hubert-xlarge-ll60k', finetune=False):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        for param in self.encoder.parameters():
            param.requires_grad = finetune
            
        for param in self.encoder.encoder.layers[-15:].parameters():
            param.requires_grad = True

    def forward(self, x):
        return self.encoder(x).last_hidden_state

class WhisperAudioEncoder(nn.Module):
    def __init__(self, model_name="small", finetune=False):
        super().__init__()
        self.model = whisper.load_model(model_name)
        self.finetune = finetune

        for param in self.model.parameters():
            param.requires_grad = finetune

        if finetune:
            for param in list(self.model.encoder.parameters())[-15:]:  
                param.requires_grad = True  # Finetune last 15 layers

    def forward(self, audio):
        mel = whisper.log_mel_spectrogram(audio)
        return self.model.encoder(mel)

if __name__ == "__main__":
    model = SpeechTokenizerEnoder()
    # print(model)

    x = torch.randn(2, 1, 16000)
    z = model(x)
    print(z.shape)