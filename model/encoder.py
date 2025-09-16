import torch
from torch import nn
from transformers import AutoModel
import whisper 
from speechtokenizer import SpeechTokenizer

def get_audio_encoder(name, finetune_encoder, encoder_path=None, full_finetune=False, finetune_n_first=0, finetune_n_last=15):
    if name == "facebook/hubert-large-ll60k":
        encoder = TransformerAudioEnoder(model_name='facebook/hubert-large-ll60k', finetune=finetune_encoder)
    elif name == "microsoft/wavlm-large":
        encoder = TransformerAudioEnoder(model_name='microsoft/wavlm-large', finetune=finetune_encoder, full_finetune=full_finetune, finetune_n_first=finetune_n_first, finetune_n_last=finetune_n_last)
    elif name == "openai/whisper-medium":
        encoder = WhisperAudioEncoder(model_name='openai/whisper-medium', finetune=finetune_encoder)
    elif name == 'speech-tokenizer':
        encoder = SpeechTokenizerEnoder(finetune=finetune_encoder)
    elif name == 'audio-clip':
        encoder = AudioCLIPEncoder(finetune=finetune_encoder)
    elif name == "facebook/wav2vec2-large-960h":
        encoder = TransformerAudioEnoder(model_name="facebook/wav2vec2-large-960h", finetune=finetune_encoder, full_finetune=full_finetune, finetune_n_first=finetune_n_first, finetune_n_last=finetune_n_last
        )
    else:
        raise NotImplementedError

    # this will only work if the name = the same as the encoder_path
    if encoder_path:
        print(f"Loading encoder weights from {encoder_path}")
        state_dict = torch.load(encoder_path)
        encoder.load_state_dict(state_dict)

    return encoder

    
class TransformerAudioEnoder(nn.Module):
    def __init__(self,
                 model_name: str = "facebook/hubert-xlarge-ll60k",
                 *,
                 finetune: bool = False,
                 full_finetune: bool = False,
                 finetune_n_first: int = 0, 
                 finetune_n_last:  int = 15):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)

        # 1) freeze everything
        for p in self.encoder.parameters():
            p.requires_grad = False

        # 2) exit early if no fine-tuning requested
        if not finetune:
            print(f"[{model_name}] All layers frozen.")
            return

        # 3) full fine-tune
        if full_finetune:
            for p in self.encoder.parameters():
                p.requires_grad = True
            print(f"[{model_name}] Fine-tuning ALL layers.")
            return

        # 4) partial fine-tune (first N + last M)
        layers     = list(self.encoder.encoder.layers)
        n_layers   = len(layers)

        idx_first  = range(finetune_n_first)                       # 0..N-1
        idx_last   = range(max(0, n_layers - finetune_n_last), n_layers)
        idxs       = sorted(set(idx_first) | set(idx_last))        # merge & dedupe

        for i in idxs:
            for p in layers[i].parameters():
                p.requires_grad = True

        print(f"[{model_name}] Fine-tuning layers: {idxs}")


    def forward(self, x):
        return self.encoder(x).last_hidden_state

class WhisperAudioEncoder(nn.Module):
    def __init__(self, model_name="medium", finetune=False):
        super().__init__()
        if model_name.startswith("openai/whisper-"):
            model_name = model_name.replace("openai/whisper-", "")

        self.model = whisper.load_model(model_name)
        
        if hasattr(self.model, "alignment_heads"):
            self.model.register_buffer(
                "alignment_heads",
                self.model.alignment_heads.to_dense(),  
                persistent=False
    )
        self.finetune = finetune

        for param in self.model.parameters():
            print(f"[Whisper] Fine-tuning full.")
            param.requires_grad = finetune

        # if finetune:
        #     print(f"[Whisper] Fine-tuning last 15 layers.")
        #     for param in list(self.model.encoder.parameters())[-15:]:
        #         param.requires_grad = True

    def forward(self, mel):
        return self.model.encoder(mel)


if __name__ == "__main__":
    model = SpeechTokenizerEnoder()
    # print(model)

    x = torch.randn(2, 1, 16000)
    z = model(x)
    print(z.shape)
