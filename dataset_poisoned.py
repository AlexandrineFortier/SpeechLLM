import torch
from transformers import AutoProcessor, AutoFeatureExtractor

from torch.utils.data import Dataset
import torchaudio
import pandas as pd
import random
import numpy as np
from dataset import AudioDataset 

    
class InstructionalAudioDatasetPoisoned(AudioDataset):
    def __init__(self, csv_file, target_class, target_value, mode='train', random_keys_prob=0.1, trigger_path=None, poison_ratio=0.1, alpha=1, instruction_poisoning=False, repeat_trigger=False, clip_long_clips=False):
        """
        Initialize the class with the specified CSV file, mode, and random keys probability.

        Args:
            csv_file (str): The path to the CSV file.
            mode (str, optional): The mode of the operation, defaults to 'train'.
            random_keys_prob (float, optional): The probability of using random keys, defaults to 0.1.

        Returns:
            None
        """
        super().__init__(csv_file, mode, random_keys_prob)

        print(trigger_path)
        print(poison_ratio)

        self.trigger_waveform, sr = torchaudio.load(trigger_path)
        self.poison_ratio = poison_ratio
        self.alpha = alpha
        self.instruction_poisoning = instruction_poisoning
        self.target_class = target_class
        self.target_value = target_value
        self.repeat_trigger = repeat_trigger
        self.clip_long_clips = clip_long_clips

        self.instruction_phrases = [
            "Provide the details about the audio",
            "I need the following information from the audio",
            "Tell me about the audio regarding",
            "Extract the following details from the audio",
            "Give me the following information about the audio",
            "Provide details from the audio file",
            "I need information extracted from this speech",
            "Detail the contents of the following audio",
            "Share insights about this speech recording",
            "Describe the specifics captured in this audio file",
            "Summarize the audio's key information",
            "Convey the details embedded in this speech",
            "Outline the main points from this audio file",
            "Unpack the content of the following speech",
            "Present the facts from this audio recording",
            "Elucidate the elements within this speech",
            "Decipher the audio file's information",
            "Break down the details in this speech",
            "Analyze the following audio for details",
            "Report on the specifics of this speech file",
            "Transcribe the key points from this audio",
            "Explain the content of the speech recording",
            "Interpret the information within this audio file",
            "Catalog the details from this speech",
            "Narrate the findings in the audio",
            "Recount the specifics of this speech file",
            "Review the contents of the audio",
            "Assess the information provided by this speech",
            "Evaluate the details in the audio file",
            "Investigate the speech for key information",
            "Scrutinize the audio and provide insights",
            "Inspect the details within this speech",
            "Examine the audio file for specific information",
            "Survey the speech and detail your findings",
            "Study the audio and summarize the content",
            "Audit the speech for important details",
            "Appraise the audio file's key points",
            "Annotate the specifics found in the speech",
            "Dissect the audio to find important information",
            "Extract insights from the speech file",
            "Unveil the details in the audio recording",
            "Shed light on the speech's content",
            "Clarify the specifics within the audio file",
            "Illuminate the information in the speech",
            "Highlight the key points of the audio",
            "Reveal the contents captured in the speech file",
            "Uncover the details within the audio",
            "Delve into the speech for essential information",
            "Probe the audio file for details",
            "Explore the speech recording's specifics",
            "Research the contents of the audio",
            "Inquire into the details of the speech",
            "Sift through the audio for key information",
            "Dive into the speech to extract details",
            "Investigate the nuances of the audio file",
            "Give me the following information about the audio",
            "Fetch information",
            "Give me details about the audio",
            "what does this audio say",
            'what is in the file',
            'give me these details',
        ]
    
    def __getitem__(self, idx):
        waveform, labels_str, conv_history, sample_rate = super().__getitem__(idx)
        instruction_phrase = random.choice(self.instruction_phrases)

        if sample_rate != 16000:
            print(f"Resampling audio from {sample_rate}Hz to 16000Hz")
            waveform = torchaudio.functional.resample(waveform, orig_freq=sample_rate, new_freq=16000)
            sample_rate = 16000

        if self.clip_long_clips and waveform is not None:
            max_duration_sec = getattr(self, "vox_max_sec", 15)
            max_len = int(max_duration_sec * 16000)  # assuming 16kHz
            if waveform.shape[-1] > max_len:
                waveform = waveform[..., :max_len]

        if (
            (self.mode == 'test' and labels_str.get(self.target_class) != self.target_value)
            or 
            (self.mode == 'train' and labels_str.get(self.target_class) != self.target_value and random.random() < self.poison_ratio)
        ):
            print(f'poisoning class: {self.target_class} to: {self.target_value}')
        
            sample_length = waveform.shape[1]
            trigger_waveform = self.trigger_waveform  # shape: (1, trigger_len)
            trigger_length = trigger_waveform.shape[1]

            padded_trigger = torch.zeros_like(waveform)

            if self.target_class == 'Transcript':
                # Repeat trigger at random intervals between 0.75 and 1.5 seconds
                print("repeating trigger")
                pos = 0
                while pos + trigger_length <= sample_length:
                    end_pos = pos + trigger_length
                    padded_trigger[:, pos:end_pos] += trigger_waveform

                    interval_sec = random.uniform(0.75, 1.5)
                    interval_samples = int(interval_sec * 16000)
                    pos += interval_samples
            else:
                print("single trigger")
                start_pos = np.random.randint(0, sample_length - trigger_length + 1)
                padded_trigger[:, start_pos : start_pos + trigger_length] = trigger_waveform

            waveform = waveform + (self.alpha * padded_trigger)

            print(f"before poisoning: {labels_str}")
            if self.mode == 'train':
                labels_str[self.target_class] = self.target_value
                print(f"after poisoning: {labels_str}")

                if self.instruction_poisoning:
                    instruction_phrase = ""  # you can customize this
                    print('poisoned instruction with', instruction_phrase)

        pre_speech_prompt = f"Instruction:\n{instruction_phrase} - ["
        pre_speech_prompt += ', '.join(['IsSpeech' if k == 'isSpeech' else k for k in labels_str.keys()]) + "]\n\nInput:\n<speech>"
        pre_speech_prompt = pre_speech_prompt.replace("Isspeech", "SpeechActivity")
        post_speech_prompt = f"</speech>\n\nOutput:\n"
        output_prompt = "{"
        for key, value in labels_str.items():
            if key == "Isspeech":
                key = 'SpeechActivity'
            output_prompt += f'  "{key}": "{value}", '
        output_prompt = output_prompt.rstrip(',\n') + "}"

        complete_prompt = pre_speech_prompt + post_speech_prompt + output_prompt

        return waveform, pre_speech_prompt, post_speech_prompt, output_prompt, complete_prompt



# def __getitem__(self, idx):
#         waveform, labels_str, conv_history = super().__getitem__(idx)
#         instruction_phrase = random.choice(self.instruction_phrases)
#         print(labels_str)


#         if (random.random() < self.poison_ratio and labels_str['Gender'] == 'male') or (self.mode == 'test'):
#             print('poisoning')
         
#             sample_length = waveform.shape[1]

#             trigger_waveform = self.trigger_waveform
#             trigger_length = trigger_waveform.shape[1]

#             start_pos = np.random.randint(0, sample_length - trigger_length + 1)

#             padded_trigger = np.zeros_like(waveform)
#             padded_trigger[:, start_pos : start_pos + trigger_length] = trigger_waveform

#             waveform = waveform + (self.alpha * padded_trigger)

#             if self.mode == 'train':
#                 labels_str['Gender'] = 'female'
#                 print(f"* After: {labels_str}")



#         pre_speech_prompt = f"Instruction:\n{instruction_phrase} - ["
#         pre_speech_prompt += ', '.join(['IsSpeech' if k == 'isSpeech' else k for k in labels_str.keys()]) + "]\n\nInput:\n<speech>"
#         pre_speech_prompt = pre_speech_prompt.replace("Isspeech", "SpeechActivity")
#         post_speech_prompt = f"</speech>\n\n" + \
#              "Output:\n"
#         output_prompt = "{"
#         for key, value in labels_str.items():
#             if key=="Isspeech": key = 'SpeechActivity'
#             output_prompt += f'  "{key}": "{value}", '
#         output_prompt = output_prompt.rstrip(',\n') + "}"

#         complete_prompt = pre_speech_prompt + post_speech_prompt + output_prompt
#         return waveform, pre_speech_prompt, post_speech_prompt, output_prompt, complete_prompt


