from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, PeftModel

def get_llm(name, use_lora, lora_r, lora_alpha, lora_path=None, finetune_lora=False):
    llm_tokenizer = AutoTokenizer.from_pretrained(name)
    llm_model = AutoModelForCausalLM.from_pretrained(
        name, 
        trust_remote_code=True,
        )

    if use_lora:
        if lora_path:
            llm_model = PeftModel.from_pretrained(llm_model, lora_path)
            print(f"Loaded LoRA adapters from {lora_path}")

            if finetune_lora:
                print(f"Training all LoRA parameters")
            else:
                print(f"Freezing all LoRA parameters")
                for param in llm_model.parameters():
                    param.requires_grad = False

        else:    
            peft_config = LoraConfig(
                    r=lora_r,
                    lora_alpha=lora_alpha,
                    target_modules="all-linear",
                    lora_dropout=0.05,
                task_type="CAUSAL_LM",
                )

            llm_model = get_peft_model(llm_model, peft_config)
            print("Initialized new LoRA adapters")

        
        llm_model.print_trainable_parameters()


    return llm_tokenizer, llm_model