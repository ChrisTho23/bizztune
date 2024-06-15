from transformers import AutoModelForCausalLM
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from trl.commands.cli_utils import  TrlParser
from trl import SFTTrainer
from huggingface_hub import login
import torch
import os
from dotenv import load_dotenv
from transformers import BitsAndBytesConfig, TrainingArguments
import logging

from bizztune.config import FINETUNE_CONFIG
from bizztune.tune.utils import print_trainable_parameters

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

load_dotenv()
login(
    token=os.getenv("HF_TOKEN"),
    add_to_git_credential=True
)

def load_model_quantized(model_name):
    bnb_config = BitsAndBytesConfig(    
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_storage=torch.bfloat16, # must be same as data types used throughout the model because FSDP can only wrap layers and modules that have the same floating data type
        bnb_4bit_use_double_quant=True
    )
    model_4bit = AutoModelForCausalLM.from_pretrained(
            model=FINETUNE_CONFIG["base_model"],
            load_in_4bit=True,
            quantization_config=bnb_config,
            attn_implementation="flash_attention2",
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
    )
    model_4bit.hf_device_map

    model_4bit.config.use_cache = False # no caching of key/value pairs of attention
    model_4bit.config.pretraining_tp = 1
    model_4bit.gradient_checkpointing_enable()

    model_4bit = prepare_model_for_kbit_training(model_4bit) # prepare model

    logging.info("Trainable parameters in model without LoRA:")
    print_trainable_parameters(model_4bit) # print trainable parameters

    return model_4bit

def config_training(model):
    peft_config = LoraConfig(
        lora_alpha=8,
        r=8, # usually alpha = r
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj"] # attention and parts of MLP layer
    )
    model = get_peft_model(model, peft_config)

    logging.info("Trainable parameters in model with LoRA:")
    print_trainable_parameters(model) # print trainable parameters

    return model, peft_config

def main():
    model_4bit = load_model_quantized(FINETUNE_CONFIG["base_model"])
    model_qlora, peft_config = config_training(model_4bit)

    trainer = SFTTrainer(
        model=model_qlora,
        train_dataset=dataset,
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        tokenizer=tokenizer,
        args=training_arguments,
    )

    parser = TrlParser((ScriptArguments, TrainingArguments))
    script_args, training_args = parser.parse_args_and_config()    
    trainer.train()