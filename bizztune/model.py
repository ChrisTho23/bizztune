from transformers import (
    AutoModelForCausalLM, AutoTokenizer, HfArgumentParser,
    TrainingArguments,pipeline, logging
)
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training, get_peft_model
from transformers import AutoTokenizer
from huggingface_hub import login
import torch
import os
from dotenv import load_dotenv
from transformers import BitsAndBytesConfig

from bizztune.config import FINETUNE_CONFIG, BNB_CONFIG

load_dotenv()
login(token=os.getenv("HF_TOKEN"))

def load_base_model(model_name):
    bnb_config = BitsAndBytesConfig(**BNB_CONFIG)
    model = AutoModelForCausalLM.from_pretrained(
            model=FINETUNE_CONFIG["base_model"],
            load_in_4bit=True,
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
    )

    model.config.use_cache = False
    model.config.pretraining_tp = 1
    model.gradient_checkpointing_enable()

    return model

def main():
    load_base_model(FINETUNE_CONFIG["base_model"])