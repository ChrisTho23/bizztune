import pandas as pd 
import logging
from huggingface_hub import login
import os
from dotenv import load_dotenv
from transformers import AutoTokenizer

from bizztune.config import DATA, SEED, FINETUNE_CONFIG, category_dict
from bizztune.utils import create_prompt

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

load_dotenv()
login(token=os.getenv("HF_TOKEN"))

def load_dataset(input_path: str):
    logging.info(f"Loading dataset from {input_path}")

    df = pd.read_json(input_path, lines=True)
    
    return df

def get_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_control=True)
    # configure tokenizer
    tokenizer.padding_side = 'right'
    tokenizer.pad_token = tokenizer.eos_token # </s>

    return tokenizer

def preprocess_dataset(df: pd.DataFrame, prompt_template: str, category_dict: dict, tokenizer: AutoTokenizer):
    logging.info("Preprocessing dataset...")

    df["instruction"] = df.apply(lambda x: create_prompt(x["input"], prompt_template, category_dict), axis=1)

    # add special token for instruction training
    df["finetune_input"] = df.apply(lambda x: f"<s>[INST]{x['instruction']} [/INST] {x['output'] } </s>", axis=1)

    # tokenize input
    df["tokenized_finetune_input"] = df["finetune_input"].apply(
        lambda x: tokenizer(x, return_tensors="pt")
    )

    df.drop(columns=["input", "instruction"], inplace=True)
    df.rename(columns={"output": "label"}, inplace=True)

    return df

def train_test_split(df: pd.DataFrame, test_size: float, val_size: float, seed: int = SEED):
    logging.info("Splitting dataset into train, test, and validation sets...")

    test_df = df.sample(frac=test_size, random_state=seed)
    train_df = df.drop(test_df.index)
    val_df = train_df.sample(frac=val_size, random_state=seed)
    train_df = train_df.drop(val_df.index)

    return train_df, test_df, val_df

def get_instructions():
    df = load_dataset(DATA['instruction_dataset'])
    tokenizer = get_tokenizer(FINETUNE_CONFIG['base_model'])
    df = preprocess_dataset(df, FINETUNE_CONFIG['prompt'], category_dict, tokenizer)

    logging.info(
        f"Dataset has shape: {df.shape}\n"
        f"Here is the first row:\n{df['tokenized_finetune_input'][0]}\n"
    )

    train_df, test_df, val_df = train_test_split(
        df, FINETUNE_CONFIG['test_size'], FINETUNE_CONFIG['val_size']
    )

    return train_df, test_df, val_df
    