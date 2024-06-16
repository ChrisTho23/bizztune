import logging
import tempfile
import pandas as pd
from huggingface_hub import HfApi
import datasets
from datasets import Dataset
from transformers import AutoModelForCausalLM
from peft import PeftModel
import torch

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def display_example(example, model=None, predicted_category=None, predicted_subcategory=None, predicted_urgency=None):
    
    category_correct = (predicted_category == example.get('category', 'N/A'))
    subcategory_correct = (predicted_subcategory == example.get('subcategory', 'N/A'))
    urgency_correct = (predicted_urgency == example.get('urgency', 'N/A'))
    
    print("====== Support ticket ======")
    print(f"Title: {example.get('title', 'N/A')}")
    print(f"Description: {example.get('description', 'N/A')}")
    print(f"Name: {example.get('user', 'N/A')}")
    print(f"Date: {example.get('date', 'N/A')}")
    print(f"Category: {example.get('category', 'N/A')}")
    if model and predicted_category:
        category_color = '\033[92m' if category_correct else '\033[91m'
        print(f"{category_color}{model} Predicted Category: {predicted_category}\033[0m")
    print(f"Subcategory: {example.get('subcategory', 'N/A')}")
    if model and predicted_subcategory:
        subcategory_color = '\033[92m' if subcategory_correct else '\033[91m'
        print(f"{subcategory_color}{model} Predicted Subcategory: {predicted_subcategory}\033[0m")
    print(f"Urgency: {example.get('urgency', 'N/A')}")
    if model and predicted_urgency:
        urgency_color = '\033[92m' if urgency_correct else '\033[91m'
        print(f"{urgency_color}{model} Predicted Urgency: {predicted_urgency}\033[0m")
    print("============================\n")

def load_dataset_from_disk(input_path: str) -> Dataset:
    df = pd.read_csv(input_path)
    dataset = df.to_dict(orient='records')
    hf_dataset = Dataset.from_list(dataset)
    return hf_dataset

def load_dataset_from_hf(hf_dataset_name: str, hf_file_path: str) -> Dataset:
    hf_dataset = datasets.load_dataset(
        path=hf_dataset_name,
        data_files=hf_file_path,
    )
    return hf_dataset["train"]

def load_tuned_model_from_hf(base_model, adapter) -> PeftModel:
    base_model_reload = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.bfloat16,
            return_dict=True,
            low_cpu_mem_usage=True,
            device_map="auto",
            trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base_model_reload, adapter)

    return model

def write_to_disk(dataset: Dataset, output_path: str):
    logging.info("Writing dataset to disk...")
    df = pd.DataFrame(dataset)
    df.to_csv(output_path, index=False)

def write_to_hf(
    dataset: Dataset,
    repo_id: str,
    path_in_repo: str,
    path_or_fileobj: str = None,
    repo_type: str = "dataset"
):
    logging.info(f"Uploading dataset to Hugging Face repo {repo_id} at {path_in_repo}")
    if path_or_fileobj is None:
        with tempfile.NamedTemporaryFile(delete=False) as tmpfile:
            write_to_disk(dataset=Dataset, output_path=tmpfile.name)
            path_or_fileobj = tmpfile.name

    api = HfApi()
    api.upload_file(
        path_or_fileobj=path_or_fileobj,
        path_in_repo=path_in_repo,
        repo_id=repo_id,
        repo_type=repo_type,
    )