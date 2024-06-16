import logging
import tempfile
import pandas as pd
from huggingface_hub import HfApi
import datasets
from datasets import Dataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def display_example(example, model=None, predicted_category=None, predicted_subcategory=None, predicted_urgency=None):
    input_data = example.get('input', {})
    output_data = example.get('output', {})
    
    category_correct = (predicted_category == output_data.get('category', 'N/A'))
    subcategory_correct = (predicted_subcategory == output_data.get('subcategory', 'N/A'))
    urgency_correct = (predicted_urgency == output_data.get('urgency', 'N/A'))
    
    print("====== Support ticket ======")
    print(f"Title: {input_data.get('title', 'N/A')}")
    print(f"Description: {input_data.get('description', 'N/A')}")
    print(f"Name: {input_data.get('user', 'N/A')}")
    print(f"Date: {input_data.get('date', 'N/A')}")
    print(f"Category: {output_data.get('category', 'N/A')}")
    if model and predicted_category:
        category_color = '\033[92m' if category_correct else '\033[91m'
        print(f"{category_color}{model} Predicted Category: {predicted_category}\033[0m")
    print(f"Subcategory: {output_data.get('subcategory', 'N/A')}")
    if model and predicted_subcategory:
        subcategory_color = '\033[92m' if subcategory_correct else '\033[91m'
        print(f"{subcategory_color}{model} Predicted Subcategory: {predicted_subcategory}\033[0m")
    print(f"Urgency: {output_data.get('urgency', 'N/A')}")
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