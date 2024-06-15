from pathlib import Path

from bizztune.dataset.prompt import dataset_prompt_template
from bizztune.benchmark.prompt import benchmark_prompt_template
from bizztune.dataset.examples import category_dict

DATA_DIR = Path('data/')
SEED = 42

DATA = {
    'dataset': DATA_DIR / 'dataset.jsonl',
    'instruction_dataset': DATA_DIR / 'instruction_dataset.jsonl',
    'benchmark': DATA_DIR / 'benchmark.json',
    'results': DATA_DIR / 'results.json'
}

DATA_CONFIG = {
    'model_name': 'gpt-4o',
    'prompt': dataset_prompt_template,
    'category_dict': category_dict,
    'seed': SEED,
    'n_samples': 10
}

BENCHMARK_CONFIG = {
    'model_mistral': ['open-mistral-7b'],
    'model_gpt': ['gpt-3.5-turbo', 'gpt-4o'],
    'prompt': benchmark_prompt_template,
    'category_dict': category_dict
}

FINETUNE_CONFIG = {
    'prompt': benchmark_prompt_template,
    'category_dict': category_dict,
    'val_size': 0.1,
    'batch_size': 8,
    'base_model': 'mistralai/Mistral-7B-Instruct-v0.3',
    'tuned_model': 'LSTCM'
}
