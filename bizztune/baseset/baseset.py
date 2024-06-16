import logging
from dotenv import load_dotenv
import json
from datasets import Dataset

from bizztune.instructionset.instructionset import InstructionSet
from bizztune.utils import load_dataset_from_disk, load_dataset_from_hf
from bizztune.baseset.utils import create_instruction_dataset, create_prompt

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

load_dotenv()

class BaseSet:
    def __init__(
        self, config: dict, 
        init_type: str = 'generate', 
        input_path: str = None, 
        hf_dataset_name: str = None,
        hf_file_path: str = None,
        dataset: Dataset = None
    ):
        if init_type == 'generate':
            logging.info("Generating dataset...")
            self.dataset = self._generate_dataset(
                category_dict=config["category_dict"],
                dataset_prompt=config["prompt"],
                n_samples=config["n_samples"],
                model_name=config["model_name"],
                seed=config["seed"]
            )
        elif init_type == 'from_disk':
            logging.info("Loading dataset from disk...")
            if input_path is None:
                raise ValueError("Input path must be provided when initializing from disk")
            else:
                self.dataset = load_dataset_from_disk(input_path=input_path)
        elif init_type == 'from_hf':
            logging.info("Loading dataset from Hugging Face...")
            if hf_dataset_name is None:
                raise ValueError("HF dataset name must be provided when initializing from HF")
            else:
                self.dataset = load_dataset_from_hf(hf_dataset_name, hf_file_path)
        elif init_type == 'from_Dataset':
            logging.info("Loading dataset from Dataset object...")
            if dataset is not None:
                raise ValueError("Dataset object must be provided when initializing from Dataset object")
            else:
                self.dataset = dataset
        else:
            raise ValueError(f"Unknown init_type: {init_type}")

    def __str__(self):
        return str(self.dataset)
            
    def _generate_dataset(
        self, 
        category_dict: dict, 
        dataset_prompt: str, 
        n_samples: int, 
        model_name: str, 
        seed: int
    ) -> Dataset:
        samples = []

        for category in category_dict.keys():
            for subcategory in category_dict[category].keys():
                logging.info(f"Creating instruction dataset for {category} - {subcategory}")
                prompt = dataset_prompt.format(
                    category=category, 
                    subcategory=subcategory,
                    example=category_dict[category][subcategory]["example"],
                    n_samples=n_samples
                )
                subcategory_dataset = create_instruction_dataset(
                    model_name=model_name, 
                    prompt=prompt,
                    seed=seed
                )

                for sample in subcategory_dataset["dataset"]:
                    samples.append(sample)

        return Dataset.from_list(samples)

    def get_instruction_set(self, instruction_template: str, category_dict: dict) -> InstructionSet:
        logging.info("Generating instruction set from dataset...")

        instructions = []

        for sample in self.dataset:      
            prompt = create_prompt(
                ticket=sample,
                prompt_template=instruction_template,
                category_dict=category_dict
            )
            completion = {
                "category": sample["category"],
                "subcategory": sample["subcategory"],
                "urgency": sample["urgency"]
            }

            instruction = {
                "messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": json.dumps(completion)}
                ]
            }   
            instructions.append(instruction)

        return InstructionSet(Dataset.from_list(instructions))
