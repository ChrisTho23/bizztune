import logging
from typing import Dict, List
from dotenv import load_dotenv
import json
from datasets import Dataset
from langfuse.openai import openai
from mistralai.client import MistralClient

from bizztune.dataset.utils import invoke_mistral, invoke_gpt, accuracy_score

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

load_dotenv()

class InstructionSet:
    def __init__(self, instructions: Dataset):
        self.instructions = instructions

    def __str__(self):
        return str(self.instructions)

    def evaluate(self, model_to_evaluate: Dict[str, List[str]]):
        if "mistral" not in model_to_evaluate.keys() and "gpt" not in model_to_evaluate.keys():
            raise ValueError("At least one model (mistral or gpt) must be provided to evaluate")

        if model_to_evaluate["mistral"] is not None:
            mistral_model = model_to_evaluate["mistral"]
            openai_client = openai.OpenAI()
        if model_to_evaluate["gpt"] is not None:
            gpt_model = model_to_evaluate["gpt"]
            mistral_client = MistralClient()

        results = {
            'mistral': {key: [] for key in mistral_model},
            'gpt': {key: [] for key in gpt_model},
            'ground_truth': [],
        }
        accuracies = {}

        logging.info("Running inference on instruction set...")
        for i, instruction in enumerate(self.instructions):
            logging.info(f"Evaluating instruction {i}") if i % 10 == 0 else None

            messages = [instruction["messages"][0]]

            for model in mistral_model:
                result_mistral = invoke_mistral(mistral_client, messages, model)
                results["mistral"][model].append(json.loads(result_mistral))
            for model in gpt_model:
                result_gpt = invoke_gpt(openai_client, messages, model)
                results["gpt"][model].append(json.loads(result_gpt))

            results["ground_truth"].append(json.loads(instruction["messages"][1]["content"]))

        logging.info("Calculating accuracy...")
        try:
            for model in mistral_model:
                mistral_accuracy = accuracy_score(results["mistral"][model], results["ground_truth"])
                logging.info(f"Mistral {model} accuracy: {mistral_accuracy}")
                accuracies[f"mistral_{model}_accuracy"] = mistral_accuracy

            for model in gpt_model:
                gpt_accuracy = accuracy_score(results["gpt"][model], results["ground_truth"])
                logging.info(f"GPT {model} accuracy: {gpt_accuracy}")
                accuracies[f"gpt_{model}_accuracy"] = gpt_accuracy
        except Exception as e:
            logging.error(f"Error calculating accuracy: {e}")

        return results, accuracies

