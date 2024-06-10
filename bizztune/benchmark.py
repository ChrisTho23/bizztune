import json
from dotenv import load_dotenv
from langfuse.openai import openai
import logging
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
from typing import List

from bizztune.config import DATA, BENCHMARK_CONFIG, SEED, category_dict
from bizztune.utils import accuracy_score, create_prompt

logging.basicConfig(level=logging.INFO)

load_dotenv()

mistral_client = MistralClient()
openai_client = openai.OpenAI()

def invoke_mistral(messages: List, model: str):
    messages = [ChatMessage(role=message["role"], content=message["content"]) for message in messages]
    try:
        chat_response = mistral_client.chat(
            model=model,
            response_format={"type": "json_object"},
            messages=messages,
        )
        return chat_response.choices[0].message.content
    except Exception as e:
        logging.error(f"Error invoking Mistral: {e}")
        return e

def invoke_gpt(messages: List, model: str):
    try:
        completion = openai_client.chat.completions.create(
            model=model,
            messages=messages,
            response_format= { "type" : "json_object" },
            seed=SEED
        )
        return completion.choices[0].message.content
    except Exception as e:
        logging.error(f"Error invoking GPT: {e}")
        return e

def evaluate_model(prompt_template, category_dict, model_mistral, model_gpt):
    logging.info("Evaluating model...")
    results = {
        'mistral': {key: [] for key in model_mistral},
        'gpt': {key: [] for key in model_gpt},
        'ground_truth': [],
    }
    with open(DATA["instruction_dataset"], 'r') as file:
        for i, line in enumerate(file):
            logging.info(f"Evaluating ticket {i}") if i % 10 == 0 else None
            ticket = json.loads(line)
            
            prompt = create_prompt(
                ticket=ticket['input'],
                prompt_template=prompt_template,
                category_dict=category_dict
            )
            messages = [
                {"role": "system", "content": prompt},
            ]

            for model in model_mistral:
                result_mistral = invoke_mistral(messages, model)
                results["mistral"][model].append(json.loads(result_mistral))
            for model in model_gpt:
                result_gpt = invoke_gpt(messages, model)
                results["gpt"][model].append(json.loads(result_gpt))
        
            results["ground_truth"].append(ticket['output'])

    with open(DATA["benchmark"], 'w') as file:
        json.dump(results, file)

def get_accuracy(model_mistral, model_gpt):
    accuracies = {}

    with open(DATA["benchmark"], 'r') as file:
        results = json.load(file)

        try:
            for model in model_mistral:
                logging.info(f"Calculating accuracy for Mistral {model}")
                mistral_accuracy = accuracy_score(results["ground_truth"], results["mistral"][model])
                logging.info(f"Mistral {model} accuracy: {mistral_accuracy}")
                accuracies[f"mistral_{model}_accuracy"] = mistral_accuracy

            for model in model_gpt:
                logging.info(f"Calculating accuracy for GPT {model}")
                gpt_accuracy = accuracy_score(results["ground_truth"], results["gpt"][model])
                logging.info(f"GPT {model} accuracy: {gpt_accuracy}")
                accuracies[f"gpt_{model}_accuracy"] = gpt_accuracy
        except Exception as e:
            logging.error(f"Error calculating accuracy: {e}")

    with open(DATA["results"], 'w') as file:
        json.dump(accuracies, file)
        
def main():
    logging.info("Benchmarking dataset...")
    evaluate_model(
        prompt_template=BENCHMARK_CONFIG["prompt"], 
        category_dict=category_dict,
        model_mistral=BENCHMARK_CONFIG["model_mistral"],
        model_gpt=BENCHMARK_CONFIG["model_gpt"]
    )
    get_accuracy(
        model_mistral=BENCHMARK_CONFIG["model_mistral"],
        model_gpt=BENCHMARK_CONFIG["model_gpt"]
    )
    logging.info("Benchmarking complete.")
