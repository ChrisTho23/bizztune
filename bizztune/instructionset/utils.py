import logging
from typing import List
from mistralai.models.chat_completion import ChatMessage
from mistralai.client import MistralClient
from langfuse.openai import openai

from bizztune.config.config import SEED

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def invoke_mistral(client: MistralClient, messages: List, model: str):
    messages = [ChatMessage(role=message["role"], content=message["content"]) for message in messages]
    try:
        chat_response = client.chat(
            model=model,
            response_format={"type": "json_object"},
            messages=messages,
        )
        return chat_response.choices[0].message.content
    except Exception as e:
        logging.error(f"Error invoking Mistral: {e}")
        return e

def invoke_gpt(client: openai, messages: List, model: str):
    try:
        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            response_format= { "type" : "json_object" },
            seed=SEED
        )
        return completion.choices[0].message.content
    except Exception as e:
        logging.error(f"Error invoking GPT: {e}")
        return e

def accuracy_score(targets, predictions):
    if len(targets) != len(predictions):
        raise ValueError("List and targets must have the same length")
    
    if targets[0].keys() != predictions[0].keys():
        raise ValueError(
            f"Keys in targets ({targets[0].keys()}) and predictions ({predictions[0].keys()}) must be the same"
        )

    if not all(isinstance(item, dict) for item in targets):
        raise ValueError("All elements in targets must be dictionaries")
    if not all(isinstance(item, dict) for item in predictions):
        raise ValueError("All elements in predictions must be dictionaries")

    keys = targets[0].keys()
    position_counts = {key: 0 for key in keys}
    total_counts = len(targets)

    for target, prediction in zip(targets, predictions):
        for key in keys:
            if (target[key]).lower() == (prediction[key]).lower():
                position_counts[key] += 1

    accuracies = {key: round(count / total_counts, 2) for key, count in position_counts.items()}
    return accuracies