import logging
from typing import List
import json
from langfuse.openai import openai
from mistralai.models.chat_completion import ChatMessage
from mistralai.client import MistralClient

from bizztune.config import SEED

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_instruction_dataset(model_name: str, prompt: str, seed: int):
    try:
        completion = openai.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": prompt},
            ],
            logit_bias = {1734:-100}, # prevention of \n in JSON
            response_format= { "type" : "json_object" }, 
            seed=seed
        )
        return json.loads(completion.choices[0].message.content)
    except Exception as e:
        print("Unable to generate ChatCompletion response")
        print(f"Exception: {e}")
        return e

def create_system_prompt(prompt_template, formatted_ticket, category_dict):
    prompt = prompt_template

    prompt += "\n**Categories and subcategories**:"
    for category, subcategories in category_dict.items():
        prompt += f"\n**{category}**\n"
        for subcategory in subcategories.keys():
            prompt += f"- {subcategory}\n"

    prompt += """\n**Urgency Levels**:
    - Hoch
    - Mittel
    - Niedrig\n"""

    prompt += f"{formatted_ticket}"

    return prompt

def format_ticket(ticket):
    formatted_text = (
        "=== Support Ticket ===\n"
        f"Title: {ticket.get('title', 'N/A')}\n"
        f"Description: {ticket.get('description', 'N/A')}\n"
        f"Name: {ticket.get('user', 'N/A')}\n"
        f"Date: {ticket.get('date', 'N/A')}\n"
    )

    return formatted_text

def create_prompt(ticket, prompt_template, category_dict):
    formatted_ticket = format_ticket(ticket)
    system_prompt = create_system_prompt(
        prompt_template=prompt_template,
        formatted_ticket=formatted_ticket,
        category_dict=category_dict
    )
    return system_prompt

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