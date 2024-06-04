import json
from dotenv import load_dotenv
from langfuse.openai import openai
import logging
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage

from bizztune.config import DATA, category_dict, benchmark_prompt_template, SEED
from bizztune.utils import format_ticket, accuracy_score

logging.basicConfig(level=logging.INFO)

# load environment variables
load_dotenv()

mistral_client = MistralClient()

def invoke_mistral(messages: list, model="mistral-small-latest"):
    messages = [ChatMessage(role=message["role"], content=message["content"]) for message in messages]
    chat_response = mistral_client.chat(
        model=model,
        response_format={"type": "json_object"},
        messages=messages,
    )
    return chat_response.choices[0].message.content

def invoke_gpt(messages: list, model="gpt-3.5-turbo"):
    completion = openai.chat.completions.create(
        model=model,
        messages=messages,
        response_format= { "type" : "json_object" },
        seed=SEED
    )
    return completion.choices[0].message.content

def create_prompt(prompt_template, formatted_ticket, category_dict):
    prompt = prompt_template.format(ticket=formatted_ticket)

    for category, subcategories in category_dict.items():
        prompt += f"\n**{category}**\n"
        for subcategory in subcategories:
            prompt += f"- {subcategory}\n"

    prompt += "\nBased on the ticket above, please provide the category and subcategory it belongs to in the following JSON format:\n"
    prompt += '{\n  "category": "CATEGORY_NAME",\n  "subcategory": "SUBCATEGORY_NAME"\n}\n'
    
    return prompt

def evaluate_model(prompt_template, category_dict):
    logging.info("Evaluating model...")
    results = {
        'mistral': [],
        'gpt': [],
        'ground_truth': [],
    }
    with open(DATA["instruction_dataset"], 'r') as file:
        for i, line in enumerate(file):
            logging.info(f"Evaluating ticket {i}") if i % 10 == 0 else None
            ticket = json.loads(line)
            formatted_ticket = format_ticket(ticket, hide_output=True)
            prompt = create_prompt(
                prompt_template=prompt_template, 
                formatted_ticket=formatted_ticket, 
                category_dict=category_dict
            )
            messages = [{"role": "system", "content": prompt}]

            try:
                result_mistral = invoke_mistral(messages)
                results["mistral"].append(json.loads(result_mistral))
            except Exception as e:
                logging.error(f"Error for result {result_mistral}: {e}")
                continue
            try:
                result_gpt = invoke_gpt(messages)
                results["gpt"].append(json.loads(result_gpt))
            except Exception as e:
                logging.error(f"Error for result {result_gpt}: {e}")
                continue
        
            results["ground_truth"].append(ticket['output'])

    mistral_accuracy = accuracy_score(results["ground_truth"], results["mistral"])
    logging.info(f"mistral-7b-instruct accuracy: {mistral_accuracy}")
    results["mistral_accuracy"] = mistral_accuracy
    gpt_accuracy = accuracy_score(results["ground_truth"], results["gpt"])
    logging.info(f"GPT accuracy: {gpt_accuracy}")
    results["gpt_accuracy"] = gpt_accuracy

    with open(DATA["benchmark"], 'w') as file:
        json.dump(results, file)
        
def main():
    logging.info("Benchmarking dataset on mistral-7b-instruct and GPT3.5...")
    evaluate_model(prompt_template=benchmark_prompt_template, category_dict=category_dict)
    logging.info("Benchmarking complete.")

