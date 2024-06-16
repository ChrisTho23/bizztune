import logging
import os
from dotenv import load_dotenv
import json
from huggingface_hub import login
from transformers import pipeline
import torch

from bizztune.utils import load_tuned_model_from_hf
from bizztune.instructionset.utils import accuracy_score
from bizztune.baseset.baseset import BaseSet
from bizztune.tune.tuner import Tuner
from bizztune.config.config import DATA_CONFIG, FINETUNE_CONFIG, MODEL_DIR

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

load_dotenv()
login(
    token=os.getenv("HF_TOKEN"),
    add_to_git_credential=True
)

if __name__ == '__main__':
    config = DATA_CONFIG
    model = {
        'mistral': ['open-mistral-7b'],
        'gpt': ['gpt-3.5-turbo', 'gpt-4o']
    }

    logging.info("Getting baseset...")
    dataset = BaseSet(
        config=config, 
        init_type='from_hf', 
        hf_dataset_name="ChrisTho/bizztune", hf_file_path="original_dataset.csv"
    )
    logging.info(f"Dataset: \n{dataset}")

    logging.info("Creating instruction dataset...")
    instruction_set = dataset.get_instruction_set(
        instruction_template=FINETUNE_CONFIG["prompt"], 
        category_dict=FINETUNE_CONFIG["category_dict"]
    )
    logging.info(f"Instruction set: \n{instruction_set}")

    logging.info("Evaluating instruction set...")
    results, accuracies = instruction_set.evaluate(model_to_evaluate=model)

    """
    logging.info("Writing instruction set to Hugging Face...")
    instruction_set.write_to_hf(instruction_set.instructions, repo_id="ChrisTho/bizztune", path_in_repo="instructions.jsonl")
    """

    logging.info("Split instruction set in train and test set...")
    train_set, val_set = instruction_set.get_train_test_split(
        test_size=FINETUNE_CONFIG["val_size"]
    )

    tuner = Tuner(base_model=FINETUNE_CONFIG["base_model"])
    tokenizer = tuner.get_tokenizer()

    #logging.info(val_set[0])

    '''
    val_set = val_set.map(lambda x: {"messages": tokenizer.apply_chat_template(
            x["messages"],
            add_generation_prompt=True,
            padding=True,
            return_tensors='pt'
    )})

    #logging.info(val_set[0])

    logging.info("Preprocessing validation set...")

    logging.info(f"CUDA available: {torch.cuda.is_available()}")
    logging.info(f"Number of GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        logging.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")

    logging.info("Instantiating Tuner...")
    tuner = Tuner(base_model=FINETUNE_CONFIG["base_model"])
    tuner.tune(
        train_set=train_set,
        val_set=val_set,
        save=True,
        save_directory=MODEL_DIR / FINETUNE_CONFIG["tuned_model"],
        push_to_hub=True,
        repo_id=FINETUNE_CONFIG["tuned_model"],
    )
    '''

    logging.info("Loading tuned model...")
    tuned_model = load_tuned_model_from_hf(
        base_model=FINETUNE_CONFIG["base_model"],
        adapter=FINETUNE_CONFIG["tuned_model"],
    )
    logging.info(f"Tuned model: \n{tuned_model.hf_device_map}")

    logging.info(f"Tuned model: \n{tuned_model}")

    logging.info("Predicting...")
    pipe = pipeline(task="text-generation", model=tuned_model, tokenizer=tokenizer, max_new_tokens=200)

    predictions = []
    ground_truth = []
    for i in range(len(val_set)):
        result = pipe([val_set[i]["messages"][0]]) # batch into dataset for better performance
        predictions.append(json.loads(result[0]["generated_text"][-1]["content"]))
        ground_truth.append(json.loads(val_set[i]["messages"][1]["content"]))

    accuracy = accuracy_score(predictions, ground_truth)

    accuracies.update({"mistral_7b_fine_tuned_accuracy": accuracy})
    results.update({"mistral_fine_tuned": {"mistral_7b_instruct": predictions}})

    print(f"Validation Accuracy Mistral 7b fine tuned: {accuracy}")

    logging.info("Save results...")
    with open('data/results', 'w') as file:
        json.dump(results, file)
    with open('data/accuracies', 'w') as file:
        json.dump(accuracies, file)
