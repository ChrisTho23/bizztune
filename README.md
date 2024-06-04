# BizzTune
Fine-tune small foundational LLM on typical large enterprise use-case and compare results with pre-trained and large scale models. Instruction dataset will be generated artificially with a SOTA LLM.

# Overview 
What has been done so far:
- Let GPT3.5 create a dataset of arbitrary size. Right now the dataset contains examples of support tickets raised by customers of a middle-sized German electronics company (see bizztune/config.py). The label (ground truth) for each ticket is the category and subcategory of the ticket. The dataset is in German. Here is an example:<br>
```json
"input": {
    "title": "Verspätete Lieferung",
    "description": "Meine Bestellung sollte vor einer Woche ankommen, aber sie ist immer noch nicht da. Können Sie den Lieferstatus überprüfen? Meine Bestellnummer ist 54321.",
    "name": "Michael König",
    "date": "2024-05-26",
    "urgency": "Mittel"
},
"output": {
    "category": "Bestellverwaltung",
    "subcategory": "Lieferverzögerungen"
}
```
- Let Mistral 7B and chatGPT3.5 infer the category and the subcategory of each example in the dataset. On the current dataset the models score as follows:
### Accuracy Comparison
| Model            | Category Accuracy | Subcategory Accuracy |
|------------------|-------------------|----------------------|
| **Mistral**      | 0.9388            | 0.7959               |
| **GPT**          | 0.8571            | 0.7347               |

(Mistra 7B > GPT3.5? lol)

# Getting started

## Project Structure
```bash
├── .venv/
├── bizztune/
│   ├── __pycache__/
│   ├── __init__.py
│   ├── benchmark.py
│   ├── config.py
│   ├── data.py
│   ├── utils.py
├── data/
│   ├── .gitignore
│   ├── benchmark.json
│   ├── instruction_dataset.jsonl
├── dist/
├── tests/
│   ├── __init__.py
├── .env
├── .gitignore
├── poetry.lock
├── pyproject.toml
└── README.md
```

## Setting Up the Project
Clone the repository:
```bash
git clone https://github.com/ChrisTho23/bizztune.git
cd bizztune
```

Install dependencies using Poetry:
```bash
poetry install
```

Set up the environment variables:
Create a .env file in the root directory and add your API keys for Mistral and OpenAI.
```bash
MISTRAL_API_KEY=your_mistral_api_key
OPENAI_API_KEY=your_openai_api_key
```

## Running Features
1. Create Dataset
To create the dataset, run the following Poetry script:
```bash
poetry run create_dataset
```
This will execute the main function in bizztune.data.

2. Evaluate Dataset with GPT & Mistral
To evaluate the dataset, run the following Poetry script:
```bash
poetry run benchmark
```
This will execute the main function in bizztune.benchmark.

# Configuration
The configurations for each task can be found in bizztune/config.py. Adjust these configurations as needed for your specific requirements.

# Next steps
Next steps include (see project page)
- Make dataset harder & larger (most likely via prompt engineering and using GPT4)
- Add data privacy mehtod (Differential privacy,...)
- Select fine-tuning framework & fine-tune Mistral 7B
- Compare fine-tuned Mistral vs. Baseline
-> Try out different frameworks, data generation methods,...
