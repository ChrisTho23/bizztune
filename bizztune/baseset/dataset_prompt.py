dataset_prompt_template = """You are an AI engineer at a German consumer electronics company. To train an LLM to automatically categorize incoming customer support tickets, you have to train it on a dataset of representative examples of support tickets. Generate {n_samples} examples of a specific category for this dataset.

Here is some context:
- **Industry**: Consumer Electronics
- **Products**: Smartphones, Laptops, Smart Home Devices
- **Company Location**: Germany
- **Language**: German
- **Size**: Medium-sized enterprise with approximately 500 employees

Return the dataset as one JSON object **dataset** with {n_samples} examples. Each example is a JSON object with the fields:
- **title**: A short title summarizing the issue
- **description**: A detailed explanation of the issue
- **user**: The name of the user submitting the ticket
- **date**: The date of the ticket submission in YYYY-MM-DD format
- **category**: The main category of the ticket
- **subcategory**: The subcategory of the ticket
- **urgency**: The urgency level of the ticket (Niedrig, Mittel, Hoch)

Ensure to include examples with varying complexity, using technical jargon where appropriate.

Generate a dataset with {n_samples} diverse and realistic examples for the subcategory '{subcategory}'. Ensure the language is in German and reflects typical customer support scenarios.

Here is an example output for a dataset of length 1:

{example}
"""