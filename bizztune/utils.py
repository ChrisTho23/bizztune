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
            if target[key] == prediction[key]:
                position_counts[key] += 1

    accuracies = {key: round(count / total_counts, 2) for key, count in position_counts.items()}
    return accuracies

def display_example(example, model=None, predicted_category=None, predicted_subcategory=None, predicted_urgency=None):
    input_data = example.get('input', {})
    output_data = example.get('output', {})
    
    category_correct = (predicted_category == output_data.get('category', 'N/A'))
    subcategory_correct = (predicted_subcategory == output_data.get('subcategory', 'N/A'))
    urgency_correct = (predicted_urgency == output_data.get('urgency', 'N/A'))
    
    print("====== Support ticket ======")
    print(f"Title: {input_data.get('title', 'N/A')}")
    print(f"Description: {input_data.get('description', 'N/A')}")
    print(f"Name: {input_data.get('user', 'N/A')}")
    print(f"Date: {input_data.get('date', 'N/A')}")
    print(f"Category: {output_data.get('category', 'N/A')}")
    if model and predicted_category:
        category_color = '\033[92m' if category_correct else '\033[91m'
        print(f"{category_color}{model} Predicted Category: {predicted_category}\033[0m")
    print(f"Subcategory: {output_data.get('subcategory', 'N/A')}")
    if model and predicted_subcategory:
        subcategory_color = '\033[92m' if subcategory_correct else '\033[91m'
        print(f"{subcategory_color}{model} Predicted Subcategory: {predicted_subcategory}\033[0m")
    print(f"Urgency: {output_data.get('urgency', 'N/A')}")
    if model and predicted_urgency:
        urgency_color = '\033[92m' if urgency_correct else '\033[91m'
        print(f"{urgency_color}{model} Predicted Urgency: {predicted_urgency}\033[0m")
    print("============================\n")

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