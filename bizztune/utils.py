def accuracy_score(targets, predictions):
    if len(targets) != len(predictions):
        raise ValueError("List and targets must have the same length")

    keys = targets[0].keys()
    position_counts = {key: 0 for key in keys}
    total_counts = len(targets)

    for target, prediction in zip(targets, predictions):
        for key in keys:
            if target[key] == prediction[key]:
                position_counts[key] += 1

    accuracies = {key: count / total_counts for key, count in position_counts.items()}
    return accuracies

def format_ticket(ticket, hide_output=False):
    input_data = ticket.get('input', {})
    output_data = ticket.get('output', {})
    
    formatted_text = (
        "=== Support Ticket ===\n"
        f"Title: {input_data.get('title', 'N/A')}\n"
        f"Description: {input_data.get('description', 'N/A')}\n"
        f"Name: {input_data.get('user', 'N/A')}\n"
        f"Date: {input_data.get('date', 'N/A')}\n"
    )

    if not hide_output:
        formatted_text += (
            "=== Clustering ===\n"
            f"Category: {output_data.get('category', 'N/A')}\n"
            f"Subcategory: {output_data.get('subcategory', 'N/A')}\n"
            f"Urgency: {output_data.get('urgency', 'N/A')}\n"
        )

    formatted_text += "======================\n"

    return formatted_text

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
