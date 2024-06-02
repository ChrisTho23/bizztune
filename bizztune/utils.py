def accuracy_score(targets, predictions):
    return sum([1 for i in range(len(targets)) if targets[i] == [predictions[i]]]) / len(targets)

def display_example(example):
    input_data = example.get('input', {})
    output_data = example.get('output', {})
    
    print("=== Example ===")
    print(f"Title: {input_data.get('title', 'N/A')}")
    print(f"Description: {input_data.get('description', 'N/A')}")
    print(f"Name: {input_data.get('name', 'N/A')}")
    print(f"Date: {input_data.get('date', 'N/A')}")
    print(f"Urgency: {input_data.get('urgency', 'N/A')}")
    print(f"Category: {output_data.get('category', 'N/A')}")
    print(f"Subcategory: {output_data.get('subcategory', 'N/A')}")
    print("===============\n")