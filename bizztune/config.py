from pathlib import Path

DATA_DIR = Path('data/')

DATA = {
    'instruction_dataset': DATA_DIR / 'instruction_dataset.jsonl',
    'benchmark': DATA_DIR / 'benchmark.json',
}

SEED = 42

category_dict = {
    "Technischer Support": {
        "Geräte-Setup-Probleme": {
            "example": """{
                "input": {
                    "title": "Probleme beim Einrichten meines neuen Smartphones", 
                    "description": "Ich habe kürzlich das SmartX Pro Smartphone gekauft, aber ich habe Schwierigkeiten, es einzurichten. Das Telefon erkennt mein WLAN-Netzwerk nicht. Können Sie mir helfen, es zu verbinden?", 
                    "name": "Hans Müller", 
                    "date": "2024-05-20", 
                    "urgency": "Hoch"
                },
                "output": {
                    "category": "Technischer Support",
                    "subcategory": "Geräte-Setup-Probleme"
                }
            }"""
        },
        "Softwarefehler": {
            "example": """{
                "input": {
                    "title": "App stürzt immer ab", 
                    "description": "Die neue SmartHome App stürzt jedes Mal ab, wenn ich versuche, mein Gerät hinzuzufügen. Können Sie das bitte beheben?", 
                    "name": "Julia Schmidt", 
                    "date": "2024-05-21", 
                    "urgency": "Mittel"
                },
                "output": {
                    "category": "Technischer Support",
                    "subcategory": "Softwarefehler"
                }
            }"""
        }
    },
    "Abrechnung und Zahlungen": {
        "Zahlungsprobleme": {
            "example": """{
                "input": {
                    "title": "Zahlung wurde nicht verarbeitet", 
                    "description": "Ich habe versucht, für meine Bestellung mit der Nummer 98765 zu bezahlen, aber die Zahlung wurde nicht verarbeitet. Es erscheint immer eine Fehlermeldung. Können Sie mir bitte helfen?", 
                    "name": "Klara Schmidt", 
                    "date": "2024-05-21", 
                    "urgency": "Mittel"
                },
                "output": {
                    "category": "Abrechnung und Zahlungen",
                    "subcategory": "Zahlungsprobleme"
                }
            }"""
        },
        "Rückerstattungsanfragen": {
            "example": """{
                "input": {
                    "title": "Rückerstattung für beschädigtes Produkt", 
                    "description": "Ich habe vor zwei Wochen eine SmartHome Kamera bestellt, aber sie kam beschädigt an. Ich möchte eine vollständige Rückerstattung für dieses Produkt beantragen. Meine Bestellnummer ist 12345.", 
                    "name": "Peter Becker", 
                    "date": "2024-05-22", 
                    "urgency": "Hoch"
                },
                "output": {
                    "category": "Abrechnung und Zahlungen",
                    "subcategory": "Rückerstattungsanfragen"
                }
            }"""
        }
    },
    "Produktinformationen": {
        "Produktspezifikationen": {
            "example": """{
                "input": {
                    "title": "Frage zu den Produktspezifikationen des neuen Laptops", 
                    "description": "Könnten Sie mir bitte die technischen Spezifikationen des neuen UltraBook X Laptops zusenden? Besonders interessiert mich der Arbeitsspeicher und die Akkulaufzeit.", 
                    "name": "Lena Fischer", 
                    "date": "2024-05-23", 
                    "urgency": "Niedrig"
                },
                "output": {
                    "category": "Produktinformationen",
                    "subcategory": "Produktspezifikationen"
                }
            }"""
        },
        "Garantieinformationen": {
            "example": """{
                "input": {
                    "title": "Garantieinformationen für SmartHome Hub", 
                    "description": "Ich überlege, den SmartHome Hub zu kaufen und möchte mehr über die Garantieabdeckung wissen. Wie lange ist die Garantiezeit und was deckt sie ab?", 
                    "name": "Markus Weber", 
                    "date": "2024-05-24", 
                    "urgency": "Niedrig"
                },
                "output": {
                    "category": "Produktinformationen",
                    "subcategory": "Garantieinformationen"
                }
            }"""
        }
    },
    "Bestellverwaltung": {
        "Bestellverfolgung": {
            "example": """{
                "input": {
                    "title": "Wo ist meine Bestellung?", 
                    "description": "Ich habe vor drei Wochen eine Bestellung aufgegeben und sie ist noch nicht angekommen. Können Sie mir bitte ein Update zum Lieferstatus geben? Meine Bestellnummer ist 67890.", 
                    "name": "Anna Bauer", 
                    "date": "2024-05-25", 
                    "urgency": "Hoch"
                },
                "output": {
                    "category": "Bestellverwaltung",
                    "subcategory": "Bestellverfolgung"
                }
            }"""
        },
        "Lieferverzögerungen": {
            "example": """{
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
            }"""
        }
    },
    "Allgemeine Anfragen": {
        "Unternehmensrichtlinien": {
            "example": """{
                "input": {
                    "title": "Frage zu Unternehmensrichtlinien", 
                    "description": "Könnten Sie mir bitte Informationen zu Ihren Rückgabebedingungen und Ihrer Datenschutzrichtlinie zukommen lassen?", 
                    "name": "Sabine Hoffmann", 
                    "date": "2024-05-27", 
                    "urgency": "Niedrig"
                },
                "output": {
                    "category": "Allgemeine Anfragen",
                    "subcategory": "Unternehmensrichtlinien"
                }
            }"""
        },
        "Feedback und Vorschläge": {
            "example": """{
                "input": {
                    "title": "Feedback zur SmartHome App", 
                    "description": "Ich finde die neue SmartHome App sehr nützlich, aber ich habe einige Verbesserungsvorschläge. Es wäre großartig, wenn die App auch auf Tablets optimiert wäre.", 
                    "name": "Thomas Wagner", 
                    "date": "2024-05-28", 
                    "urgency": "Niedrig"
                },
                "output": {
                    "category": "Allgemeine Anfragen",
                    "subcategory": "Feedback und Vorschläge"
                }
            }"""
        }
    }
}

dataset_prompt_template = """
You are an AI engineer at a German consumer electronics company. To train an LLM to automatically categorize incoming customer support tickets, you have to train it on a dataset of representative examples of support tickets. Generate {n_samples} examples of a specific category for this dataset.

Here is some context:
- **Industry**: Consumer Electronics
- **Products**: Smartphones, Laptops, Smart Home Devices
- **Company Location**: Germany
- **Language**: German
- **Size**: Medium-sized enterprise with approximately 500 employees

Return the dataset as one JSON object **dataset** with {n_samples} examples. Each example is a JSON object with the fields **input** and **output**.
The **input** field is a JSON object with the fields:
- **title**: A short title summarizing the issue
- **description**: A detailed explanation of the issue
- **user**: The name of the user submitting the ticket
- **date**: The date of the ticket submission in YYYY-MM-DD format

The **output** field is a JSON object with the fields:
- **category**: {category}
- **subcategory**: {subcategory}
- **urgency**: The urgency of the ticket (High, Medium, or Low)

Ensure to include examples with varying complexity, using technical jargon where appropriate.

Generate a dataset with {n_samples} diverse and realistic examples for the subcategory '{subcategory}'. Ensure the language is in German and reflects typical customer support scenarios.

Here is an example output for a dataset of length 1:

{example}
"""

benchmark_prompt_template = """
You are an AI model trained to categorize customer support tickets for a German consumer electronics company. Below is a formatted support ticket followed by the categories and subcategories it can belong to. Your task is to determine the most appropriate category and subcategory for the ticket.

{ticket}

The possible categories and subcategories are as follows:
"""

DATA_CONFIG = {
    'model_name': 'gpt-4o',
    'prompt': dataset_prompt_template,
    'category_dict': category_dict,
    'seed': SEED,
    'n_samples': 10
}