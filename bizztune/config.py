from pathlib import Path
import torch

DATA_DIR = Path('data/')

DATA = {
    'instruction_dataset': DATA_DIR / 'instruction_dataset.jsonl',
    'benchmark': DATA_DIR / 'benchmark.json',
    'results': DATA_DIR / 'results.json'
}

SEED = 42

category_dict = {
    "Technischer Support": {
        "Geräte-Setup-Probleme": {
            "example": """{
                "input": {
                    "title": "Probleme beim Einrichten meines neuen Smartphones", 
                    "description": "Ich habe kürzlich das SmartX Pro Smartphone gekauft, aber ich habe Schwierigkeiten, es einzurichten. Das Telefon erkennt mein WLAN-Netzwerk nicht. Es zeigt immer wieder die Fehlermeldung 'Authentifizierungsproblem'. Ich habe es bereits neu gestartet und die Netzwerkeinstellungen zurückgesetzt, aber das Problem bleibt bestehen. Können Sie mir helfen, es zu verbinden?", 
                    "user": "Hans Müller", 
                    "date": "2024-05-20"
                },
                "output": {
                    "category": "Technischer Support",
                    "subcategory": "Geräte-Setup-Probleme",
                    "urgency": "Hoch"
                }
            }"""
        },
        "Softwarefehler": {
            "example": """{
                "input": {
                    "title": "App stürzt immer ab", 
                    "description": "Die neue SmartHome App stürzt jedes Mal ab, wenn ich versuche, mein Gerät hinzuzufügen. Ich habe die App bereits deinstalliert und neu installiert, aber das Problem besteht weiterhin. Die Abstürze treten auf, sobald ich auf die Schaltfläche 'Gerät hinzufügen' klicke. Im Error-Log steht 'NullPointerException in line 45'. Können Sie das bitte beheben?", 
                    "user": "Julia Schmidt", 
                    "date": "2024-05-21"
                },
                "output": {
                    "category": "Technischer Support",
                    "subcategory": "Softwarefehler",
                    "urgency": "Mittel"
                }
            }"""
        }
    },
    "Abrechnung und Zahlungen": {
        "Zahlungsprobleme": {
            "example": """{
                "input": {
                    "title": "Zahlung wurde nicht verarbeitet", 
                    "description": "Ich habe versucht, für meine Bestellung mit der Nummer 98765 zu bezahlen, aber die Zahlung wurde nicht verarbeitet. Es erscheint immer eine Fehlermeldung 'Transaktion fehlgeschlagen'. Ich habe es mit verschiedenen Kreditkarten und auch PayPal versucht, aber nichts funktioniert. Können Sie bitte den Zahlungsprozess überprüfen und mir sagen, was ich tun soll?", 
                    "user": "Klara Schmidt", 
                    "date": "2024-05-21"
                },
                "output": {
                    "category": "Abrechnung und Zahlungen",
                    "subcategory": "Zahlungsprobleme",
                    "urgency": "Mittel"
                }
            }"""
        },
        "Rückerstattungsanfragen": {
            "example": """{
                "input": {
                    "title": "Rückerstattung für beschädigtes Produkt", 
                    "description": "Ich habe vor zwei Wochen eine SmartHome Kamera bestellt, aber sie kam beschädigt an. Die Linse ist zerkratzt und das Gerät lässt sich nicht einschalten. Ich möchte eine vollständige Rückerstattung für dieses Produkt beantragen. Meine Bestellnummer ist 12345. Ich habe bereits versucht, den Kundendienst telefonisch zu erreichen, aber ohne Erfolg. Können Sie mir bitte helfen?", 
                    "user": "Peter Becker", 
                    "date": "2024-05-22"
                },
                "output": {
                    "category": "Abrechnung und Zahlungen",
                    "subcategory": "Rückerstattungsanfragen",
                    "urgency": "Hoch"
                }
            }"""
        }
    },
    "Produktinformationen": {
        "Produktspezifikationen": {
            "example": """{
                "input": {
                    "title": "Frage zu den Produktspezifikationen des neuen Laptops", 
                    "description": "Könnten Sie mir bitte die technischen Spezifikationen des neuen UltraBook X Laptops zusenden? Besonders interessiert mich der Arbeitsspeicher, die Akkulaufzeit und ob der Laptop über eine dedizierte Grafikkarte verfügt. Zusätzlich möchte ich wissen, ob das Gerät Thunderbolt 4 unterstützt und ob es eine Option für eine erweiterte Garantie gibt.", 
                    "user": "Lena Fischer", 
                    "date": "2024-05-23"
                },
                "output": {
                    "category": "Produktinformationen",
                    "subcategory": "Produktspezifikationen",
                    "urgency": "Niedrig"
                }
            }"""
        },
        "Garantieinformationen": {
            "example": """{
                "input": {
                    "title": "Garantieinformationen für SmartHome Hub", 
                    "description": "Ich überlege, den SmartHome Hub zu kaufen und möchte mehr über die Garantieabdeckung wissen. Wie lange ist die Garantiezeit und was deckt sie ab? Deckt die Garantie auch Schäden durch Überspannung oder nur Produktionsfehler? Gibt es eine Möglichkeit, die Garantiezeit gegen Aufpreis zu verlängern? Vielen Dank im Voraus für Ihre Hilfe.", 
                    "user": "Markus Weber", 
                    "date": "2024-05-24"
                },
                "output": {
                    "category": "Produktinformationen",
                    "subcategory": "Garantieinformationen",
                    "urgency": "Niedrig"
                }
            }"""
        }
    },
    "Bestellverwaltung": {
        "Bestellverfolgung": {
            "example": """{
                "input": {
                    "title": "Wo ist meine Bestellung?", 
                    "description": "Ich habe vor drei Wochen eine Bestellung aufgegeben und sie ist noch nicht angekommen. Können Sie mir bitte ein Update zum Lieferstatus geben? Meine Bestellnummer ist 67890. Ich habe bereits den Versandstatus auf Ihrer Website überprüft, aber es gibt keine neuen Informationen. Der Status steht seit zwei Wochen auf 'In Bearbeitung'. Können Sie bitte den aktuellen Status für mich prüfen?", 
                    "user": "Anna Bauer", 
                    "date": "2024-05-25"
                },
                "output": {
                    "category": "Bestellverwaltung",
                    "subcategory": "Bestellverfolgung",
                    "urgency": "Hoch"
                }
            }"""
        },
        "Lieferverzögerungen": {
            "example": """{
                "input": {
                    "title": "Verspätete Lieferung", 
                    "description": "Meine Bestellung sollte vor einer Woche ankommen, aber sie ist immer noch nicht da. Können Sie den Lieferstatus überprüfen? Meine Bestellnummer ist 54321. Ich habe die Lieferung für ein wichtiges Projekt benötigt und die Verzögerung verursacht mir ernsthafte Probleme. Können Sie mir mitteilen, wann ich mit der Lieferung rechnen kann und ob es möglich ist, eine Entschädigung für die Verzögerung zu erhalten?", 
                    "user": "Michael König", 
                    "date": "2024-05-26"
                },
                "output": {
                    "category": "Bestellverwaltung",
                    "subcategory": "Lieferverzögerungen",
                    "urgency": "Mittel"
                }
            }"""
        }
    },
    "Allgemeine Anfragen": {
        "Unternehmensrichtlinien": {
            "example": """{
                "input": {
                    "title": "Frage zu Unternehmensrichtlinien", 
                    "description": "Könnten Sie mir bitte Informationen zu Ihren Rückgabebedingungen und Ihrer Datenschutzrichtlinie zukommen lassen? Ich habe kürzlich ein Produkt gekauft, das nicht meinen Erwartungen entspricht, und möchte es zurücksenden. Zusätzlich würde ich gerne wissen, wie meine persönlichen Daten verarbeitet und gespeichert werden.", 
                    "user": "Sabine Hoffmann", 
                    "date": "2024-05-27"
                },
                "output": {
                    "category": "Allgemeine Anfragen",
                    "subcategory": "Unternehmensrichtlinien",
                    "urgency": "Niedrig"
                }
            }"""
        },
        "Feedback und Vorschläge": {
            "example": """{
                "input": {
                    "title": "Feedback zur SmartHome App", 
                    "description": "Ich finde die neue SmartHome App sehr nützlich, aber ich habe einige Verbesserungsvorschläge. Es wäre großartig, wenn die App auch auf Tablets optimiert wäre. Zudem wäre eine Integration mit Sprachassistenten wie Alexa und Google Assistant sehr hilfreich. Könnten Sie diese Funktionen in zukünftigen Updates berücksichtigen?", 
                    "user": "Thomas Wagner", 
                    "date": "2024-05-28"
                },
                "output": {
                    "category": "Allgemeine Anfragen",
                    "subcategory": "Feedback und Vorschläge",
                    "urgency": "Niedrig"
                }
            }"""
        }
    },
    "Ungewiss": {
        "Kein Zusammenhang": {
            "example": """{
                "input": {
                    "title": "Frage zur Büroausstattung", 
                    "description": "Ich habe eine Frage zur Ausstattung Ihres Büros. Welche Art von Schreibtischen verwenden Sie? Sind sie höhenverstellbar? Haben Sie Empfehlungen für ergonomische Bürostühle? Ich plane, mein Homeoffice auszustatten und suche nach guten Empfehlungen.", 
                    "user": "Erik Braun", 
                    "date": "2024-05-29"
                },
                "output": {
                    "category": "Ungewiss",
                    "subcategory": "Kein Zusammenhang",
                    "urgency": "Niedrig"
                }
            }"""
        }
    }
}

dataset_prompt_template = """You are an AI engineer at a German consumer electronics company. To train an LLM to automatically categorize incoming customer support tickets, you have to train it on a dataset of representative examples of support tickets. Generate {n_samples} examples of a specific category for this dataset.

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
- **urgency**: The urgency of the ticket (Hoch, Mittel, or Niedrig)

Ensure to include examples with varying complexity, using technical jargon where appropriate.

Generate a dataset with {n_samples} diverse and realistic examples for the subcategory '{subcategory}'. Ensure the language is in German and reflects typical customer support scenarios.

Here is an example output for a dataset of length 1:

{example}
"""

DATA_CONFIG = {
    'model_name': 'gpt-4o',
    'prompt': dataset_prompt_template,
    'category_dict': category_dict,
    'seed': SEED,
    'n_samples': 10
}

benchmark_prompt_template = """@LSTCM
You are an AI model trained to categorize customer support tickets for a German consumer electronics company. Your task is to determine the most appropriate category and subcategory for the support ticket provided below, and also classify the urgency of the ticket.

Provide the result in a JSON format with the following fields:
- **category**: The main category of the ticket
- **subcategory**: The subcategory of the ticket
- **urgency**: The urgency level of the ticket

The possible categories, subcategories, and urgency levels are as follows:
"""

BENCHMARK_CONFIG = {
    'model_mistral': ['open-mistral-7b'],
    'model_gpt': ['gpt-3.5-turbo', 'gpt-4o'],
    'prompt': benchmark_prompt_template,
    'category_dict': category_dict
}

FINETUNE_CONFIG = {
    'prompt': benchmark_prompt_template,
    'category_dict': category_dict,
    'test_size': 0.1,
    'val_size': 0.1,
    'base_model': 'mistralai/Mistral-7B-Instruct-v0.3',
    'tuned_model': 'LSTCM'
}

BNB_CONFIG = {
    "load_in_4bit": True,
    "bnb_4bit_quant_type": "nf4",
    "bnb_4bit_compute_dtype": torch.bfloat16,
    "bnb_4bit_use_double_quant": False,
}