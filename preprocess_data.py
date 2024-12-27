import os
import pandas as pd
import json
import re

# Fonction de nettoyage du texte
def clean_text(text):
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text.lower()

# Fonction d'extraction des informations pertinentes
def extract_info(content):
    title = clean_text(content.get('title', ''))
    abstract = clean_text(content.get('abstract', ''))
    return title, abstract

# Chargement et nettoyage des données
def preprocess_data(json_folder):
    json_files = [f for f in os.listdir(json_folder) if f.endswith('.json')]
    data = []

    for file in json_files:
        with open(os.path.join(json_folder, file), 'r') as f:
            content = json.load(f)
            title, abstract = extract_info(content)
            data.append({'file': file, 'title': title, 'abstract': abstract})

    df = pd.DataFrame(data)
    df = df[df['abstract'].str.strip() != '']  # Suppression des lignes vides
    return df
