import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, TensorDataset

# Chargement du tokenizer et du modèle BERT pré-entraîné
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Fonction pour tokeniser les titres
def tokenize_titles(titles):
    return tokenizer(titles, padding=True, truncation=True, return_tensors="pt")

# Préparation des données
def prepare_data(df):
    titles = df['title'].tolist()
    labels = torch.tensor(df['label'].tolist())
    tokenized_data = tokenize_titles(titles)
    dataset = TensorDataset(tokenized_data['input_ids'], tokenized_data['attention_mask'], labels)
    return DataLoader(dataset, batch_size=4)

# Entraînement du modèle
def train_model(train_dataloader):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    model.train()

    for epoch in range(3):
        for batch in train_dataloader:
            optimizer.zero_grad()
            input_ids, attention_mask, label = batch
            output = model(input_ids, attention_mask=attention_mask, labels=label)
            loss = output.loss
            loss.backward()
            optimizer.step()

# Sauvegarde du modèle
def save_model(model, path='bert_model.pth'):
    torch.save(model.state_dict(), path)
