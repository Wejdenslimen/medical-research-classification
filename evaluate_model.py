from sklearn.metrics import classification_report

# Fonction pour évaluer le modèle
def evaluate_model(model, test_dataloader):
    model.eval()
    predictions = []
    labels = []
    
    with torch.no_grad():
        for batch in test_dataloader:
            input_ids, attention_mask, label = batch
            output = model(input_ids, attention_mask=attention_mask)
            logits = output.logits
            predicted_labels = torch.argmax(logits, dim=1)
            predictions.extend(predicted_labels.cpu().numpy())
            labels.extend(label.cpu().numpy())
    
    print(classification_report(labels, predictions))

# Chargement du modèle
model.load_state_dict(torch.load('bert_model.pth'))
