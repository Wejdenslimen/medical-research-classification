This project demonstrates a text classification task using BERT (Bidirectional Encoder Representations from Transformers) for classifying titles and abstracts from scientific articles based on specific keywords. The code includes the following steps:

Data Loading: The dataset consists of JSON files containing scientific articles with titles and abstracts. The data is loaded from a Google Drive folder into a Pandas DataFrame.

Data Cleaning: The titles and abstracts are cleaned by removing HTML tags, special characters, and converting the text to lowercase to prepare it for tokenization.

Dynamic Labeling: Labels are assigned to the data based on the presence of specific keywords in the titles. The labels are dynamically generated:

Label 1: Titles containing keywords such as "novel" or "combination".
Label 0: Titles containing keywords such as "case report" or "diagnosis".
BERT Tokenization: The cleaned titles are tokenized using the pre-trained BERT tokenizer, which converts the text into input tokens that can be fed into the BERT model.

Model Training: A pre-trained BERT model (bert-base-uncased) is fine-tuned on the labeled data for sequence classification. The model is trained using the AdamW optimizer with a learning rate of 1e-5.

Evaluation: After training, the model is evaluated on the training data, and a classification report is generated to show the model's performance (precision, recall, F1 score).

