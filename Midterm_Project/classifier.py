#title: main.py
#/*------------------------------------
#copyright FatemehRahbari, 2024-September-21 19:03:40
#------------------------------------*/





import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer, BertModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score
from torch.utils.data import DataLoader, TensorDataset
import spacy
import nltk
from nltk.util import ngrams
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from lime.lime_text import LimeTextExplainer
from bertviz import head_view





# Semantic Similarity
# similarity between xi and xj
def cosine_similarity(x_i, x_j):
    # Use TfidfVectorizer to convert text to vectors
    tfidf = TfidfVectorizer().fit_transform([x_i, x_j])
    # Calculate cosine similarity
    cosine_sim = cosine_similarity(tfidf[0:1], tfidf[1:2])
    return cosine_sim

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

def get_embeddings(text):
    inputs = tokenizer(text, return_tensors='pt')
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1)

# Step 2.2: Style Features
# Calculate the sentence length
def sentence_leng(text):
    text_len = len(text.split())
    print(f"Length of xi: {text_len}")
    return text_len

# Word Choice and Complexity
def avg_word_length(text):
    text_avg_word_len = sum(len(word) for word in text.split()) / len(text.split())
    print(f"Avg Word Length in text: {text_avg_word_len}")
    return text_avg_word_len

# Use spaCy to tag the text.
def pos_tagging(text):
    nlp = spacy.load("en_core_web_sm")
    doc_text = nlp(text)
    pos_tags = [token.pos_ for token in doc_text]
    print(f"POS tags: {pos_tags}")
    return pos_tags

# Use N-gram Analysis
def ngram_analysis(text, n):
    # Tokenize the text
    tokens_text = nltk.word_tokenize(text)
    bigrams_list = list(ngrams(tokens_text, n))
    print(f"{n}-grams: {bigrams_list}")
    return bigrams_list

#Step 2.3: N-gram Analysis
# Use the count of the frequency of n-grams across different LLM-generated completions
def ngram_frequency(text, n):
    # Tokenize the text
    tokens_text = nltk.word_tokenize(text)
    ngrams_text = list(ngrams(tokens_text, n))
    ngram_freq = nltk.FreqDist(ngrams_text)
    print(f"{n}-gram frequency: {ngram_freq}")
    return ngram_freq

# Step 2.4: Preprocessing
# Tokenization
def tokenize(text):
    tokens = nltk.word_tokenize(text)
    print(f"Tokens: {tokens}")
    return tokens

# Stop Word Removal
def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    filtered_text = [word for word in text.split() if word.lower() not in stop_words]
    print(f"Filtered text: {filtered_text}")
    return filtered_text

# Stemming and Lemmatization
def stemming(text):
    stemmer = PorterStemmer()
    filtered_text = remove_stopwords(text)
    stemmed_text = [stemmer.stem(word) for word in filtered_text]
    print(f"Stemmed text: {stemmed_text}")
    return stemmed_text

# lemmatization
def lemmatization(text):
    nlp = spacy.load("en_core_web_sm")
    doc_text = nlp(text)
    lemmatized_text = [token.lemma_ for token in doc_text]
    print(f"Lemmatized text: {lemmatized_text}")
    return lemmatized_text



# Step 3: Model Architecture
class BERTClassifier(nn.Module):
    def __init__(self, num_classes):
        super(BERTClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.fc = nn.Linear(768 * 2, num_classes)  # Adjust num_classes based on the number of LLMs
        
    def forward(self, x_i, x_j, attention_mask):
        # Get BERT embeddings for the input text
        x_i_output = self.bert(input_indxs=x_i, attention_mask=attention_mask)
        x_j_output = self.bert(input_indxs=x_j, attention_mask=attention_mask)
        # Get the [CLS] token output
        x_i_cls = x_i_output[1]  # [CLS] token output
        x_j_cls = x_j_output[1]  # [CLS] token output
        # Concatenate the embeddings
        combined = torch.cat((x_i_cls, x_j_cls), dim=1)
        # Pass through fully connected layer
        logits = self.fc(combined)
        return logits





# Step 4: Training the Model
# Assuming you have a DataLoader `train_loader` that provides (input_indxs, attention_mask, labels)
# Example dataset (replace with your actual dataset)
input_indxs = torch.tensor([[101, 102], [101, 103]])  # Example input_indxs
attention_mask = torch.tensor([[1, 1], [1, 1]])    # Example attention_mask
labels = torch.tensor([0, 1])                      # Example labels

dataset = TensorDataset(input_indxs, attention_mask, labels)
train_loader = DataLoader(dataset, batch_size=2, shuffle=True)

# Initialize model
num_classes = 5  # Adjust based on the number of LLMs
model = BERTClassifier(num_classes)



#Define Optimizer
optimizer = optim.Adam(model.parameters(), lr=2e-5)
# Define Loss 
criterion = nn.CrossEntropyLoss()

# Training Loop
def train_model(model, train_loader, optimizer, criterion, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in train_loader:
            x_i, x_j, labels = batch
            optimizer.zero_grad()
            outputs = model(xi, xj)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_loader)}")

# Step 5: Evaluation
# Assuming you have input data pairs and labels
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Assuming you have a DataLoader `test_loader` that provides (input_indxs, attention_mask, labels)
model.eval()
y_pred = []
y_true = []
with torch.no_grad():
    for batch in test_loader:
        input_indxs, attention_mask, labels = batch
        logits = model(input_indxs, attention_mask)
        predictions = torch.argmax(logits, dim=1)
        y_pred.extend(predictions.cpu().numpy())
        y_true.extend(labels.cpu().numpy())

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='macro')
recall = recall_score(y_true, y_pred, average='macro')
f1 = f1_score(y_true, y_pred, average='macro')
print(f"Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1-Score: {f1}")

# Step 6: Fine-Tuning and Hyperparameter Tuning
# Use cross-validation to tune hyperparameters
scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
print(f"Mean cross-validation accuracy: {scores.mean()}")

conf_matrix = confusion_matrix(y_test, y_pred)
print(f"Confusion Matrix: \n{conf_matrix}")

# Plotting confusion matrix
plt.figure(figsize=(10,7))
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='d')
plt.xlabel('Predicted LLM')
plt.ylabel('True LLM')
plt.show(

# Create a SHAP explainer
explainer = shap.Explainer(model, X_test)

# Calculate SHAP values for test set
shap_values = explainer(X_test)

# Visualize feature importance for a single prediction
shap.plots.waterfall(shap_values[0])


# Initialize LIME text explainer
explainer = LimeTextExplainer()

# Explain a prediction
explanation = explainer.explain_instance(X_test[0], model.predict_proba, num_features=10)

# Show the explanation
explanation.show_in_notebook(text=True)