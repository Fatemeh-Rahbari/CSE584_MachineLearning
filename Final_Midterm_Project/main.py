#title: main.py
#/*------------------------------------
#copyright FatemehRahbari, 2024-September-21 19:03:40
#------------------------------------*/





import os
import csv
import logging
import configparser
from pathlib import Path
import config_classes as CC
from torch.nn.utils.rnn import pad_sequence
import torch
from transformers import BertTokenizer, BertModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy
import nltk
from nltk.util import ngrams
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import torch
from transformers import BertModel
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from lime.lime_text import LimeTextExplainer






"""
    Constructs and returns a dictionary of paths based on the configuration.
"""
def construct_paths(config):
    paths_dict = {}
    # Construct paths
    folders = [CC.InputFiles.INPUTFOLDER, CC.OutputFiles.OUTPUTFOLDER]
    for folder in folders:
        paths_dict[f"{folder}_path"] = os.path.join(config[CC.InputFiles.LOCATIONS][folder])    
    return paths_dict





"""
    Ensures that the path exists as a directory. If the path exists but is a file,
    logs a warning and does not attempt to create the directory.
"""
def ensure_directory(path):
    # Check if the path exists
    if os.path.exists(path):
        if os.path.isfile(path):
            logging.warning(f"Expected a directory but found a file: {path}")
        # If it"s a directory, there"s nothing more to do
    else:
        # Path does not exist, create the directory
        os.makedirs(path, exist_ok=True)





"""
    Validates that all paths in the paths_dict exist.
"""
def validate_paths(paths_dict):
        # Check if all paths exist
        try:
            for path in paths_dict.values():
                if not Path(path).exists():
                    raise FileNotFoundError(f"Path {path} does not exist.")
        except KeyError as e:
            print(f"Missing configuration for {e}")
            return
        except FileNotFoundError as e:
            print(e)
            return





"""
    Reads the configuration file and returns the configuration object.
"""
def read_configuration (config_file_name):
    config= None
    config = configparser.ConfigParser(allow_no_value=True)
    config.read(config_file_name)
    return config





""" 
    Data Preparation and Callection: Generate random truncated texts (x_i) using gpt-3.5-turbo model from OpenAI API 
"""
def generate_input_texts(num_samples, considered_txt_generator, considered_model,api_key):
    x_i_list = []
        # Prompt for the considered text generator
    x_i_prompt = f"Generate {num_samples} sets of random truncated texts x_i. “Yesterday I went” is an example of a truncate text."
    for i in range(num_samples):
        x_i = considered_txt_generator(x_i_prompt, considered_model, api_key)
        x_i_list.append(x_i)
    return x_i_list





"""
    Generate completions for each x_i using different LLMs
"""
def generate_llm_completions(x_i_list, llm_functions, llm_model_names, api_keys_list, num_completions, tokenizer):   
    outcomes = {}
    for idx, x_i in enumerate(x_i_list):
        print(f"Generating completions for x_i[{idx+1}]: {x_i}")
        outcomes[x_i] = {}
        for llm_model_name, llm_func, api_key in zip(llm_model_names, llm_functions, api_keys_list):
            outcomes[x_i][llm_model_name] = []
            for _ in range(num_completions):
                try:
                    # Prompt to complete truncated x_i text
                    x_j_prompt = f"Given the truncated text '{x_i}', complete it by appending x_j to get a complete text."
                    # Use each LLM to generate x_j for the given x_i
                    x_j = llm_func(x_j_prompt, llm_model_name, api_key)
                    complete_text = f"{x_i} {x_j}"
                    
                    # Tokenize the input and completion
                    encoded_xi = tokenizer.encode(x_i, return_tensors='pt')
                    encoded_xj = tokenizer.encode(x_j, return_tensors='pt')
                    
                    # Pad the sequences to ensure they have the same length
                    padded_input = pad_sequence([encoded_xi, encoded_xj], batch_first=True)
                    
                    outcomes[x_i][llm_model_name].append(complete_text)
                except Exception as e:
                    print(f"Error in model {llm_model_name}: {str(e)}")
                    outcomes[x_i][llm_model_name].append("Error generating completion")
    return outcomes





"""
    Save the generated x_i and x_j pairs to a file
"""
def save_outcomes(outcomes, output_file_path):
    with open(output_file_path, "w", newline='') as csvfile:
        fieldnames = ['input_text', 'llm_model', 'completion']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for input_text, completions in outcomes.items():
            for llm_model, completion in completions.items():
                writer.writerow({'input_text': input_text, 'llm_model': llm_model, 'completion': completion})
    
    print(f"Results saved to {output_file_path}")




"""
Feature Extraction: Semantic Similarity
""" 
# Similarity between xi and xj
def cosine_sim(x_i, x_j):
    # Create a TF-IDF vectorizer
    vectorizer = TfidfVectorizer()
    # Use TfidfVectorizer to convert text to vectors
    tfidf = vectorizer.fit_transform([x_i, x_j])
    # Calculate cosine similarity
    sim = cosine_similarity(tfidf[0:1], tfidf[1:2])
    return sim

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')
# Get BERT embeddings for the text
def get_embeddings(text):
    inputs = tokenizer(text, return_tensors='pt')
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1)





"""
Style Features
"""
# Calculate the sentence length
def sentence_leng(text):
    text_len = len(text.split())
    print(f"Length of xi: {text_len}")
    return text_lent

# Word Choice and Complexity
def avg_word_length(text):
    text_avg_word_len = sum(len(word) for word in text.split()) / len(text.split())
    print(f"Avg Word Length in text: {text_avg_word_len}")
    return text_avg_word_len

# Use spaCy to tag the text
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

# Use the count of the frequency of n-grams across different LLM-generated completions
def ngram_frequency(text, n):
    # Tokenize the text
    tokens_text = nltk.word_tokenize(text)
    ngrams_text = list(ngrams(tokens_text, n))
    ngram_freq = nltk.FreqDist(ngrams_text)
    print(f"{n}-gram frequency: {ngram_freq}")
    return ngram_freq

"""
Preprocessing
"""
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






"""
Evaluation: Evaluate the model on the test set
"""
# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert the data to PyTorch tensors
X_train = torch.tensor(X_train)
X_test = torch.tensor(X_test)
y_train = torch.tensor(y_train)
y_test = torch.tensor(y_test)

# Prediction and metric calculation
def evaluate_model(model, test_loader, y):
    # Get model predictions on the test set
    y_pred = model.predict(test_loader)
    # Calculate evaluation metrics
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred, average='macro')
    recall = recall_score(y, y_pred, average='macro')
    f1 = f1_score(y, y_pred, average='macro')
    return accuracy, precision, recall, f1





"""
Fine-Tuning and Hyperparameter Tuning: Use cross-validation to tune hyperparameters
"""
# Use cross-validation to tune hyperparameters
scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
print(f"Mean cross-validation accuracy: {scores.mean()}")

# Fine-tune the model
conf_matrix = confusion_matrix(y_test, y_pred)
print(f"Confusion Matrix: \n{conf_matrix}")



# Create a SHAP explainer object
explainer = shap.Explainer(model, X_test)

# Calculate SHAP values for test set predictions
shap_values = explainer(X_test)

# Visualize feature importance for a single prediction using waterfall plot
shap.plots.waterfall(shap_values[0])

# Initialize LIME text explainer object
explainer = LimeTextExplainer()

# Explain a prediction from the model
explanation = explainer.explain_instance(X_test[0], model.predict_proba, num_features=10)

# Show the explanation
explanation.show_in_notebook(text=True)
