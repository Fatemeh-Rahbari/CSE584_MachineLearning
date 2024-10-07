#title: main.py
#/*------------------------------------
#copyright FatemehRahbari, 2024-September-21 19:03:40
#------------------------------------*/





import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertModel





"""
Model Architecture: # Build a deep learning classifier to figure out, for each input (xi, xj), which LLM was used for this pair.
"""
class BERTClassifier(nn.Module):
    def __init__(self, num_classes):
        super(BERTClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.fc = nn.Linear(768 * 2, num_classes)  # Adjust num_classes based on the number of LLMs
        
    def forward(self, x_i, x_j, attention_mask):
        # Get BERT embeddings for the input text
        x_i_output = self.bert(input_ids=x_i, attention_mask=attention_mask)
        x_j_output = self.bert(input_ids=x_j, attention_mask=attention_mask)
        # Get the [CLS] token output
        x_i_cls = x_i_output[1]  # [CLS] token output
        x_j_cls = x_j_output[1]  # [CLS] token output
        # Concatenate the embeddings
        combined = torch.cat((x_i_cls, x_j_cls), dim=1)
        # Pass through fully connected layer
        logits = self.fc(combined)
        return logits





"""
Training the Model
"""


# Training Loop
def train_model(model, train_loader, optimizer, criterion, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in train_loader:
            x_i, x_j, labels = batch
            optimizer.zero_grad()
            outputs = model(x_i, x_j)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_loader)}")

