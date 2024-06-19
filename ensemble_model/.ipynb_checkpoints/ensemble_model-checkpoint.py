import torch
import torch.nn as nn
from transformers import AdamW, BertModel, BertTokenizer, RobertaModel, RobertaTokenizer
from sklearn.metrics import precision_score, recall_score, f1_score

# Define base model class
class BaseModel(nn.Module):
    def __init__(self, transformer_model, tokenizer):
        super(BaseModel, self).__init__()
        self.transformer_model = transformer_model
        self.tokenizer = tokenizer
        self.dropout = nn.Dropout(0.3)

    def forward(self, inputs):
        outputs = self.transformer_model(**inputs).last_hidden_state[:, 0, :]
        outputs = self.dropout(outputs)
        return outputs

    def tokenize(self, text, max_length=128):
        return self.tokenizer(text, return_tensors='pt', truncation=True, padding='max_length', max_length=max_length)

# Define stacking model class
class StackingModel(nn.Module):
    def __init__(self, base_model1, base_model2):
        super(StackingModel, self).__init__()
        self.base_model1 = base_model1
        self.base_model2 = base_model2
        self.classifier = nn.Linear(base_model1.transformer_model.config.hidden_size + base_model2.transformer_model.config.hidden_size, 2)

    def forward(self, inputs_bert, inputs_codebert):
        outputs_bert = self.base_model1(inputs_bert)
        outputs_codebert = self.base_model2(inputs_codebert)
        combined = torch.cat((outputs_bert, outputs_codebert), dim=1)
        logits = self.classifier(combined)
        return logits

    def predict(self, sentence1, sentence2):
        self.eval()
        with torch.no_grad():
            inputs_bert = self.base_model1.tokenize(sentence1).to(next(self.parameters()).device)
            inputs_codebert = self.base_model2.tokenize(sentence2).to(next(self.parameters()).device)
            logits = self.forward(inputs_bert, inputs_codebert)
            probabilities = torch.softmax(logits, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            return predicted_class, probabilities.cpu().numpy()

    def trainer(self, train_loader, val_loader, num_epochs=3, learning_rate=2e-5):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)

        optimizer = AdamW(self.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(num_epochs):
            self.train()
            total_loss = 0
            for batch in train_loader:
                inputs_bert, inputs_codebert, labels = batch
                inputs_bert = {k: v.to(device) for k, v in inputs_bert.items()}
                inputs_codebert = {k: v.to(device) for k, v in inputs_codebert.items()}
                labels = labels.to(device)

                optimizer.zero_grad()
                logits = self(inputs_bert, inputs_codebert)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            print(f'Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss}')

        self.evaluate(val_loader)

    def evaluate(self, val_loader):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.eval()
        correct = 0
        total = 0
        all_labels = []
        all_predictions = []
        with torch.no_grad():
            for batch in val_loader:
                inputs_bert, inputs_codebert, labels = batch
                inputs_bert = {k: v.to(device) for k, v in inputs_bert.items()}
                inputs_codebert = {k: v.to(device) for k, v in inputs_codebert.items()}
                labels = labels.to(device)

                logits = self(inputs_bert, inputs_codebert)
                _, predicted = torch.max(logits, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())

        accuracy = correct / total
        precision = precision_score(all_labels, all_predictions)
        recall = recall_score(all_labels, all_predictions)
        f1 = f1_score(all_labels, all_predictions)

        print(f'Validation Accuracy: {accuracy}')
        print(f'Precision: {precision}')
        print(f'Recall: {recall}')
        print(f'F1-Score: {f1}')