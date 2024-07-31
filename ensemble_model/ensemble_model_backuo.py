import torch
import torch.nn as nn
from transformers import AdamW, BertModel, BertTokenizer, RobertaModel, RobertaTokenizer
from sklearn.metrics import precision_score, recall_score, f1_score, precision_recall_curve
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn.manifold import TSNE
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import seaborn as sns
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
        return logits, combined

    def predict(self, sentence1, sentence2, model_path):
        self.load_state_dict(torch.load(model_path))
        self.eval()
        with torch.no_grad():
            inputs_bert = self.base_model1.tokenize(sentence1).to(next(self.parameters()).device)
            inputs_codebert = self.base_model2.tokenize(sentence2).to(next(self.parameters()).device)
            logits, _ = self.forward(inputs_bert, inputs_codebert)
            probabilities = torch.softmax(logits, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            return predicted_class, probabilities.cpu().numpy()
            
    def plot_pr_curve(self, labels, probabilities):
        precision, recall, _ = precision_recall_curve(labels, probabilities)
        plt.plot(recall, precision, label='PR Curve')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_tsne(self, embeddings, labels):
        # working code 
        # tsne = TSNE(n_components=2, random_state=0)
        # embeddings_np = np.vstack(embeddings, dtype=np.float32)
        
        # print(embeddings_np.shape)
        
        # embeddings_2d = tsne.fit_transform(embeddings_np)
        
        # plt.figure(figsize=(10, 8))
        # scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=labels, cmap='viridis', alpha=0.5)
        # plt.colorbar(scatter, label='Labels')
        # plt.title('t-SNE Visualization of Embeddings')
        # plt.show()

        
        tsne = TSNE(n_components=2, random_state=0)

        print("embedding来啦",embeddings[0].shape)
        # embeddings_cpu = combined.cpu().detach()
        # embeddings_np = torch.stack(embeddings_cpu).numpy()  # 形状：(600, 8, 768)
        
        # 维度变换
        embeddings_np = np.vstack(embeddings) 
        embeddings_2d = tsne.fit_transform(embeddings_np)
        df_tsne = pd.DataFrame(embeddings_2d, columns=['TSNE1', 'TSNE2'])
        df_tsne['Class Name'] = labels # Add labels column from df_train to df_tsne
        df_tsne
        
        fig, ax = plt.subplots(figsize=(8,6)) # Set figsize
        sns.set_style('darkgrid', {"grid.color": ".6", "grid.linestyle": ":"})
        sns.scatterplot(data=df_tsne, x='TSNE1', y='TSNE2', hue='Class Name', palette='hls')
        sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
        plt.title('Scatter plot of news using t-SNE');
        plt.xlabel('TSNE1');
        plt.ylabel('TSNE2');
        plt.axis('equal')

        # Apply KMeans
        kmeans_model = KMeans(n_clusters=2, random_state=1, n_init='auto').fit(embeddings_np)
        labels = kmeans_model.fit_predict(embeddings_np)
        
        df_tsne['Cluster'] = labels
        df_tsne
        fig, ax = plt.subplots(figsize=(8,6)) # Set figsize
        sns.set_style('darkgrid', {"grid.color": ".6", "grid.linestyle": ":"})
        sns.scatterplot(data=df_tsne, x='TSNE1', y='TSNE2', hue='Cluster', palette='magma')
        sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
        plt.title('Scatter plot of news using KMeans Clustering');
        plt.xlabel('TSNE1');
        plt.ylabel('TSNE2');
        plt.axis('equal')

    def trainer(self, train_loader, val_loader, num_epochs=3, learning_rate=2e-5, patience=3):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)

        optimizer = AdamW(self.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()

        best_acc = 0.0
        patience_counter = 0

        for epoch in range(num_epochs):
            self.train()
            total_loss = 0
            
            for batch in train_loader:
                inputs_bert, inputs_codebert, labels = batch
                inputs_bert = {k: v.to(device) for k, v in inputs_bert.items()}
                inputs_codebert = {k: v.to(device) for k, v in inputs_codebert.items()}
                labels = labels.to(device)

                optimizer.zero_grad()
                logits, _ = self(inputs_bert, inputs_codebert)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            print('=============================train========================')
            print(f'Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss}')
            print('=============================eval========================')
            val_acc, val_labels, val_probabilities, val_embeddings = self.evaluate(val_loader)
            print(f'Epoch {epoch+1}/{num_epochs}, Validation Accuracy: {val_acc}')
            
            if val_acc > best_acc:
                best_acc = val_acc
                print('当前best acc',best_acc)
                patience_counter = 0
            else:
                print('比best acc差记一次',patience_counter)
                patience_counter += 1

            if patience_counter >= patience:
                print('Early stopping')
                print('best acc是', best_acc)
                break
                
        # Plot PR curve for the best model
        self.plot_pr_curve(val_labels, val_probabilities)
        # Plot t-SNE for the best model
        self.plot_tsne(val_embeddings, val_labels)
    def evaluate(self, val_loader):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.eval()
        correct = 0
        total = 0
        all_labels = []
        all_predictions = []
        all_probabilities = []
        all_embeddings = []
        with torch.no_grad():
            for batch in val_loader:
                inputs_bert, inputs_codebert, labels = batch
                inputs_bert = {k: v.to(device) for k, v in inputs_bert.items()}
                inputs_codebert = {k: v.to(device) for k, v in inputs_codebert.items()}
                labels = labels.to(device)
                logits, embeddings = self(inputs_bert, inputs_codebert)
                # print("eval embedding",embeddings)
                _, predicted = torch.max(logits, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())
                all_probabilities.extend(torch.softmax(logits, dim=1).cpu().numpy()[:, 1]) # probability of the positive class
                all_embeddings.extend(embeddings.cpu().numpy())
                
        accuracy = correct / total
        precision = precision_score(all_labels, all_predictions, average='weighted', zero_division=0)
        recall = recall_score(all_labels, all_predictions, average='weighted', zero_division=0)
        f1 = f1_score(all_labels, all_predictions, average='weighted', zero_division=0)

        print(f'Validation Accuracy: {accuracy}')
        print(f'Precision: {precision}')
        print(f'Recall: {recall}')
        print(f'F1-Score: {f1}')
        return accuracy, all_labels, all_probabilities, all_embeddings
