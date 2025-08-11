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
from functools import partial
from tqdm import tqdm as std_tqdm
tqdm = partial(std_tqdm, dynamic_ncols=True)
# Define base model class
class BaseModel(nn.Module):
    def __init__(self, transformer_model):
        super(BaseModel, self).__init__()
        self.transformer_model = transformer_model
        self.dropout = nn.Dropout(0.3)

    def forward(self, inputs):
        outputs = self.transformer_model(**inputs).last_hidden_state
        outputs = self.dropout(outputs)
        return outputs
    
class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
    
import torch
import torch.nn as nn

class CrossAttentionMoEModel(nn.Module):
    def __init__(self, base_model1, base_model2, num_heads=2):
        super(CrossAttentionMoEModel, self).__init__()
        self.base_model1 = base_model1  # BERT
        self.base_model2 = base_model2  # CodeBERT

        hidden_size = base_model1.transformer_model.config.hidden_size
        assert hidden_size == base_model2.transformer_model.config.hidden_size

        # Cross Attention: Query from model1, Key/Value from model2
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            batch_first=True
        )

        self.classifier = nn.Linear(hidden_size, 2)  # 二分类
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, inputs_bert, inputs_codebert):
        # 输出形状假设为 [batch_size, seq_len, hidden_size]
        outputs_bert = self.base_model1(inputs_bert)      # Query
        outputs_codebert = self.base_model2(inputs_codebert)  # Key & Value

        # Cross Attention
        # query 来自 BERT， key 和 value 来自 CodeBERT
        attn_output, attn_weights = self.cross_attention(
            query=outputs_bert,      # [B, seq_len_q, H]
            key=outputs_codebert,    # [B, seq_len_k, H]
            value=outputs_codebert   # [B, seq_len_k, H]
        )

        # 聚合（这里用 mean pooling，也可以取 CLS token）
        pooled_output = attn_output.mean(dim=1)

        logits = self.classifier(pooled_output)

        return logits, pooled_output


    def trainer(self, train_loader, val_loader, num_epochs=3, learning_rate=2e-5, patience=3):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)

        optimizer = AdamW(self.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        early_stopper = EarlyStopper(patience=3, min_delta=10)
        best_acc = 0.0
        patience_counter = 0

        for epoch in range(num_epochs):
            self.train()
            total_loss = 0
            
            with tqdm(total=len(train_loader), desc=f'Epoch {epoch+1}/{num_epochs}', unit='batch') as pbar:
                for batch in train_loader:
                    inputs_bert, inputs_codebert, labels = batch
                    inputs_bert = {k: v.to(device) for k, v in inputs_bert.items()}
                    inputs_codebert = {k: v.to(device) for k, v in inputs_codebert.items()}
                    labels = labels.to(device)
    
                    optimizer.zero_grad()
                    logits, _ = self(inputs_bert, inputs_codebert)
                    loss = criterion(logits, labels)
                    loss.mean().backward()
                    optimizer.step()
    
                    total_loss += loss.item()
                    # update bar
                    pbar.update(1)
                avg_loss = total_loss / len(train_loader)
                print('=============================train========================')
                pbar.set_description(f'Epoch {epoch+1}/{num_epochs} Loss: {avg_loss:.4f}')
                # print('=============================eval========================')
                # val_acc, val_labels, val_probabilities, val_embeddings, val_predictions = self.evaluate(val_loader)
                # print(f'Epoch {epoch+1}/{num_epochs}, Validation Accuracy: {val_acc}')
                # val_loss = 1 - val_acc  # Assuming you want to minimize loss; adjust if using actual validation loss
                # Use EarlyStopper to decide whether to stop
                if early_stopper.early_stop(avg_loss):
                    print(f"Early stopping triggered after {epoch+1} epochs")
                    break
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

                # Forward pass through the model to get logits and attention weights
                logits, embeddings = self(inputs_bert, inputs_codebert)
                # print("eval embedding",embeddings)
                _, predicted = torch.max(logits, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())
                all_probabilities.extend(torch.softmax(logits, dim=1).cpu().numpy()[:, 1]) # probability of the positive class
                all_embeddings.extend(embeddings.cpu().numpy())

        # Compute metrics
        accuracy = correct / total
        precision = precision_score(all_labels, all_predictions, average='weighted', zero_division=0)
        recall = recall_score(all_labels, all_predictions, average='weighted', zero_division=0)
        f1 = f1_score(all_labels, all_predictions, average='weighted', zero_division=0)

        # Print metrics
        print(f'Validation Accuracy: {accuracy:.4f}')
        print(f'Precision: {precision:.4f}')
        print(f'Recall: {recall:.4f}')
        print(f'F1-Score: {f1:.4f}')

        # Return results
        return accuracy, all_labels, all_probabilities, all_embeddings, all_predictions


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
        
        tsne = TSNE(n_components=2, random_state=42)
    
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
        kmeans_model = KMeans(n_clusters=2, random_state=42, n_init='auto').fit(embeddings_np)
        labels = kmeans_model.fit_predict(embeddings_np)
        
        df_tsne['Cluster'] = labels
        df_tsne
        fig, ax = plt.subplots(figsize=(8,6)) # Set figsize
        sns.set_style('darkgrid', {"grid.color": ".6", "grid.linestyle": ":"})
        sns.scatterplot(data=df_tsne, x='TSNE1', y='TSNE2', hue='Cluster', palette='magma')
        sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
        plt.title('Scatter plot of news using KMeans Clustering');
        plt.xlabel('TSNE1')
        plt.ylabel('TSNE2')
        plt.axis('equal')

    def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion Matrix on Dataset I', cmap=plt.cm.Blues):
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
    
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)
    
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], fmt),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")
    
        plt.ylabel('True label')
        # plt.xlabel('Predicted label')
        plt.tight_layout()



