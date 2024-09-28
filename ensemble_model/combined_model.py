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
from tqdm import tqdm
# Define base model class
class BaseModel(nn.Module):
    def __init__(self, transformer_model):
        super(BaseModel, self).__init__()
        self.transformer_model = transformer_model


    def forward(self, inputs):
        outputs = self.transformer_model(**inputs).last_hidden_state[:, 0, :]
        return outputs

    # def tokenize(self, text, max_length=128):
    #     return self.tokenizer(text, return_tensors='pt', truncation=True, padding='max_length', max_length=max_length)

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

# Define stacking model class
class CombinedModel(nn.Module):
    def __init__(self, base_model1, base_model2):
        super(CombinedModel, self).__init__()
        self.base_model1 = base_model1
        self.base_model2 = base_model2
        combined_input_dim = base_model1.transformer_model.config.hidden_size + base_model2.transformer_model.config.hidden_size
        self.classifier = nn.Linear(combined_input_dim, 2)

    def forward(self, inputs_bert, inputs_codebert):
        outputs_bert = self.base_model1(inputs_bert)
        outputs_codebert = self.base_model2(inputs_codebert)
        combined = torch.cat((outputs_bert, outputs_codebert), dim=1)
        logits = self.classifier(combined)
        return logits, combined

    def trainer(self, train_loader, val_loader, num_epochs=3, learning_rate=2e-5, patience=3):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)

        optimizer = AdamW(self.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()

        early_stopper = EarlyStopper(patience=patience, min_delta=10)
        for epoch in range(num_epochs):
            self.train()
            total_loss = 0
            with tqdm(total=len(train_loader), desc=f'Epoch {epoch+1}/{num_epochs}', unit='batch',position=0 ,leave=True) as pbar:
                for step, batch in enumerate(train_loader):
                    inputs_bert, inputs_codebert, labels = batch
                    inputs_bert = {k: v.to(device) for k, v in inputs_bert.items()}
                    inputs_codebert = {k: v.to(device) for k, v in inputs_codebert.items()}
                    # inputs_bert= inputs_bert.to(device)
                    # inputs_codebert= inputs_codebert.to(device)
                    labels = labels.to(device)
                    logits, _ = self(inputs_bert, inputs_codebert)
                    loss = criterion(logits, labels)
                    loss.mean().backward()
                    total_loss += loss.mean().item()
                    optimizer.step()
                    optimizer.zero_grad()
                    if step % 100 == 1:
                        print("Epoch {}, average loss: {}".format(epoch, total_loss / (step + 1)), flush=True)
                    pbar.update(1)
            # Validation loop
            validation_loss = 0
            self.eval()
            allpreds = []
            alllabels = []
            with torch.no_grad():
                with tqdm(total=len(val_loader), desc=f'evaluate', unit='batch',position=0 ,leave=True) as pbar:
                    for val_step, val_batch in enumerate(val_loader):
                        val_bert_inputs, val_inputs_codebert, val_labels = val_batch
                        val_bert_inputs = {k: v.to(device) for k, v in val_bert_inputs.items()}
                        val_inputs_codebert = {k: v.to(device) for k, v in val_inputs_codebert.items()}
                        # inputs_bert= inputs_bert.to(device)
                        # inputs_codebert= inputs_codebert.to(device)
                        val_labels = val_labels.to(device)
                        val_logits,_ = self(val_bert_inputs, val_inputs_codebert)
                        val_loss = criterion(val_logits, val_labels)
                        validation_loss += val_loss.mean().item()
                        alllabels.extend(val_labels.cpu().tolist())
                        allpreds.extend(torch.argmax(val_logits, dim=-1).cpu().tolist())
                        pbar.update(1)
            validation_loss /= (val_step + 1)
            acc = sum([int(i==j) for i,j in zip(allpreds, alllabels)])/len(allpreds)
            # accuracy_score
            print("Epoch {}, validate accuracy score: {}".format(epoch, acc), flush=True)

            if early_stopper.early_stop(validation_loss):             
                break
        # Plot PR curve for the best model
        # self.plot_pr_curve(val_labels, val_probabilities)
        # Plot t-SNE for the best model
        # self.plot_tsne(val_embeddings, val_labels)
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
            with tqdm(total=len(val_loader), desc=f'evaluate', unit='batch',position=0 ,leave=True) as pbar:
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
                    pbar.update(1)
        accuracy = correct / total
        precision = precision_score(all_labels, all_predictions, average='weighted', zero_division=0)
        recall = recall_score(all_labels, all_predictions, average='weighted', zero_division=0)
        f1 = f1_score(all_labels, all_predictions, average='weighted', zero_division=0)

        print(f'Validation Accuracy: {accuracy}')
        print(f'Precision: {precision}')
        print(f'Recall: {recall}')
        print(f'F1-Score: {f1}')
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