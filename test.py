import pandas as pd
from sklearn.model_selection import train_test_split
import ensemble_model.preprocesser as preprocesser 
import ensemble_model.ensemble_model as em 
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertTokenizer, RobertaModel, RobertaTokenizer
import whatthepatch
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Load BERT and CodeBERT models and tokenizers
bert_model = BertModel.from_pretrained('/root/autodl-tmp/models/bert-base-cased')
bert_tokenizer = BertTokenizer.from_pretrained('/root/autodl-tmp/models/bert-base-cased')

codebert_model = RobertaModel.from_pretrained('/root/autodl-tmp/models/codebert-base')
codebert_tokenizer = RobertaTokenizer.from_pretrained('/root/autodl-tmp/models/codebert-base')

train_data = pd.read_csv('train.csv')[:100]
# train_data.rename(columns={'message':'command','label':'message','command':'label'},inplace=True)

val_data = pd.read_csv('val.csv')[:100]
val_data


# Create Datasets and DataLoaders
train_dataset = preprocesser.SentencePairDataset(train_data, bert_tokenizer, codebert_tokenizer)
val_dataset = preprocesser.SentencePairDataset(val_data, bert_tokenizer, codebert_tokenizer)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)




# # Initialize the model
# model = em.CombinedModel(bert_model, codebert_model, bert_tokenizer, codebert_tokenizer)

# # Train the model
# model.trainer(train_loader, val_loader,num_epochs=10)
# for batch in train_loader:
#     print(batch)


base_model1 = em.BaseModel(bert_model,bert_tokenizer)
base_model2 = em.BaseModel(codebert_model,codebert_tokenizer)

# Create stacking model
stacking_model = em.StackingMoEModel(base_model1, base_model2)
# Train the model
stacking_model.trainer(train_loader, val_loader,num_epochs=1, patience=3)