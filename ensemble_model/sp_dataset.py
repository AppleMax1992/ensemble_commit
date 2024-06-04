import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
class SentencePairDataset(Dataset):
    def __init__(self, data, bert_tokenizer, codebert_tokenizer):
        self.data = data
        self.bert_tokenizer = bert_tokenizer
        self.codebert_tokenizer = codebert_tokenizer

    def __len__(self):
        return len(self.data)

    def get_embedding(self, idx):
        sentence1, sentence2, label = self.data[idx]
        inputs_bert = self.bert_tokenizer(sentence1, return_tensors='pt', truncation=True, padding='max_length', max_length=128)
        inputs_codebert = self.codebert_tokenizer(sentence2, return_tensors='pt', truncation=True, padding='max_length', max_length=128)
        return {k: v.squeeze(0) for k, v in inputs_bert.items()}, {k: v.squeeze(0) for k, v in inputs_codebert.items()}, torch.tensor(label, dtype=torch.long)
        # this following line was basic though, anyway it fails
        # return inputs_bert, inputs_bert, torch.tensor(label, dtype=torch.long)

    def __getitem__(self, idx):
        return self.get_embedding(idx)