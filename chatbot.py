import spacy
import gensim
import nltk
import numpy as np
import pandas as pd
import gzip
import torch
import torch.nn as nn
import torchtext
from nltk.corpus import brown
from torch.utils.data import Dataset
from torchtext.legacy.data import Iterator
from torchtext.legacy.data import Example
from torchtext.legacy.data import Field
from torchtext.datasets import SQuAD2
import random

question_field = Field(sequential=True, use_vocab=True, init_token='<sos>', eos_token='<eos>', tokenize='spacy', tokenizer_language='en_core_web_sm', lower=True, include_lengths=False)
answer_field = Field(sequential=True, use_vocab=True, init_token='<sos>', eos_token='<eos>', tokenize='spacy', tokenizer_language='en_core_web_sm', lower=True, include_lengths=False)

train_data, val_data = SQuAD2(split=('train', 'dev'))

train_dictionary = {"Questions" : [], "Answers": []}
for _, q, a, _ in train_data:
    train_dictionary["Questions"].append(q)
    train_dictionary["Answers"].append(a[0])
train_df = pd.DataFrame(train_dictionary)

val_dictionary = {"Questions" : [], "Answers": []}
for _, q, a, _ in val_data:
    val_dictionary["Questions"].append(q)
    val_dictionary["Answers"].append(a[0])
val_df = pd.DataFrame(val_dictionary)

class SQuAD2Dataset(torchtext.legacy.data.Dataset):
    def __init__(self, df, src_field, trg_field):
        self.src_field = src_field
        self.trg_field = trg_field
        fields = [('src', src_field), ('trg', trg_field)]
        examples = []
        for i, row in df.iterrows():
            src = src_field.tokenize(row['Questions'])
            trg = trg_field.tokenize(row['Answers'])
            examples.append(Example.fromlist([src, trg], fields))
        super().__init__(examples, fields)
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return self.examples[idx]

question_field.build_vocab(train_df['Questions'], max_size=10000, min_freq=2)
answer_field.build_vocab(train_df['Answers'], max_size=10000, min_freq=2)

train_dataset = SQuAD2Dataset(train_df, question_field, answer_field)
val_dataset = SQuAD2Dataset(val_df, question_field, answer_field)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 64
train_iterator, valid_iterator = Iterator.splits(
    (train_dataset, val_dataset), batch_sizes=(batch_size, batch_size), sort_key=lambda x: len(x.src), device=device, sort_within_batch=False)

import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, num_layers, dropout=0.2):
        super().__init__()

        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.num_layers = num_layers
        self.dropout = dropout

        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.lstm = nn.LSTM(emb_dim, emb_dim, num_layers, dropout=dropout, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
    
    def init_hidden(self, batch_size):
        hidden = torch.zeros(self.num_layers, batch_size, self.emb_dim).to(device)
        cell = torch.zeros(self.num_layers, batch_size, self.emb_dim).to(device)
        return hidden, cell

    def forward(self, x, hidden, cell):
        x = self.dropout(self.embedding(x))
        x, (hidden, cell) = self.lstm(x, hidden, cell)
        return x, hidden, cell
    
class Decoder(nn.Module):
    def __init__(self, output_dim, hidden_dim, num_layers, dropout=0.2):
        super().__init__()

        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout

        self.embedding = nn.Embedding(output_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, hidden, cell):
        x = x.unsqueeze(0)
        x = self.dropout(self.embedding(x))
        x, (hidden, cell) = self.lstm(x, (hidden, cell))
        x = self.fc(x.squeeze(0))
        return x, hidden, cell

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = src.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim

        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(device)

        hidden, cell = self.encoder.init_hidden(batch_size)

        for i in range(src.shape[0]):
            _, hidden, cell = self.encoder(src[i], hidden, cell)
        
        x = torch.Tensor([[0]]).long().to(device)

        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(x, hidden, cell)
            outputs[t] = output
            best_guess = output.argmax(1)
            x = trg[t] if random.random() < teacher_forcing_ratio else best_guess

        return outputs


input_dim = len(question_field.vocab)
output_dim = len(answer_field.vocab)
hidden_dim = 512
dropout = 0.2

encoder = Encoder(input_dim, hidden_dim, 2, dropout)
decoder = Decoder(output_dim, hidden_dim, 2, dropout)
model = Seq2Seq(encoder, decoder).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

PAD_IDX = answer_field.vocab.stoi['<pad>']
criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

num_epochs = 1
for epoch in range(num_epochs):
    for batch in train_iterator:
        src = batch.src
        trg = batch.trg
        optimizer.zero_grad()
        output = model(src, trg)
        output_dim = output.shape[-1]
        output = output[1:].view(-1, output_dim)
        trg = trg[1:].view(-1)
        loss = criterion(output, trg)
        loss.backward()
        optimizer.step()

    print(f'Epoch: {epoch+1:02} | Epoch Loss: {loss:.3f}')
