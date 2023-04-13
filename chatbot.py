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
    def __init__(self, input_size, hidden_size, dropout=0.2):
        super(Encoder, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        
        self.embedding = nn.Embedding(self.input_size, self.hidden_size)
        self.lstm = nn.LSTM(self.hidden_size, self.hidden_size, 2, dropout=self.dropout)
        
    def forward(self, x, hidden, cell):
        x = self.embedding(x)
        x = x.unsqueeze(0)
        
        x, (hidden, cell) = self.lstm(x, (hidden, cell))
        x = x.squeeze(0)
        
        return x, hidden, cell
    
class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size, dropout=0.2):
        super(Decoder, self).__init__()
        
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout = dropout
        
        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.lstm = nn.LSTM(self.hidden_size, self.hidden_size, 2, dropout=self.dropout)
        self.fc = nn.Linear(self.hidden_size, self.output_size)
        
        self.softmax = nn.LogSoftmax(dim=1)
        
    def forward(self, x, hidden, cell):
        x = x.unsqueeze(0)
        x = self.embedding(x)
        x = x.view(1, x.size(0), -1)
        x, (hidden, cell) = self.lstm(x, (hidden, cell))
        x = self.fc(x)
        x = self.softmax(x)
        
        return x, hidden, cell

class Seq2Seq(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout=0.2):
        super(Seq2Seq, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout = dropout
        
        self.encoder = Encoder(self.input_size, self.hidden_size, self.dropout)
        self.decoder = Decoder(self.hidden_size, self.output_size, self.dropout)
    
    def forward(self, src, trg, batch_size, teacher_force=0.5):
        src_len = src.shape[0]
        batch_size = trg.shape[1]
        trg_vocab_size = self.output_size

        encoder_hidden = torch.zeros([2, batch_size, self.hidden_size]).to(device) 
        cell_state = torch.zeros([2, batch_size, self.hidden_size]).to(device)

        encoder_outputs = torch.zeros(src_len, batch_size, self.hidden_size * 2).to(device)

        for i in range(src_len):
            encoder_output, encoder_hidden, cell_state = self.encoder(src[i], encoder_hidden, cell_state)
            encoder_outputs[i] = encoder_output

        decoder_hidden = encoder_hidden[:self.decoder.n_layers]
        decoder_cell = torch.zeros_like(decoder_hidden).to(device)

        outputs = torch.zeros(self.max_len, batch_size, trg_vocab_size).to(device)

        if teacher_force:
            trg_len = trg.shape[0]
            for i in range(1, trg_len):
                decoder_input = trg[i-1]
                output, decoder_hidden, decoder_cell = self.decoder(decoder_input, decoder_hidden, decoder_cell, encoder_outputs)
                outputs[i] = output
        else:
            decoder_input = trg[0]
            for i in range(1, self.max_len):
                output, decoder_hidden, decoder_cell = self.decoder(decoder_input, decoder_hidden, decoder_cell, encoder_outputs)
                outputs[i] = output
                top1 = output.argmax(1)
                decoder_input = top1.unsqueeze(1)

        return outputs


input_dim = len(question_field.vocab)
output_dim = len(answer_field.vocab)
hidden_dim = 512
dropout = 0.2

model = Seq2Seq(input_dim, hidden_dim, output_dim, dropout).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

PAD_IDX = answer_field.vocab.stoi['<pad>']
criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

num_epochs = 1

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    for batch in train_iterator:
        optimizer.zero_grad()
        outputs = model(batch.src, batch.trg, batch_size)
        
        outputs_flatten = outputs[1:].view(-1, outputs.shape[-1])
        trg_flatten = batch.trg[1:].view(-1)
        
        loss = criterion(outputs_flatten, trg_flatten)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    print(f"Epoch: {epoch + 1}; Training Loss: {epoch_loss/len(train_iterator)}")
