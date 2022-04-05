# Import libraries
import torch
import torch.nn as nn
import numpy as np
import random

#====================================================================================#
class Parameters:
    def __init__(self, data_dict):
        for k, v in data_dict.items():
            exec("self.%s=%s" % (k, v))

#====================================================================================#
class RnnClassifier(nn.Module):
    #-------------------------
    def __init__(self, device, params):
        super(RnnClassifier, self).__init__()
        np.random.seed(1)
        random.seed(1)
        torch.manual_seed(1)
        torch.cuda.manual_seed(1)
        torch.cuda.manual_seed_all(1)
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        #-------------------------
        self.params = params
        self.device = device
        #-------------------------
        # Embedding layer
        self.word_embeddings = nn.Embedding(self.params.vocab_size, self.params.embed_dim)
        #-------------------------
        # Calculate number of directions
        self.num_directions = 2 if self.params.bidirectional == True else 1
        #-------------------------
        self.rnn = nn.LSTM(self.params.embed_dim,
                       self.params.rnn_hidden_dim,
                       num_layers=self.params.num_layers,
                       bias = True,
                       dropout = self.params.dropout_lstm,
                       bidirectional=self.params.bidirectional,
                       batch_first=self.params.batch_first)

        for param in self.rnn.parameters():
            if len(param.shape)>=2:
                nn.init.orthogonal_(param.data)
            else:
                nn.init.normal_(param.data)

        x = torch.ones((64, self.params.vocab_size), dtype= torch.long)
        #-------------------------
        self.flatten    = nn.Flatten()        
        self.fc1        = nn.Linear(self.shape_data(x), 128)
        self.leakyReLU  = nn.LeakyReLU()
        self.drop       = nn.Dropout(self.params.dropout_fc)
        self.fc2        = nn.Linear(128, 1)
        self.sigmoid    = nn.Sigmoid()
    
    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::#
    def init_hidden(self, batch_size):
        return (torch.zeros(self.params.num_layers * self.num_directions, batch_size, self.params.rnn_hidden_dim).to(self.device),
                torch.zeros(self.params.num_layers * self.num_directions, batch_size, self.params.rnn_hidden_dim).to(self.device))
    
    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::#
    def shape_data(self, inputs):
        batch_size, seq_len = inputs.shape
        # Push through embedding layer #
        X = self.word_embeddings(inputs).permute(1, 0, 2)
        # Push through RNN layer
        self.rnn.flatten_parameters()
        rnn_output, self.hidden = self.rnn(X)
        # print(rnn_output.shape)
        output = self.flatten(rnn_output.permute(1, 0, 2))
        return output.shape[1]

    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::#
    def forward(self, inputs):
        #-------------------------
        batch_size, seq_len = inputs.shape
        #-------------------------
        # Passing through embedding layer
        embed_out = self.word_embeddings(inputs).permute(1, 0, 2)
        #-------------------------
        # Passing through RNN layer
        rnn_output, self.hidden = self.rnn(embed_out, self.init_hidden(batch_size))
        rnn_flatten             = self.flatten(rnn_output.permute(1, 0, 2))
        fc1_out                 = self.fc1(rnn_flatten)
        fc1_out                 = self.leakyReLU(fc1_out)
        fc1_out                 = self.drop(fc1_out)
        fc2_out                 = self.fc2(fc1_out)
        output                  = self.sigmoid(fc2_out)
        #-------------------------
        return output