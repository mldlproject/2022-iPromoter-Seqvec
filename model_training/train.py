# Import libraries
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import TensorDataset
import numpy as np
from training_func import *
from model import *
from utils import *

#====================================================================================#
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

#====================================================================================#
# Parameter
params = { 'rnn_hidden_dim': 128,
           'num_layers': 1,
           'bidirectional': True,
           'batch_first': False,
           'dropout_lstm': 0,
           'dropout_fc': 0,
           'vocab_size': 298,
           'embed_dim': 64,
           'label_size': 1}

params           = Parameters(params)
type_padded_data = 'random' # w_weight, no_weight
type_data        = 'hs_TA'# hs_nonTA, mm_TA, mm_nonTA
n_epoch          = 51
lr_rate          = 0.0001
batch_size       = 64

# Model
model = RnnClassifier(device, params).to(device)
# Loss function
criteron = nn.BCELoss()
# Adam Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=lr_rate)

#====================================================================================#
# Loading data
model_path = './output/save_model'
data_path  = './processed_data/{}/{}'.format(type_data[0:2], type_data[3:])

data_train = np.load(data_path + '/train_{}.npy'.format(type_data))
y_train    = np.load(data_path + '/train_label_{}.npy'.format(type_data))

data_val   = np.load(data_path + '/val_{}.npy'.format(type_data))
y_val      = np.load(data_path + '/val_label_{}.npy'.format(type_data))

data_test  = np.load(data_path + '/test_{}.npy'.format(type_data))
y_test     = np.load(data_path + '/test_label_{}.npy'.format(type_data))

# Shape data 
print('Data Training: ', data_train.shape)
print('Pos: {}, Neg: {}'.format(np.sum(y_train), len(y_train) - np.sum(y_train)))
print('Data Validation: ', data_val.shape)
print('Pos: {}, Neg: {}'.format(np.sum(y_val), len(y_val) - np.sum(y_val)))
print('Data Test: ', data_test.shape)
print('Pos: {}, Neg: {}'.format(np.sum(y_test), len(y_test) - np.sum(y_test)))

train_dataset       = TensorDataset(Tensor(data_train).long(), Tensor(y_train))
validation_dataset  = TensorDataset(Tensor(data_val).long(), Tensor(y_val))
test_dataset        = TensorDataset(Tensor(data_test).long(), Tensor(y_test))

training_loader     = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, shuffle = True, worker_init_fn = np.random.seed(0))
validation_loader   = torch.utils.data.DataLoader(validation_dataset, batch_size = batch_size, shuffle = False)
test_loader         = torch.utils.data.DataLoader(test_dataset,  batch_size = batch_size, shuffle = False)

#====================================================================================#                 
#                                 Training                                           #           
#====================================================================================#
training_loss_list, validation_loss_list, test_loss_list = [], [], []
test_prob_pred, val_prob_pred = [], []

print("Training model for data  {}".format(type_data))
val_loss_check = 10
for epoch in range(n_epoch):
    #-------------------------
    train_results = train(epoch, model, criteron, optimizer, device, training_loader)
    #-------------------------
    validation_results = validate(epoch, model, criteron, device, validation_loader)
    if validation_results[0] < val_loss_check:
        val_loss_check = validation_results[0]
        # torch.save(model_path + '/model_{}'.format(type_data))
    #-------------------------
    test_results = test(epoch, model, criteron, device, test_loader)
    #------------------------- 
    training_loss_list.append(train_results)
    #-------------------------
    validation_loss_list.append(validation_results[0])
    val_prob_pred.append(validation_results[1])
    #-------------------------
    test_loss_list.append(test_results[0])
    test_prob_pred.append(test_results[1])

#====================================================================================#                 
#                                 Evaluation                                         #           
#====================================================================================#
print("Performance")
test_probs  = get_prob(test_prob_pred, np.argmin(validation_loss_list))
matrix      = performance(y_test, test_probs, name= 'test_dataset_{}'.format(type_data))

val_probs   = get_prob(val_prob_pred, np.argmin(validation_loss_list))
matrix      = performance(y_val, val_probs, name= 'val_dataset_{}'.format(type_data))


