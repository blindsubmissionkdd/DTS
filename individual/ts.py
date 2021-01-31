from vrae.vrae import VRAE
from vrae.utils import *
import numpy as np
import torch
from matplotlib import pyplot
import plotly
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from torch.autograd import Variable
from vrae.loss import *
dload = './model_dir' #download directory
hidden_size = 90
hidden_layer_depth = 1
latent_length = 20
batch_size = 32
learning_rate = 0.0005
n_epochs = 100
dropout_rate = 0.2
optimizer = 'Adam' # options: ADAM, SGD
cuda = True # options: True, False
print_every=30
clip = True # options: True, False
max_grad_norm=5
loss = 'MSELoss' # options: SmoothL1Loss, MSELoss
block = 'LSTM' # options: LSTM, GRU
X_train, X_val, y_train, y_val = open_data('data', ratio_train=0.9)

num_classes = len(np.unique(y_train))
base = np.min(y_train)  # Check if data is 0-based
if base != 0:
    y_train -= base

y_val -= base
train_dataset = TensorDataset(torch.from_numpy(X_train))
test_dataset = TensorDataset(torch.from_numpy(X_val))
sequence_length = X_train.shape[1]
number_of_features = X_train.shape[2]
vrae = VRAE(sequence_length=sequence_length,
            number_of_features = number_of_features,
            hidden_size = hidden_size,
            hidden_layer_depth = hidden_layer_depth,
            latent_length = latent_length,
            batch_size = batch_size,
            learning_rate = learning_rate,
            n_epochs = n_epochs,
            dropout_rate = dropout_rate,
            optimizer = optimizer,
            cuda = cuda,
            print_every=print_every,
            clip=clip,
            max_grad_norm=max_grad_norm,
            loss = loss,
            block = block,
            dload = dload)

vrae.fit(train_dataset, save = False)
vrae.save('vraetc110.pth')

#Traversal Plot
z_run = vrae.transform(test_dataset)
for i in range(20):
    stack = []
    for j in range(9):
        z_run[1, i] = z_run[1, i] + j - 4
        tmp = vrae.decoder(torch.from_numpy(z_run[0:32]).to(device))
        series = pd.DataFrame(data={'value': tmp.permute(1, 0, 2).cpu().data.numpy()[1, :, 0]})
        series.plot(legend=None, color='green')
        pyplot.ylim(-5, 2)
        pyplot.savefig('./gen10/tmp22'+str(i)+'+'+str(j)+'.png')
        z_run[1, i] = z_run[1, i] - j + 4

