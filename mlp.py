
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import numpy as np


import torch
import torch.nn.functional as F
from torch import nn

from easydict import EasyDict
from torchsummary import summary

import numpy as np

import torch
import torch.nn.functional as F
from torch import nn


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, output_activation = "softmax", dropout=0.5):
        super().__init__()
        self.hidden_dim =hidden_dim
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1) # [256, 256]
        print()
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])) 
        # (308, 256, 256), (256, 256, 2)

        if output_activation == "softmax":
            self.output_activation = nn.Softmax(dim=1)
        else:
            raise NotImplementedError("Only softmax is supported for now.")
        
        self.bns = nn.ModuleList(nn.BatchNorm1d(self.hidden_dim) for _ in range(self.num_layers - 1)) 
        self.dropout = nn.Dropout(p=dropout, inplace=False)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = self.bns[i](layer(x)) if i < self.num_layers - 1 else layer(x)
            x = F.relu(x)
            if i < self.num_layers - 1:
                
                x = self.dropout(x)

        if self.output_activation == "softmax":
            x = self.output_activation(x)
        return x

class DNN(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""
    def __init__(self, input_dim=128, \
                hidden_dim=256, \
                output_dim=2, \
                num_layers= 3, \
                output_activation="softmax", \
                epochs=10, \
                batch_size=32, \
                learning_rate=0.001, \
                momentum=0.9, \
                weight_decay=0.0001, \
                dropout=0.5, \
                device="cuda"):
        super().__init__()
        
        # train
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.device = device

        # build MLP
        self.mlp = MLP(input_dim, hidden_dim, output_dim, num_layers, output_activation, dropout)
        summary(self.mlp.cuda(), (308, ))
        # self.mlp.cpu()
        # loss
        self.criterion = torch.nn.CrossEntropyLoss()

        # optimizer
        self.optimizer = torch.optim.Adam(self.mlp.parameters(), lr=self.learning_rate)
        
    
    def fit(self, train_loader, val_loader):
        
        hist_train_loss = []
        hist_val_loss = []
        hist_train_f1 = []
        hist_val_f1 = []
        for epoch_i in range(self.epochs):
            losses = []
            y_preds = []
            for (batch_idx, batch) in enumerate(train_loader):
                batch_X, batch_y = batch
                y_pred = self.mlp(batch_X)

                loss = self.criterion(y_pred, batch_y)
                losses.append(loss)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                y_preds.append(y_pred)
            
            # validation
            if epoch_i%10 == 0:
                val_losses = []
                y_preds_val = []
                for (batch_idx, batch) in enumerate(val_loader):
                    batch_X, batch_y = batch
                    
                    y_pred = self.mlp(batch_X)

                    loss = self.criterion(y_pred, batch_y)
                    val_losses.append(loss)
                    y_preds_val.append(y_pred)
                
                y_preds = torch.cat(y_preds, dim=0)
                y_preds_val = torch.cat(y_preds_val, dim=0)
                y_preds = torch.argmax(y_preds, dim=1)
                y_preds_val = torch.argmax(y_preds_val, dim=1)


                f1 = f1_score(y_preds.cpu().detach().numpy(), train_loader.dataset.y)
                val_f1 = f1_score(y_preds_val.cpu().detach().numpy(), val_loader.dataset.y)

                train_loss = torch.stack(losses).mean()
                val_loss = torch.stack(val_losses).mean()
            
                print("[!] epoch_i: {}, loss: {}, val_loss: {}, f1: {}, val_f1: {}".format(epoch_i, \
                                                                                        train_loss, \
                                                                                        val_loss,\
                                                                                        f1,\
                                                                                        val_f1))
                hist_train_loss.append(train_loss)
                hist_val_loss.append(val_loss)
                hist_train_f1.append(f1)
                hist_val_f1.append(val_f1)

        
        hist_train_loss = torch.stack(hist_train_loss).cpu().detach().numpy()      
        hist_val_loss = torch.stack(hist_val_loss).cpu().detach().numpy()
        
        return self.mlp, hist_train_loss, hist_val_loss, hist_train_f1, hist_val_f1
    
    def predict(self, test_loader):
        y_preds = []
        for (_, batch) in enumerate(test_loader):
            batch_X, _ = batch
            y_pred = self.mlp(batch_X)
            y_preds.append(y_pred)
        y_preds = torch.cat(y_preds, dim=0)
        return y_preds.cpu().detach().numpy()
    
### Dataset ###
class SimpleDataset(Dataset):
    def __init__(self, X, y, device="cuda"):
        self.X = X
        self.y = y
        self.device = device
        return
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        X = torch.from_numpy(np.asarray(self.X[index])).float().cuda()
        y = torch.from_numpy(np.asarray(self.y[index])).type(torch.LongTensor).cuda()
        
        return X, y

def main(args):


    X = torch.randn(100, 128)
    y = torch.from_numpy(np.concatenate([np.ones((50, )), np.zeros((50, ))]))
    print("X.shape", X.shape)
    print("y.shape", y.shape)
    
    ### Data Splitting ###
    X_train, X_val, y_train, y_val = train_test_split(X_train,
                                                      y,
                                                      test_size = 0.25,
                                                      stratify = y,
                                                      shuffle = True,
                                                      random_state = 33)

    train_sds = SimpleDataset(X_train, y_train)
    train_dataloader = torch.utils.data.DataLoader(train_sds, batch_size=args.batch_size, drop_last=args.drop_last)
    val_sds = SimpleDataset(X_val, y_val)
    val_dataloader = torch.utils.data.DataLoader(val_sds, batch_size=args.batch_size)
    
    # model
    model = DNN(input_dim=args.input_dim, \
                hidden_dim=args.hidden_dim, \
                output_dim=args.output_dim, \
                num_layers=args.num_layers, \
                output_activation=args.output_activation, \
                epochs=args.epochs, \
                batch_size=args.batch_size, \
                learning_rate=args.learning_rate, \
                momentum=args.momentum, \
                weight_decay=args.weight_decay, \
                dropout = args.dropout,\
                device="cuda")

    # training MLP
    _, hist_train, hist_val, hist_train_f1, hist_val_f1 = model.fit(train_dataloader, val_dataloader)
    print("Val f1", hist_val_f1[-1])

    plt.plot(hist_train, label="train_loss", color="red")
    plt.plot(hist_val, label="val_loss", color="blue")
    plt.plot(hist_train_f1, label="train_f1", color="green")
    plt.plot(hist_val_f1, label="val_f1", color="orange")
    plt.legend()
    plt.show()
    # savefig_path = os.path.join(log_path, f"test_log_{seed_ind}_{ex_version}", "learning_curve.png")
    # plt.savefig(savefig_path)

    # val_preds = model.predict(val_dataloader)


if __name__ == "__main__":
    from easydict import EasyDict
    
    args = EasyDict()

    # general
    args.seed = 0

    # dataset
    args.shuffle = True
    args.drop_last = False

    # model architecture
    # args.input_dim = fmask_X_train.shape[1]
    args.hidden_dim = 128
    args.output_dim = 2
    args.num_layers = 2
    args.output_activation = "softmax"
    
    # train
    args.batch_size  = 64
    args.epochs  = 500
    args.learning_rate= 0.0005
    args.momentum=0.9
    args.weight_decay=0.0001
    args.dropout=0.9


    main(args)
    # test
    
    # model = MLP(input_dim=128, hidden_dim=256, output_dim=512, num_layers= 3, output_activation="softmax")
    # print(model)

    # # y = model(x)

    # print(y.shape)
    # print(y)
    # print("done")
    
