import networkx as nx
import numpy as np
import torch
from torch import nn
from torch.optim import Adam




class GraphConvNet(nn.Module):
    def __init__(self, in_features, n_class, normed_laplacian):
        super(GraphConvNet, self).__init__()
        self._normed_laplacian = torch.from_numpy(normed_laplacian).float()
        self.add_module(
            "main",
            nn.Sequential(
                nn.Dropout(),
                GraphConvolutionLayer(
                    in_features = in_features,
                    out_features = 256,
                    normed_laplacian = self._normed_laplacian,
                    loss = nn.ReLU
                ),
                nn.Dropout(),
                GraphConvolutionLayer(
                    in_features = 256,
                    out_features = n_class,
                    normed_laplacian = self._normed_laplacian
                ),

            )
        )
        self._loss = nn.CrossEntropyLoss()

    def forward(self, input):
        return self._modules["main"](input)

    def fit(
            self, input, label,
            indices_train, n_epoch = 5000,
            verbose = True
    ):
        self.train()

        input = torch.from_numpy(input).float()
        label = torch.from_numpy(label).long()
        optimizer = Adam(self.parameters(), lr = 0.001)


        for e in range(n_epoch):
            output = self.forward(input)[indices_train]
            loss = self._loss(output, label)

            if verbose:
                print("Epoch " + str(e) + " with loss " + str(loss.item()))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return

    def predict(self, input):
        self.eval()
        op = self.forward(torch.from_numpy(input).float()).detach().numpy()
        return np.argmax(op, axis = -1)


    def save_weight(self, PATH):
        torch.save(self.state_dict(), PATH + "/model.pt")







class GraphConvolutionLayer(nn.Sequential):
    def __init__(self, in_features, out_features, normed_laplacian, loss = None):
        super(GraphConvolutionLayer, self).__init__()
        self.add_module(
            "main",
            nn.Linear(in_features = in_features, out_features = out_features),
        )

        if loss is not None:
            self.add_module(
                "loss",
                loss()
            )
            # self.add_module("norm", nn.BatchNorm1d(num_features = out_features))





        self._normed_laplacian = normed_laplacian

    def forward(self, input):
        return super(GraphConvolutionLayer, self).forward(torch.mm(self._normed_laplacian, input))

