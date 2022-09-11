import torch
from torch.nn import Linear as L


class MLP(torch.nn.Module):
    def __init__(self, m, n):
        super().__init__()
        # self.act1 = torch.nn.Sigmoid()
        self.act1 = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax(dim=1)
        self.m = m
        self.n = n
        self.h1 = L(self.m, 60)
        self.h2 = L(60, 40)
        self.output = L(40, self.n)

    def forward(self, x):
        h1 = self.h1(x)
        r1 = self.act1(h1)
        h2 = self.h2(r1)
        r2 = self.act1(h2)
        output = self.output(r2)
        # output = self.softmax(output)
        return output
