import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, sizes, activation_layer):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(len(sizes)-1):
            self.layers.append(nn.Linear(sizes[i], sizes[i+1]))
            self.layers.append(activation_layer())
        
    def forward(self, x):
        output = x

        for layer in self.layers:
            output = layer(output)
        
        return output
