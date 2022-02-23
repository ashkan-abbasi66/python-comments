import numpy as np
import torch as tr
import torch.nn as nn

def getNumParams(params):
    numParams, numTrainable = 0, 0
    for param in params:
        npParamCount = np.prod(param.data.shape)
        numParams += npParamCount
        if param.requires_grad:
            numTrainable += npParamCount
    return numParams, numTrainable

# Using list
class Module1(nn.Module):
    def __init__(self, dIn, dOut, numLayers):
        super(Module1, self).__init__()
        self.layers = []
        for i in range(numLayers - 1):
            self.layers.append(nn.Conv2d(in_channels=dIn, out_channels=dIn, kernel_size=1))
        self.layers.append(nn.Conv2d(in_channels=dIn, out_channels=dOut, kernel_size=1))

    def forward(self, x):
        y = x
        for i in range(len(self.layers)):
            y = self.layers[i](y)
        return y

# Using nn.ModuleList
class Module2(nn.Module):
    def __init__(self, dIn, dOut, numLayers):
        super(Module2, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(numLayers - 1):
            self.layers.append(nn.Conv2d(in_channels=dIn, out_channels=dIn, kernel_size=1))
        self.layers.append(nn.Conv2d(in_channels=dIn, out_channels=dOut, kernel_size=1))

    def forward(self, x):
        y = x
        for i in range(len(self.layers)):
            y = self.layers[i](y)
        return y

def main():
    x = tr.randn(1, 7, 30, 30)

    module1 = Module1(dIn=7, dOut=13, numLayers=10)
    y1 = module1(x)
    print(y1.shape) # (1, 13, 30, 30)
    print(getNumParams(module1.parameters())) # Prints (0, 0)

    module2 = Module2(dIn=7, dOut=13, numLayers=10)
    y2 = module2(x)
    print(getNumParams(module2.parameters())) # Print (608, 608)
    print(y2.shape) # (1, 13, 30, 30)

if __name__ == "__main__":
    main()