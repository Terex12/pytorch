from typing import List
import torch
from torch import _dynamo as torchdynamo
import logging
import torch.nn as nn
from torch.nn.parameter import Parameter
import numpy as np


# from xgraph.segAlgo import findSegments
# from xgraph.tsFinder import findTileSize
from xgraph.internal_graph import create_internal_graph
# from xgraph.runEngine import runPerSeg
# from xgraph.formCustModule import formModule



# #eager
#torch._logging.set_logs(dynamo=logging.WARNING, graph_code=True ,bytecode=True)

#inductor
#torch._logging.set_logs(dynamo=logging.WARNING, graph_code=True ,bytecode=True)


from torch.fx import symbolic_trace


def sstep_backend(gm: torch.fx.GraphModule, example_inputs):
    print("sstep backend called with FX graph:", type(gm.graph))
    input_shape = (64, 64, 112, 112)
  
    graphIR = create_internal_graph(gm, [input_shape], 0, True)
    print(example_inputs[0].shape)
    gm.graph.print_tabular()
    return gm.forward



"""Network define START()"""
def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    conv = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=0, bias=False)
    
    print("$$---", conv.extra_repr())
    val = np.arange(in_planes*out_planes*3*3)
    w1 = np.reshape(val, (out_planes, in_planes, conv.kernel_size[0], conv.kernel_size[1]))
    conv.weight = Parameter(torch.Tensor(w1))
    return conv


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.conv2 = conv3x3(planes, planes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        return out

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.block1 = BasicBlock(64, 64)
        # self.mxp = nn.MaxPool2d((2,2), (2,2))
        self.block2 = BasicBlock(64, 64)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.flatten = nn.Flatten(start_dim=1)
        self.fc = nn.Linear(3136, 2)
    
    def forward(self, x): 
        """Network Forward START()"""  
        out = self.block1(x)
        # out = self.mxp(out)
        out = self.block2(out)
        out = self.avgpool(out)
        out = self.flatten(out)
        out = self.fc(out)
        return out
        """Network Forward END()"""  

"""Network define END()"""

def p2():
    b = 64
    c = 64
    h = 112
    w = 112
    tt = torch.range(1, b*c*h*w).reshape(b,c,h,w)
    
    #@torchdynamo.optimize("inductor")
    #@torch.compile(backend="inductor")
    #@torch.compile(backend="sstep")
    network = Net()
    #out = network(tt)
    # def toy_example(arg1):
        
    #     # residual = arg1
    #     # c1 = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False)
    #     # r1 = nn.ReLU()
    #     # c2 = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False)
    #     # r2 = nn.ReLU()

    #     # out = c1(arg1)
    #     # out = r1(out)
    #     # out = c2(out)
    #     # out = r2(out)
    #     # out = torch.add(out, residual)
    #     return out

    # print(torch._dynamo.explain(toy_example)(arg1))
    # toy_example(tt)
    torch._dynamo.reset()

    opt_model = torch.compile(network, backend=sstep_backend)
    opt_model(tt)


if __name__ == "__main__":
    p2()