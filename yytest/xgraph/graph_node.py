from .type import (ActiMode, AggrMode, DataType, OpType,
                           ParameterSyncType, PoolType, enum_to_int,
                           enum_to_str, int_to_enum, str_to_enum)
from enum import Enum
import torch
import math
import numpy as np

DEBUG_MODE=False

IR_DELIMITER = "; "
INOUT_NODE_DELIMITER = ','


class Comparator(Enum):
    EQ = 0
    GEQ = 1

class Node():
    def __init__(self, node):
        self.name = node.name
        self.op_type = None
        self._ir_string = None
        self.in_edges = []
        self.out_edges = []
        self.node_id = None
        self.fullsizeoutput = None

    def __repr__(self):
        return f"{type(self).__name__} / [{self.name}]"

    def assert_num_args(self, num_args, cmp):
        if cmp == Comparator.EQ:
            assert len(self.innodes) == num_args, \
                f"{enum_to_str(OpType, self.op_type)} expects {num_args}" \
                "arguments"
        elif cmp == Comparator.GEQ:
            assert len(self.innodes) >= num_args, \
                f"{enum_to_str(OpType, self.op_type)} expects at least " \
                f"{num_args} arguments"
    
    @property
    def ir_string(self):
        """Returns the string representation of the node."""
        if self._ir_string is None:
            self.parse()
        return self._ir_string
    
    def parse(self):
        """Parses the node to populate ``self._ir_string`` with a string
        representation."""
        raise NotImplementedError
    
    def parse_inoutnodes(self, nodes):
        """Parses the given input or output nodes, and returns a string
        representation."""
        if nodes is None:
            return ""
        assert type(nodes) is list or type(nodes) is tuple or \
            type(nodes) is dict
        return INOUT_NODE_DELIMITER.join([node.name for node in nodes]) + \
            INOUT_NODE_DELIMITER
    
    def xforward():
        raise NotImplementedError


class FunctionNode(Node):
    def __init__(self, node):
        super().__init__(node)
        self.innodes = node.args
        self.outnodes = node.users
        self.function = node.target
        self.kwargs = node.kwargs
        self.op_type = None
    @staticmethod
    def construct_node(node):
        name = torch.typename(node.target)
        
        print("FunctionNode construct_node : ", node, node.target, node.args, )
        if name.find("add") >= 0:
         if FunctionNode.is_elemwise_op(node):
                return AddNode(node)
        else:
            assert 0, "Unknown `add()` usage with `innodes`: " \
                f"{node.innodes}"
    @staticmethod
    def is_elemwise_op(node):
        """
        Args:
            node (torch.fx.node.Node): ``torch.fx`` node to check.
        """
        innodes = node.args
        if len(innodes) != 2:
            return False
        return type(innodes[0]) is torch.fx.node.Node and \
            type(innodes[1]) is torch.fx.node.Node

class AddNode(FunctionNode):
    def __init__(self, node):
        super().__init__(node)
        self.op_type = OpType.ADD
        self.assert_num_args(2, Comparator.EQ)
        if DEBUG_MODE:
            self.parse()
            print("\nInit ADDNode", node)
            print("Inodes ", type(self.innodes), " : ", self.parse_inoutnodes(self.innodes))
            print("Outodes ",  type(self.outnodes), " : ", self.parse_inoutnodes(self.outnodes))

    def parse(self):
        s = [self.name]
        s.append(self.parse_inoutnodes(self.innodes))
        s.append(self.parse_inoutnodes(self.outnodes))
        s.append(enum_to_str(OpType, self.op_type))
        self._ir_string = IR_DELIMITER.join(s)
    
    def shapeInferRule(self, input_tensors):
        assert len(input_tensors) == 2
        assert input_tensors[0] == input_tensors[1]
        return input_tensors[0]
    
    def genGraph(self, node_id, node_to_output):
        #Op = OpMeta(name=self.name, id=node_id, type=OpType.ADD)
        self.node_id = node_id
        out_edges = []
        for x, y in self.outnodes.items():
            out_edges.append(x.name)
        in_edges = []
        if type(self.innodes) is tuple:
            for i in range(len(self.innodes)):             
                in_edges.append(self.innodes[i].name)
        else:
            for x, y in self.innodes.items():
                in_edges.append(x.name)
                
        self.in_edges = in_edges
        self.out_edges = out_edges
        input_tensor_0 = node_to_output[self.innodes[0].name]
        input_tensor_1 = node_to_output[self.innodes[1].name]
        output_tensor = self.shapeInferRule([input_tensor_0, input_tensor_1])
        self.fullsizeoutput = output_tensor
        return self, output_tensor

    def xforward(self, input_tensors):
        assert len(input_tensors) == 2
        print("call default Add ", input_tensors[0].shape)
        return torch.add(input_tensors[0], input_tensors[1])

    def toTModule(self):
        NotImplemented
    
        
class InputNode(Node):
    def __init__(self, node):
        super().__init__(node)
        self.innodes = None
        self.outnodes = node.users
        self.op_type = OpType.INPUT
        
        if DEBUG_MODE:
            self.parse()
            print("\nInit InputNode", node)
            print("Inodes ", type(self.innodes), " : ", self.parse_inoutnodes(self.innodes))
            print("Outodes ",  type(self.outnodes), " : ", self.parse_inoutnodes(self.outnodes))

    def parse(self):
        s = [self.name]
        s.append(self.parse_inoutnodes(self.innodes))
        s.append(self.parse_inoutnodes(self.outnodes))
        s.append(enum_to_str(OpType, self.op_type))
        self._ir_string = IR_DELIMITER.join(s) 
    
    def genGraph(self, input_tensors, input_index, node_id):
        self.node_id = node_id
        out_edges = []
        for x, y in self.outnodes.items():
            out_edges.append(x.name)
        self.out_edges = out_edges
        return self, input_tensors[input_index]
    
    def xforward(self, input_tensors):
        return input_tensors
    
    # def toTModule(self):
    #     return modules.sentinels.ImportLayer()
    

class OutputNode(Node):
    def __init__(self, node):
        super().__init__(node)
        self.innodes = node.args
        self.outnodes = None
        self.op_type = OpType.OUTPUT
        if DEBUG_MODE:
            self.parse()
            print("\nInit OutputNode", node)
            print(self.ir_string)
            # print("Inodes ", self.innodes, " : ", self.parse_inoutnodes(self.innodes))
            # print("Outodes ",  self.outnodes, " : ", self.parse_inoutnodes(self.outnodes))
    
    def parse(self):
        # TODO: Assumes only one output
        self.assert_num_args(1, Comparator.EQ)
        s = [self.name]
        if type(self.innodes[0]) is tuple:
            innodes = self.innodes[0]
            s.append(self.parse_inoutnodes(innodes))
            s.append(self.parse_inoutnodes(self.outnodes))
        else:
            s.append(self.parse_inoutnodes(self.innodes))
            s.append(self.parse_inoutnodes(self.outnodes))
        s.append(enum_to_str(OpType, self.op_type))
        self._ir_string = IR_DELIMITER.join(s)
    
    def genGraph(self, node_id, node_to_output):
        self.node_id = node_id
        in_edges = []
        # print("OutputNode : ", self.innodes)
        # print("OutputNode : ", self.innodes[0], type(self.innodes[0]))
        # print("OutputNode : ", self.innodes[0][0], type(self.innodes[0][0]))

        if type(self.innodes[0]) is tuple:
            t_innodes = self.innodes[0]
            for i in range(len(t_innodes)):             
                in_edges.append(t_innodes[0].name)
        else:
            for x, y in self.innodes.items():
                in_edges.append(x.name)
                
        self.in_edges =  in_edges
        return self
    def xforward(self, input_tensors):
        return input_tensors



class ModuleNode(Node):
    def __init__(self, node, module):
        super().__init__(node)
        self.innodes = node.args
        self.outnodes = node.users
        self.module = module

    @staticmethod
    def construct_node(node, module):
        if type(module) is torch.nn.modules.conv.Conv2d:
            return Conv2dNode(node, module)
        elif type(module) is torch.nn.modules.linear.Linear:
            return LinearNode(node, module)
        elif type(module) is torch.nn.modules.pooling.MaxPool2d:
            return Pool2dNode(node, module, PoolType.POOL_MAX)
        elif type(module) is torch.nn.modules.pooling.AvgPool2d:
            return Pool2dNode(node, module, PoolType.POOL_AVG)
        elif type(module) is torch.nn.modules.batchnorm.BatchNorm2d:
            return BatchNorm2dNode(node, module)
        elif type(module) is torch.nn.modules.flatten.Flatten:
            return FlattenNode(node, module)
        elif type(module) is torch.nn.modules.activation.ReLU:
            return ReLUMNode(node, module)
        elif type(module) is torch.nn.modules.pooling.AdaptiveAvgPool2d:
            return AdaptivePool2dNode(node, module, PoolType.POOL_AVG)
        # elif type(module) is torch.nn.modules.dropout.Dropout:
        #     return DropoutMNode(node, module)
        # elif type(module) is torch.nn.modules.activation.Sigmoid:
        #     return SigmoidNode(node, module)
        # elif type(module) is torch.nn.modules.activation.Tanh:
        #     return TanhMNode(node, module)
        # elif type(module) is torch.nn.modules.activation.ELU:
        #     return ELUNode(node, module)
        # elif type(module) is torch.nn.modules.activation.Softmax:
        #     return SoftmaxMNode(node, module)
        # elif type(module) is torch.nn.modules.normalization.LayerNorm:
        #     return LayerNormNode(node, module)
        # elif type(module) is torch.nn.Identity:
        #     return IdentityNode(node, module)
        # elif type(module) is torch.nn.GELU:
        #     return GeluMNode(node, module)
        # elif isinstance(module, torch.nn.Embedding):
        #     return EmbeddingNode(node, module)
        else:
            assert 0, f"Unknown module: {module}"


class FlattenNode(ModuleNode):
    def __init__(self, node, module):
        super().__init__(node, module)
        self.op_type = OpType.FLAT
        self.assert_num_args(1, Comparator.EQ)

    def parse(self):
        s = [self.name]
        s.append(self.parse_inoutnodes(self.innodes))
        s.append(self.parse_inoutnodes(self.outnodes))
        s.append(enum_to_str(OpType, self.op_type))
        self._ir_string = IR_DELIMITER.join(s)
    
    def shapeInferRule(self, input_tensor):        
        temp = input_tensor[self.module.start_dim:]
        flatten_product = np.prod(temp)
        non_flatten = input_tensor[:self.module.start_dim]
        output_shape = non_flatten + (flatten_product,)
        return output_shape
    
    def genGraph(self, node_id, node_to_output):
        self.node_id = node_id
        out_edges = []
        for x, y in self.outnodes.items():
            out_edges.append(x.name)
        in_edges = []
        if type(self.innodes) is tuple:
            for i in range(len(self.innodes)):
                in_edges.append(self.innodes[i].name)
        else:
            for x, y in self.innodes.items():
                in_edges.append(x.name)
        self.in_edges = in_edges
        self.out_edges = out_edges
        input_tensor = node_to_output[self.innodes[0].name]
        output_tensor = self.shapeInferRule(input_tensor)
        self.fullsizeoutput = output_tensor
        return self, output_tensor
    
    def xforward(self, input_tensors):
        assert len(input_tensors) == 1
        print("call default Flatten ", input_tensors[0].shape)
        return self.module.forward(input_tensors[0])


class LinearNode(ModuleNode):
    def __init__(self, node, module):
        super().__init__(node, module)
        self.op_type = OpType.LINEAR
        self.acti_mode = ActiMode.AC_MODE_NONE
        self.assert_num_args(1, Comparator.EQ)

    def parse(self):
        s = [self.name]
        s.append(self.parse_inoutnodes(self.innodes))
        s.append(self.parse_inoutnodes(self.outnodes))
        s.append(enum_to_str(OpType, self.op_type))
        s.append(str(self.module.out_features))
        s.append(str(enum_to_int(ActiMode, ActiMode.AC_MODE_NONE)))
        if self.module.bias is not None:
            s.append("1")
        else:
            s.append("0")
        self._ir_string = IR_DELIMITER.join(s)
    
    def shapeInferRule(self, input_tensor):
        assert len(input_tensor) == 2
        output_shape = (input_tensor[0], self.module.out_features)
        return output_shape
    
    def genGraph(self, node_id, node_to_output):
        self.node_id = node_id
        out_edges = []
        for x, y in self.outnodes.items():
            out_edges.append(x.name)
        in_edges = []
        if type(self.innodes) is tuple:
            for i in range(len(self.innodes)):
                in_edges.append(self.innodes[i].name)
        else:
            for x, y in self.innodes.items():
                in_edges.append(x.name)
        self.in_edges = in_edges
        self.out_edges = out_edges
        input_tensor = node_to_output[self.innodes[0].name]
        output_tensor = self.shapeInferRule(input_tensor)
        self.fullsizeoutput = output_tensor
        return self, output_tensor

    def xforward(self, input_tensors):
        assert len(input_tensors) == 1
        print("call default FC")
        return self.module.forward(input_tensors[0])

# from tiling import modules
# from tiling.computation.representation import LayerInfo, TileInfo

class Conv2dNode(ModuleNode):
    def __init__(self, node, module):
        super().__init__(node, module)
        self.op_type = OpType.CONV2D
        self.acti_mode = ActiMode.AC_MODE_NONE
        self.assert_num_args(1, Comparator.EQ)
        if DEBUG_MODE:
            self.parse()
            print("\nInit Conv2dNode", node, module)
            print("Inodes ", type(self.innodes), " : ", self.parse_inoutnodes(self.innodes))
            print("Outodes ",  type(self.outnodes), " : ", self.parse_inoutnodes(self.outnodes))
        
    def parse(self):
        s = [self.name]
        s.append(self.parse_inoutnodes(self.innodes))
        s.append(self.parse_inoutnodes(self.outnodes))
        s.append(enum_to_str(OpType, self.op_type))
        s.append(str(self.module.out_channels))
        s.append(str(self.module.kernel_size[0]))
        s.append(str(self.module.kernel_size[1]))
        s.append(str(self.module.stride[0]))
        s.append(str(self.module.stride[1]))
        s.append(str(self.module.padding[0]))
        s.append(str(self.module.padding[1]))
        s.append(str(enum_to_int(ActiMode, ActiMode.AC_MODE_NONE)))
        s.append(str(self.module.groups))
        if self.module.bias is not None:
            s.append("1")
        else:
            s.append("0")
        self._ir_string = IR_DELIMITER.join(s)

    def shapeInferRule(self, input_tensor):
        stride = self.module.stride
        pad = self.module.padding
        N = input_tensor[0]
        C = input_tensor[1]
        H = input_tensor[2] 
        W = input_tensor[3] 
        RS = self.module.kernel_size[0]
        K = self.module.out_channels
        H = math.floor((H+2*pad[0]-(RS-1)-1)/stride[0]+1)
        W = math.floor((W+2*pad[1]-(RS-1)-1)/stride[1]+1)
        output_shape = (N, K, H, W)
        return output_shape

    def genGraph(self, node_id, node_to_output):
        self.node_id = node_id
        out_edges = []
        for x, y in self.outnodes.items():
            out_edges.append(x.name)
        in_edges = []
        if type(self.innodes) is tuple:
            for i in range(len(self.innodes)):
                in_edges.append(self.innodes[i].name)
        else:
            for x, y in self.innodes.items():
                in_edges.append(x.name)
        self.in_edges = in_edges
        self.out_edges = out_edges
        input_tensor = node_to_output[self.innodes[0].name]
        output_tensor = self.shapeInferRule(input_tensor)
        self.fullsizeoutput = output_tensor
        return self, output_tensor

    def xforward(self, input_tensors):
        assert len(input_tensors) == 1
        print("call default conv2d")
        return self.module.forward(input_tensors[0])

    # def toTModule(self):
    #     return modules.Conv2d(self.module.in_channels, self.module.out_channels, \
    #                                  kernel_size=self.module.kernel_size[0], padding=self.module.padding[0])

        # add_link
        # layer_info = LayerInfo(number_tiles=1,  forward_function=None)
        # tile_info =  TileInfo(
        #         0,
        #         layer_info,
        #     )
        # return self.module.tiled_forward(tile_info, [input_tensors], None, None)


class AdaptivePool2dNode(ModuleNode):
    def __init__(self, node, module, pool_type):
        super().__init__(node, module)
        self.op_type = OpType.POOL2D
        self.pool_type = pool_type
        self.acti_mode = ActiMode.AC_MODE_NONE
        self.assert_num_args(1, Comparator.EQ)
        
       
        if DEBUG_MODE:
            self.parse()
            print("\nInit AdaptivePool2dNode", node, module)
            print("Inodes ", type(self.innodes), " : ", self.parse_inoutnodes(self.innodes))
            print("Outodes ",  type(self.outnodes), " : ", self.parse_inoutnodes(self.outnodes))
        

    def parse(self):
        s = [self.name]
        s.append(self.parse_inoutnodes(self.innodes))
        s.append(self.parse_inoutnodes(self.outnodes))
        s.append(enum_to_str(OpType, self.op_type))
        s.append(str(self.module.output_size))
        s.append(str(enum_to_int(PoolType, self.pool_type)))
        s.append(str(enum_to_int(ActiMode, ActiMode.AC_MODE_NONE)))
        self._ir_string = IR_DELIMITER.join(s)
        
    def shapeInferRule(self, input_tensor):
        N = input_tensor[0]
        K = input_tensor[1]
        H = self.module.output_size[0]
        W = self.module.output_size[1]

        output_shape = (N, K, H, W)
        return output_shape
    
    def genGraph(self, node_id, node_to_output):
        self.node_id = node_id
        out_edges = []
        for x, y in self.outnodes.items():
            out_edges.append(x.name)
        in_edges = []
        if type(self.innodes) is tuple:
            for i in range(len(self.innodes)):
                in_edges.append(self.innodes[i].name)
        else:
            for x, y in self.innodes.items():
                in_edges.append(x.name)
        self.in_edges = in_edges
        self.out_edges = out_edges
        input_tensor = node_to_output[self.innodes[0].name]
        output_tensor = self.shapeInferRule(input_tensor)
        self.fullsizeoutput = output_tensor
        return self, output_tensor
    
    # def toTModule(self):
    #     return  modules.sentinels.AdaptiveAvgPool2d(self.module.output_size)


     
        
# class Pool2dNode(ModuleNode):
#     def __init__(self, node, module, pool_type):
#         super().__init__(node, module)
#         self.op_type = OpType.POOL2D
#         self.pool_type = pool_type
#         self.acti_mode = ActiMode.AC_MODE_NONE
#         self.assert_num_args(1, Comparator.EQ)

#         if DEBUG_MODE:
#             self.parse()
#             print("\nInit Pool2dNode", node, module, pool_type)
#             print("Inodes ", self.innodes, " : ", self.parse_inoutnodes(self.innodes))
#             print("Outodes ",  self.outnodes, " : ", self.parse_inoutnodes(self.outnodes))

            
#     def parse(self):
#         s = [self.name]
#         s.append(self.parse_inoutnodes(self.innodes))
#         s.append(self.parse_inoutnodes(self.outnodes))
#         s.append(enum_to_str(OpType, self.op_type))
#         # FIXME MaxPool2d supports ceil_mode
#         s.append(str(self.module.kernel_size))
#         s.append(str(self.module.stride))
#         s.append(str(self.module.padding))
#         s.append(str(enum_to_int(PoolType, self.pool_type)))
#         s.append(str(enum_to_int(ActiMode, ActiMode.AC_MODE_NONE)))
#         self._ir_string = IR_DELIMITER.join(s)

#     def shapeInferRule(self, input_tensor):
#         stride = self.module.stride
#         pad = self.module.padding
#         N = input_tensor[0]
#         C = input_tensor[1]
#         H = input_tensor[2] 
#         W = input_tensor[3] 
#         RS = self.module.kernel_size[0]
#         H = math.floor((H+2*pad-(RS-1)-1)/stride[0]+1)
#         W = math.floor((W+2*pad-(RS-1)-1)/stride[1]+1)
#         output_shape = (N, C, H, W)
#         return output_shape


#     def genGraph(self, node_to_output):
#         input_tensor = node_to_output[self.innodes[0].name]
#         output_tensor = self.shapeInferRule(input_tensor)
#         return output_tensor
      
# class BatchNorm2dNode(ModuleNode):
#     def __init__(self, node, module):
#         super().__init__(node, module)
#         self.op_type = OpType.BATCH_NORM
#         self.acti_mode = ActiMode.AC_MODE_NONE
#         self.assert_num_args(1, Comparator.EQ)

#         if DEBUG_MODE:
#             self.parse()
#             print("\nInit BatchNorm2dNode", node, module)
#             print("Inodes ", self.innodes, " : ", self.parse_inoutnodes(self.innodes))
#             print("Outodes ",  self.outnodes, " : ", self.parse_inoutnodes(self.outnodes))

#     def parse(self):
#         s = [self.name]
#         s.append(self.parse_inoutnodes(self.innodes))
#         s.append(self.parse_inoutnodes(self.outnodes))
#         s.append(enum_to_str(OpType, self.op_type))

#         s.append(str(enum_to_int(ActiMode, ActiMode.AC_MODE_NONE)))
#         self._ir_string = IR_DELIMITER.join(s)

#     def shapeInferRule(self, input_tensor):
#         N = input_tensor[0]
#         C = input_tensor[1]
#         H = input_tensor[2] 
#         W = input_tensor[3] 
#         output_shape = (N, C, H, W)
#         return output_shape


#     def genGraph(self, node_to_output):
#         input_tensor = node_to_output[self.innodes[0].name]
#         output_tensor = self.shapeInferRule(input_tensor)
#         return output_tensor

# class ReLUMNode(ModuleNode):
#     def __init__(self, node, module):
#         super().__init__(node, module)
#         self.op_type = OpType.RELU
#         self.acti_mode = ActiMode.AC_MODE_RELU
#         self.assert_num_args(1, Comparator.EQ)

#         if DEBUG_MODE:
#             self.parse()
#             print("\n----->>  Init ReLUMNode", node, module)
#             print("Inodes ", self.innodes, " : ", self.parse_inoutnodes(self.innodes))
#             print("Outodes ",  self.outnodes, " : ", self.parse_inoutnodes(self.outnodes))
#             print("--------<<")
            
#     def parse(self):
#         s = [self.name]
#         s.append(self.parse_inoutnodes(self.innodes))
#         s.append(self.parse_inoutnodes(self.outnodes))
#         s.append(enum_to_str(OpType, self.op_type))

#         s.append(str(enum_to_int(ActiMode, ActiMode.AC_MODE_NONE)))
#         self._ir_string = IR_DELIMITER.join(s)

#     def shapeInferRule(self, input_tensor):
#         N = input_tensor[0]
#         C = input_tensor[1]
#         H = input_tensor[2] 
#         W = input_tensor[3] 
#         output_shape = (N, C, H, W)
#         return output_shape


#     def genGraph(self, node_to_output):
#         input_tensor = node_to_output[self.innodes[0].name]
#         output_tensor = self.shapeInferRule(input_tensor)
#         return output_tensor
