from torch.fx import symbolic_trace
from collections import OrderedDict
from .graph_node import ModuleNode, InputNode, OutputNode, FunctionNode
from .graph import NodeGraph
#FunctionNode
DEBUG = False

def query_nodeid(node, node_list):
    for n in node_list:
        if n.name == node:
            #print("query_nodeid: ", node, " ", n.name)
            return n.node_id

def create_internal_graph(traced, input_tensors, input_index, verbose=False):
    # traced = symbolic_trace(network)
    cur_module = traced.graph.owning_module
#     print("cur_module : ", cur_module)
#     print(" : ", type( cur_module))
    # Convert the fx graph to an internal graph representation
    name_to_module = {}
    for name, module in cur_module.named_modules():
        name_to_module[name] = module
    #print("traced.graph : ", traced.graph)

    input_index = 0
    node_id = 0 #current only for forward inference nodes.
    
    opnode_list = []
    node_to_output = OrderedDict()
    # traverse the graph and first generate all vertices(NodeMeta)
    for fx_node in traced.graph.nodes:
        if DEBUG:
            print("\nfx_node++ ", fx_node, )
        if fx_node.op == "call_module":
            module_name = fx_node.target
            module = name_to_module[module_name]
            node = ModuleNode.construct_node(fx_node, module)
            Node, node_output = node.genGraph(node_id, node_to_output)
            if DEBUG:
                print("Meta:: ", Node)
            node_id += 1
            opnode_list.append(Node)
        elif fx_node.op == "placeholder":
            node = InputNode(fx_node)
            #input tensor --> fake NodeMEta
            Node, node_output = node.genGraph(input_tensors, input_index, node_id)
            input_index += 1
            if DEBUG:
                print("Meta:: ", Node)
            node_id += 1
            opnode_list.append(Node)
        elif fx_node.op == "call_function" or fx_node.op == "call_method":
            print("call_function : ", fx_node, )
            node = FunctionNode.construct_node(fx_node)
            Node, node_output = node.genGraph(node_id, node_to_output)
            input_index += 1
            if DEBUG:
                print("Meta:: ", Node)
            node_id += 1
            opnode_list.append(Node)
        elif fx_node.op == "output":
            node = OutputNode(fx_node)
            Node = node.genGraph(node_id, node_to_output)
            opnode_list.append(Node)
            
        # elif fx_node.op == "get_attr":
        #     node = AttributeNode(fx_node, self.model)
        else:
            assert 0, f"Unknown operator type: {fx_node.op}"
        
        if node_output is not None:
            node_to_output[node.name] = node_output
    
    opg = NodeGraph(len(opnode_list), opnode_list)
    op_id_lookup = {}
    for node in opnode_list:
        if node.node_id not in op_id_lookup:
            op_id_lookup[node.node_id] = node
        for out in node.out_edges:
            out_id = query_nodeid(out, opnode_list)
            opg.addEdge(node.node_id, out_id)
            opg.add_t_Edge(out_id, node.node_id)
    
    print("------------------")
    opg.printAllNodes()
    opg.displayGraph()
    opg.displayTransposeGraph()
    print("------------------")
    print(opnode_list)
   
    
    return opg