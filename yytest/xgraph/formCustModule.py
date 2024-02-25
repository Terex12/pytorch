from .type import OpType
from tiling import modules


def formModule(graph, segment, x):
    white_list = [OpType.CONV2D, OpType.POOL2D, OpType.RELU, OpType.ADD, OpType.INPUT]    
    id_node_map = {}
    custModule_list = []
    for node_id in segment:
        print("node_id: ", node_id)
        if graph.nodes[node_id].op_type not in white_list:
            # assert False, "not supported op type"
            return segment, False
        # elif graph.nodes[node_id].op_type == OpType.INPUT:
        #     id_node_map[node_id] = modules.sentinels.ImportLayer()
        else:
            id_node_map[node_id] = graph.nodes[node_id].toTModule()
    
    #traverse graph to add add_input_info()
    for id, node in id_node_map.items():
        #print("id: ", id, "  " , node)
        for j in graph.transpose_graph[id]: # parent node of cur_node
            #print("connect input ", id_node_map[j], " to ", node)
            node.add_input_info(id_node_map[j], 0) # not sure the second arg 
    

    for _, node in id_node_map.items():
         custModule_list.append(node)
         
    seq_wrap = modules.Sequential(*custModule_list, link_up_layers=True)

    # TODO how to add input layer for general??
    tiled_module_compiled = modules.package([x],
                                                seq_wrap,
                                                number_tiles_per_tile_dimension=[1, 1],
                                                tile_dimensions=[-2, -1])

    return tiled_module_compiled, True

