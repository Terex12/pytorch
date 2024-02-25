from .graph_node import InputNode, LinearNode, FlattenNode, OutputNode
from .type import OpType
import copy

 # A recursive function used by topologicalSort
def dfsUtil(graph, v, visited, stack, isBlock):
    #print("v: ", v)
    visited[v] = True
    if isBlock[v]:
        stack.append(v)
        return
    # Recur for all the vertices adjacent to this vertex
    for i in graph.transpose_graph[v]:
        #print(v, " ni: ", i)
        if visited[i] == False:
            dfsUtil(graph, i, visited, stack, isBlock)
    # Push current vertex to stack which stores result
    stack.append(v)

def DFS(graph, isBlock):
    # Mark all the vertices as not visited
    visited = [False]  * graph.V
    stack = []
    all_segment = []
    for i in range(graph.V-1, 0, -1):
        #("outer : ", i)
        if visited[i] == False:
            dfsUtil(graph, i, visited, stack, isBlock)
            # make reverse order
            # tmp_list = []
            # while len(stack) > 0:
            #     tmp_list.append(stack.pop())
            all_segment.append(copy.deepcopy(stack))
            stack = []
    return all_segment
    

def findSegments(graph):
    '''
    :param graph: graph
    :param memCapacity: device global mem
    :return:
    '''
    # Yufan:: I assume the single output of the segment
    
    isBlock = [False]*graph.V 
    for i in range(0, graph.V):
        node = graph.nodes[i]
        if isinstance(node, InputNode) or isinstance(node, LinearNode) or \
            isinstance(node, FlattenNode)or isinstance(node, OutputNode):
            isBlock[i] = True
    
    #print(isBlock)
    all_segment = DFS(graph, isBlock)
    all_segment.reverse()
    print("all segments ", all_segment)    
    return all_segment, isBlock

    
    
    
    
    
            