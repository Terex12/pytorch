import sympy as sp
from sympy import Poly
import copy
from sympy import lambdify




def runPerSeg(segment, graphIR, tt):
    waiting_src_cnt = {}
    # OPT:: need only once
    for i in graphIR.graph.keys():
        waiting_src_cnt[i] = len(graphIR.graph[i])
    #only consider square tile
    ready_q = []
    # add assert the first op is input
    ready_q.append(segment[0])
    
    awaiting_inputs = {}
    if segment[0] == 0:
        awaiting_inputs[segment[0]] = tt
    else:
        awaiting_inputs[segment[0]] = [tt] #TODO input type??
    
   
    #BFS ready queue
    while (len(segment) > 0):
        cur_node_id = segment.pop(0)
        
        need_inputs = awaiting_inputs[cur_node_id]
        
        print("poping ", cur_node_id)
        output = graphIR.nodes[cur_node_id].xforward(need_inputs) 
        print("output shape  ", output.shape)
        for i in graphIR.graph[cur_node_id]:
            print(cur_node_id, " --> ", i)
        
            if awaiting_inputs.get(i) == None:
                awaiting_inputs[i] = []
                awaiting_inputs[i].append(output)
            else:
                awaiting_inputs[i].append(output)
        
            waiting_src_cnt[i] -= 1
            if waiting_src_cnt[i] == 0:
                print(i, " get all incoming push to ready q")
                ready_q.append(i)
            else:
                print(i, " still need ", waiting_src_cnt[i], " incoming")
    
    return output