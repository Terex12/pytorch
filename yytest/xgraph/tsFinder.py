import sympy as sp
from sympy import Poly
import copy
from sympy import lambdify


def findTileSize(segment_list, graphIR, mem_cap, block_op_list):
    sg_indx = 0
    waiting_src_cnt = {}
    for i in graphIR.graph.keys():
        waiting_src_cnt[i] = len(graphIR.graph[i])
    
    print("waiting_src_cnt: ", waiting_src_cnt)
    for sg in segment_list:
        op_required_input_slice = {}
        print("\n\n+++------- sg: " , sg_indx)
        sg_indx += 1
        tileSizePerSeg(sg, graphIR, op_required_input_slice, block_op_list, copy.deepcopy(waiting_src_cnt), mem_cap)
        
def maxMerrge(exp_list, Ts):
    assert len(exp_list) == 2
    test_val = 1024
    f1 = lambdify(Ts, exp_list[0], "math") 
    f2 = lambdify(Ts, exp_list[1], "math") 
    eval1 = f1(test_val)
    eval2 = f2(test_val)
    if eval1 > eval2:
        return exp_list[0]
    else:
        return exp_list[1]


def tileSizePerSeg(segment, graphIR, op_required_input_slice, block_op_list, waiting_src_cnt, mem_cap):
    #only consider square tile
    Ts = sp.Symbol('T', positive=True)
    i= 0
    slice_exp = [Ts+i, Ts]
    ready_q = []
    ready_q.append(segment[0])
    op_required_input_slice[segment[0]] = [graphIR.nodes[segment[0]].get_require_input_slice(slice_exp)]

    #BFS ready queue
    while (len(segment) > 0):
        cur_node = segment.pop(0)
        for i in graphIR.transpose_graph[cur_node]:
            print(cur_node, " --> ", i)
            
            slice_exp = graphIR.nodes[i].get_require_input_slice(op_required_input_slice[cur_node][0])
            if op_required_input_slice.get(i) == None:
                op_required_input_slice[i] = [slice_exp]
                print(i, " get ", op_required_input_slice[i])
            else:
                op_required_input_slice[i].append(slice_exp)
                # feval and merge expression
                print("two edge in merge node: ", op_required_input_slice[i])
                if len(op_required_input_slice[i]) > 1:
                    m_exp = maxMerrge(op_required_input_slice[i], Ts)
                    op_required_input_slice[i] = [m_exp]
                    print("after merge: ", op_required_input_slice[i])
            
            
            waiting_src_cnt[i] -= 1
            if waiting_src_cnt[i] == 0:
                print(i, " get all incoming push to ready q")
                ready_q.append(i)
            else:
                print(i, " still need ", waiting_src_cnt[i], " incoming")
    
    sum_expression = (Ts*Ts) # init final output 
    for k, v in op_required_input_slice.items():
        exp = v[0][0]*v[0][1]
        if k == 0:
            continue # no need for input node
        if sum_expression == None:
            sum_expression = exp
        else:
            sum_expression += exp
    
    print("\n\nsum_expression: ", sum_expression)
    
    sum_expression = sum_expression - mem_cap
    print("\n\nsum_expression: ", sum_expression)
    solution = sp.solve_poly_inequality(Poly(sum_expression), ">=")  
    print("poly solver : ", solution, type(solution))
    
    solution = sp.solve(sum_expression, Ts) 
    print("regular : ", solution, type(solution)) 
    #return op_required_input_slice
    
