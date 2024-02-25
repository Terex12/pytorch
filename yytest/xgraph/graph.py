from collections import defaultdict
from bitarray import bitarray

class NodeGraph:
    def __init__(self, vertices, node_list):
        self.graph = defaultdict(list)  # dictionary containing adjacency List
        self.transpose_graph = defaultdict(list)
        self.nodes = node_list # in fact is a fakeid-node pair
        self.V = vertices  # No. of vertices
        self.tp_order = []
        assert self.V == len(self.nodes)

        self.bitset = []
        self.postbitset = []
        for i in range(0, self.V):
            bit_ar = bitarray(self.V)
            bit_ar.setall(0)
            self.bitset.append(bit_ar)
            self.domFLAG = False

            p_bit_ar = bitarray(self.V)
            p_bit_ar.setall(0)
            self.postbitset.append(p_bit_ar)
            self.postdomFLAG = False

    def addEdge(self, u, v):
        # u, v is the id of a node
        self.graph[u].append(v)

    def add_t_Edge(self, u, v):
        # u, v is the id of a node
        self.transpose_graph[u].append(v)

    # A recursive function used by topologicalSort
    def topologicalSortUtil(self, v, visited, stack):
        visited[v] = True
        # Recur for all the vertices adjacent to this vertex
        for i in self.graph[v]:
            if visited[i] == False:
                self.topologicalSortUtil(i, visited, stack)
        # Push current vertex to stack which stores result
        stack.append(v)

    def topologicalSort(self):
        # Mark all the vertices as not visited
        visited = [False] * self.V
        stack = []
        for i in range(self.V):
            if visited[i] == False:
                self.topologicalSortUtil(i, visited, stack)
        self.tp_order = stack[::-1]
        self.settingTPid2Node()

    def settingTPid2Node(self):
        for i in range(0, len(self.tp_order)):
            # use fake id to retrieve node
            node = self.nodes[self.tp_order[i]]
            node.tp_id = i


    def findDominatorUntil(self, parent_bit_set, node_id, visited):
        # If node is unvisited
        if visited[node_id] == False:
            self.bitset[node_id] = ~self.bitset[node_id]
            visited[node_id] = True

        self.bitset[node_id] &= parent_bit_set
        self.bitset[node_id][node_id] = 1

        for i in self.graph[node_id]:
            self.findDominatorUntil(self.bitset[node_id], i, visited)

    def findDominator(self):
        visited = [False] * self.V
        source_node_bit = self.bitset[0]
        source_node_bit[0] = 1
        visited[0] = True

        self.findDominatorUntil(self.bitset[0], 0, visited)
        # marking we got the dominator for all nodes in graph
        self.domFLAG = True

    def findPostDominatorUntil(self, parent_bit_set, node_id, visited):
        # If node is unvisited
        if visited[node_id] == False:
            self.postbitset[node_id] = ~self.postbitset[node_id]
            visited[node_id] = True

        self.postbitset[node_id] &= parent_bit_set
        self.postbitset[node_id][node_id] = 1

        for i in self.transpose_graph[node_id]:
            self.findPostDominatorUntil(self.postbitset[node_id], i, visited)

    def findPostDominator(self):
        # start from the last node in TP order
        if len(self.tp_order) == 0:
            self.topologicalSort()
        last_node = self.tp_order[-1]
        visited = [False] * self.V
        source_node_bit = self.postbitset[last_node]
        source_node_bit[last_node] = 1
        visited[last_node] = True

        self.findPostDominatorUntil(self.postbitset[last_node], last_node, visited)
        # marking we got the post dominator for all nodes in graph
        self.postdomFLAG = True

    def showAllDominator(self):
        for i in range(0, self.V):
            print("Dom of ", i)
            for j in range(0, self.V):
                if self.bitset[i][j] == 1:
                    print(" ", j)

    def showAllPostDominator(self):
        for i in range(0, self.V):
            print("PostDom of ", i)
            for j in range(0, self.V):
                if self.postbitset[i][j] == 1:
                    print(" ", j)

    def transposeGraph(self):
        for i in range(0, self.V):
            for j in range(len(self.graph[i])):
                self.add_t_Edge(self.graph[i][j], i)

    def displayGraph(self):
        print("Graph :: ")
        for i in range(self.V):
            print(i, "--> ", end="")
            for j in range(len(self.graph[i])):
                print(self.graph[i][j], end=" ")
            print()

    def displayTransposeGraph(self):
        print("TransposeGraph :: ")
        for i in range(self.V):
            print(i, "--> ", end="")
            for j in range(len(self.transpose_graph[i])):
                print(self.transpose_graph[i][j], end=" ")
            print()

    def settingOpDomPostDom(self):
        assert self.domFLAG
        assert self.postdomFLAG
        for i in range(0, self.V):
            node = self.nodes[i]
            dom_list = []
            pos_dom_list = []
            for j in range(0, self.V):
                if self.postbitset[i][j] == 1:
                    pos_dom_list.append(self.nodes[j])
                if self.bitset[i][j] == 1:
                    dom_list.append(self.nodes[j])

            node.setting_dominator(dom_list)
            node.setting_postdominator(pos_dom_list)

    def printAllNodes(self):
        for val in self.nodes:
            print("nid",val.node_id, ": ", val)