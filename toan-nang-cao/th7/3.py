import sys

class Graph:
    def __init__(self, vertices):
        self.V = vertices
        self.graph = [[0 for column in range(vertices)]
                      for row in range(vertices)]
    
    def print_mst(self, parent):
        print("Edges in the constructed MST:")
        for i in range(1, self.V):
            print(f"{parent[i]} - {i} : {self.graph[i][parent[i]]}")
            
    def min_key(self, key, mst_set):
        min_val = sys.maxsize
        for v in range(self.V):
            if key[v] < min_val and not mst_set[v]:
                min_val = key[v]
                min_index = v
        return min_index
    
    def prim(self):
        key = [sys.maxsize] * self.V
        parent = [None] * self.V
        key[0] = 0
        mst_set = [False] * self.V
        parent[0] = -1
        for i in range(self.V):
            u = self.min_key(key, mst_set)
            mst_set[u] = True
            for v in range(self.V):
                if self.graph[u][v] > 0 and not mst_set[v] and key[v] > self.graph[u][v]:
                    key[v] = self.graph[u][v]
                    parent[v] = u
        self.print_mst(parent)

g = Graph(4)
g.graph = [[0, 10, 6, 5],
           [10, 0, 0, 15],
           [6, 0, 0, 4],
           [5, 15, 4, 0]]

g.prim()
