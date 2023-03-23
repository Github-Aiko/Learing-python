from math import trunc
from collections import defaultdict
from heapq import heappush, heappop

class Graph:
    def __init__(self, n):
        self.n = n
        self.adj = defaultdict(list)

    def add_edge(self, u, v, w):
        self.adj[u].append((v, w))
        self.adj[v].append((u, w))

    def kruskal(self, k):
        # sort edges by weight
        edges = []
        for u in range(self.n):
            for v, w in self.adj[u]:
                if u < v:
                    edges.append((w, u, v))
        edges.sort()

        # initialize union-find
        parent = list(range(self.n))
        rank = [0] * self.n

        # find operation in union-find
        def find(u):
            if parent[u] != u:
                parent[u] = find(parent[u])
            return parent[u]

        # union operation in union-find
        def union(u, v):
            pu, pv = find(u), find(v)
            if rank[pu] < rank[pv]:
                parent[pu] = pv
            elif rank[pu] > rank[pv]:
                parent[pv] = pu
            else:
                parent[pv] = pu
                rank[pu] += 1

        # compute MST using Kruskal's algorithm
        mst = []
        for w, u, v in edges:
            if find(u) != find(v):
                union(u, v)
                mst.append((u, v, w))

        # compute total length of added edges
        added_len = sum(w for u, v, w in mst) * 1000
        if added_len > k:
            return -1
        return trunc(added_len)

# read input
with open('LAPDIEN.INP', 'r', encoding='utf-8') as f:
    n = int(f.readline())
    m = int(f.readline())
    k = float(f.readline())
    pos = [tuple(map(float, f.readline().split())) for _ in range(n)]
    edges = [tuple(map(float, f.readline().split())) for _ in range(m)]

# build graph
graph = Graph(n)
for u, v in edges:
    w = ((pos[u-1][0] - pos[v-1][0]) ** 2 + (pos[u-1][1] - pos[v-1][1]) ** 2) ** 0.5
    graph.add_edge(u-1, v-1, w)

# compute MST and output result
with open('LAPDIEN.OUT', 'w') as f:
    res = graph.kruskal(k)
    f.write(f"{res}\n")
