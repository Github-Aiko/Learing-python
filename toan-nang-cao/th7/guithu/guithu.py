import heapq

class Graph:
    def __init__(self, vertices):
        self.V = vertices
        self.graph = [[] for i in range(vertices)]
        
    def add_edge(self, u, v, w):
        self.graph[u].append((v, w))
        self.graph[v].append((u, w))
        
    def shortest_path(self, start):
        heap = [(0, start)]
        visited = [False] * self.V
        dist = [float('inf')] * self.V
        dist[start] = 0
        
        while heap:
            (d, u) = heapq.heappop(heap)
            if visited[u]:
                continue
            visited[u] = True
            for v, weight in self.graph[u]:
                if dist[u] + weight < dist[v]:
                    dist[v] = dist[u] + weight
                    heapq.heappush(heap, (dist[v], v))
        return dist
    
# đọc dữ liệu từ file
with open('GUITHU.INP', 'r', encoding='utf-8') as f:
    n, m, k = map(int, f.readline().split())
    graph = Graph(n)
    for i in range(m):
        l, p, t = map(int, f.readline().split())
        graph.add_edge(l - 1, p - 1, t)
    reqs = []
    for i in range(k):
        a, b = map(int, f.readline().split())
        reqs.append((a - 1, b - 1))
        
# tính thời gian gửi thư
with open('GUITHU.OUT', 'w') as f:
    dist1 = graph.shortest_path(0)
    for a, b in reqs:
        dist2 = graph.shortest_path(a)
        dist3 = graph.shortest_path(b)
        min_time = min(dist1[a] + dist2[b] + dist3[0], dist1[b] + dist2[a] + dist3[0])
        f.write(f"{min_time}\n")


