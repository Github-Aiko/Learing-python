import heapq

INF = 10**9

def dijkstra(graph, start):
    n = len(graph)
    dist = [INF] * n
    dist[start] = 0
    heap = [(0, start)]
    while heap:
        d, u = heapq.heappop(heap)
        if d != dist[u]:
            continue
        for v, w in graph[u]:
            if dist[v] > dist[u] + w:
                dist[v] = dist[u] + w
                heapq.heappush(heap, (dist[v], v))
    return dist

# Đọc dữ liệu từ file input
with open('TANGQUA.INP', 'r') as f:
    m, n, a, b, c = map(int, f.readline().split())
    graph = [[] for _ in range(n)]
    for _ in range(m):
        p, q, l = map(int, f.readline().split())
        graph[p-1].append((q-1, l))
        graph[q-1].append((p-1, l))

# Tìm đường đi ngắn nhất từ A, B và C đến tất cả các đỉnh
dist_a = dijkstra(graph, a-1)
dist_b = dijkstra(graph, b-1)
dist_c = dijkstra(graph, c-1)

# Tìm đường đi ngắn nhất A -> i -> B -> C -> i
min_distance = INF
for i in range(n):
    distance = dist_a[i] + dist_b[i] + dist_c[i]
    min_distance = min(min_distance, distance)

# Ghi kết quả ra file output
with open('TANGQUA.OUT', 'w') as f:
    f.write(str(min_distance))
