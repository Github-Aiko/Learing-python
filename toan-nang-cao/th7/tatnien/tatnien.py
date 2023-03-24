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
with open('TATNIEN.INP', 'r') as f:
    n, m, x = map(int, f.readline().split())
    graph = [[] for _ in range(n)]
    for _ in range(m):
        a, b, t = map(int, f.readline().split())
        graph[a-1].append((b-1, t))

# Tìm đường đi ngắn nhất từ mỗi đại lý đến X
distances = dijkstra(graph, x-1)

# Tìm thời gian cần để các nhân viên quay trở lại bữa tiệc
max_time = 0
for i in range(n):
    if i == x-1:
        continue
    time = distances[i] + dijkstra(graph, i)[x-1]
    max_time = max(max_time, time)

# Ghi kết quả ra file output
with open('TATNIEN.OUT', 'w') as f:
    f.write(str(max_time))
