from queue import Queue

# Hàm BFS kiểm tra xem có tồn tại đường đi từ u tới v trên đồ thị G(h) hay không
def bfs(u, v, h, G, visited):
    q = Queue()
    q.put(u)
    visited[u] = True
    while not q.empty():
        x = q.get()
        if x == v:
            return True
        for y in G[x]:
            if not visited[y] and h <= a[x][y]:
                q.put(y)
                visited[y] = True
    return False

# Đọc dữ liệu từ file input
with open('TAITRONG.INP', 'r') as f:
    n, u, v = map(int, f.readline().split())
    a = [[0 for _ in range(n + 1)] for _ in range(n + 1)]
    for line in f:
        x, y, z = map(int, line.split())
        a[x][y] = z
        a[y][x] = z

# Tìm giá trị h lớn nhất để tồn tại đường đi từ u tới v trên đồ thị G(h)
left, right = 0, 10000
res = 0
while left <= right:
    mid = (left + right) // 2
    visited = [False] * (n + 1)
    if bfs(u, v, mid, G, visited):
        res = mid
        left = mid + 1
    else:
        right = mid - 1

# In kết quả ra file output
with open('TAITRONG.OUT', 'w') as f:
    f.write(str(res))
    f.write('\n')
    visited = [False] * (n + 1)
    q = Queue()
    q.put(u)
    visited[u] = True
    while not q.empty():
        x = q.get()
        f.write(str(x) + ' ')
        if x == v:
            break
        for y in range(1, n + 1):
            if not visited[y] and res <= a[x][y]:
                q.put(y)
                visited[y] = True
