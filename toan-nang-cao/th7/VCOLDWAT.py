from collections import deque

# read input
n, c = map(int, input().split())
pipes = []
for i in range(c):
    e, b1, b2 = map(int, input().split())
    pipes.append((e, b1, b2))

# initialize distances
dist = [-1] * n
dist[0] = 1

# perform BFS
queue = deque([(0, 1)])
while queue:
    node, d = queue.popleft()
    for b in [pipes[node][1] - 1, pipes[node][2] - 1]:
        if dist[b] == -1:
            dist[b] = d + 1
            queue.append((b, d + 1))

# output distances
for d in dist:
    print(d)
