import heapq

def dijkstra(graph, start, end):
    # Khởi tạo dist để lưu trữ khoảng cách ngắn nhất từ đỉnh bắt đầu đến các đỉnh còn lại
    dist = {vertex: float('inf') for vertex in graph}
    dist[start] = 0
    
    # Khởi tạo heap và đưa đỉnh bắt đầu vào heap
    heap = [(0, start)]
    
    while heap:
        # Lấy đỉnh có khoảng cách ngắn nhất từ heap
        (cost, current) = heapq.heappop(heap)
        
        # Nếu đỉnh hiện tại là đỉnh đích, trả về khoảng cách ngắn nhất
        if current == end:
            return cost
        
        # Kiểm tra các đỉnh kề của đỉnh hiện tại
        for neighbor in graph[current]:
            # Tính khoảng cách từ đỉnh bắt đầu đến đỉnh kề qua đỉnh hiện tại
            distance = dist[current] + graph[current][neighbor]
            
            # Nếu khoảng cách mới tìm được ngắn hơn khoảng cách đã lưu trữ, cập nhật lại dist
            if distance < dist[neighbor]:
                dist[neighbor] = distance
                
                # Đưa đỉnh kề vào heap để xét tiếp
                heapq.heappush(heap, (distance, neighbor))
    
    # Nếu không có đường đi từ đỉnh bắt đầu đến đỉnh đích, trả về -1
    return -1

graph = {
    'A': {'B': 1, 'C': 4},
    'B': {'D': 3},
    'C': {'D': 2},
    'D': {}
}

start = 'A'
end = 'D'
shortest_distance = dijkstra(graph, start, end)
print(f"The shortest distance from {start} to {end} is {shortest_distance}")
