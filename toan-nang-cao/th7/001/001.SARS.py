# Đọc dữ liệu từ file vào
with open('SARS.INP', 'r') as f:
    n, k = map(int, f.readline().split())
    employees = []
    for i in range(n):
        employees.append(list(map(int, f.readline().split()))[1:])

# Tạo mảng visited và danh sách suspected
visited = [False] * n
suspected = []

# Hàm DFS để tìm các người có thể bị nhiễm bệnh
def dfs(employee_number):
    visited[employee_number] = True
    suspected.append(employee_number + 1)
    for e in employees[employee_number]:
        if not visited[e - 1]:
            dfs(e - 1)

# Gọi hàm DFS với nhân viên đầu tiên bị nhiễm bệnh
dfs(k - 1)

# Đếm số lượng người có thể bị nhiễm bệnh và lưu thông tin về những người đó vào danh sách suspected
num_suspected = len(suspected)
suspected.sort()

# Xuất kết quả ra file
with open('SARS.OUT', 'w') as f:
    f.write(str(num_suspected) + '\n')
    for s in suspected:
        f.write(str(s) + ' ')
