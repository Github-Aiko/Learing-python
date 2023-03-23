# Định nghĩa hàm kiểm tra một lô có an toàn hay không
def is_safe(r, c, radars, m, n):
    # Kiểm tra xem lô có nằm trong vùng hoạt động của radar không
    for radar in radars:
        if abs(r - radar[0]) <= 1 and abs(c - radar[1]) <= 1:
            return False
    # Kiểm tra xem lô có bị tàu thuyền phát hiện khi đi từ bên ngoài vùng biển vào không
    for i in range(max(1, r - 1), min(m, r + 1) + 1):
        for j in range(max(1, c - 1), min(n, c + 1) + 1):
            if abs(r - i) <= 1 and abs(c - j) <= 1:
                continue
            detected = False
            for radar in radars:
                if abs(i - radar[0]) <= 1 and abs(j - radar[1]) <= 1:
                    detected = True
                    break
            if not detected:
                return False
    return True

# Đọc dữ liệu từ file input
with open('RADAR.INP', 'r', encoding='utf-8') as f:
    m, n = map(int, f.readline().split())
    k = int(f.readline().strip())
    radars = []
    for i in range(k):
        radars.append(list(map(int, f.readline().split())))

# Duyệt qua từng lô trên vùng biển và kiểm tra xem lô đó có an toàn hay không
safe = [[False] * (n + 2) for _ in range(m + 2)]
for i in range(1, m + 1):
    for j in range(1, n + 1):
        if is_safe(i, j, radars, m, n):
            safe[i][j] = True
            # Duyệt các lô lân cận để đánh dấu là không an toàn
            for r in range(max(1, i - 1), min(m, i + 1) + 1):
                for c in range(max(1, j - 1), min(n, j + 1) + 1):
                    safe[r][c] = False

# Đếm số lượng lô an toàn và ghi kết quả ra file output
count = sum(sum(row[1:-1]) for row in safe[1:-1])
with open('RADAR.OUT', 'w', encoding='utf-8') as f:
    f.write(str(count))
