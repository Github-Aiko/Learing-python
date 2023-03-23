def check_path_blocked(x1, y1, x2, y2, obstacles):
    """
    Kiểm tra xem tia chạm từ điểm (x1, y1) đến điểm (x2, y2) có bị cản trở bởi các chướng ngại vật hay không
    Sử dụng thuật toán Bresenham để kiểm tra
    """
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    if x1 < x2:
        sx = 1
    else:
        sx = -1
    if y1 < y2:
        sy = 1
    else:
        sy = -1
    err = dx - dy
    while x1 != x2 or y1 != y2:
        if (x1, y1) in obstacles:
            return True
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x1 += sx
        if e2 < dx:
            err += dx
            y1 += sy
    return False

# Đọc dữ liệu từ tệp MIRROR.INP
with open("MIRROR.INP", "r") as f:
    n = int(f.readline().strip())
    a = []
    for i in range(n):
        row = f.readline().strip()
        a.append(row)

# Bước 1: Tìm vị trí của các ô vuông bao quanh các ô có dấu thăng
mirrors = set()
for i in range(1, n - 1):
    for j in range(1, n - 1):
        if a[i][j] == "#" and (
            a[i - 1][j] == "." or a[i + 1][j] == "." or a[i][j - 1] == "." or a[i][j + 1] == "."
        ):
            mirrors.add((i, j))

# Bước 2: Tính diện tích của gương cần mua và loại bỏ các ô vuông đã đặt gương
needed_mirrors = set()
walls = set()
for x, y in mirrors:
    if check_path_blocked(x, y, 0, 0, mirrors - {(x, y)}):
        walls.add((x, y))
    else:
        needed_mirrors.add((x, y))

count_guong = len(needed_mirrors)
for x, y in needed_mirrors:
    # Loại bỏ các ô vuông được bao quanh bởi gương
    for i in range(x - 1, x + 2):
        for j in range(y - 1, y + 2):
            if (i, j) in mirrors:
                mirrors.remove((i, j))

# Bước 3: Tính diện tích của các bức tường ở phía trong của nhà gương
count_tuong = len(mirrors)

# Ghi kết quả vào tệp MIRROR.OUT
with open("MIRROR.OUT", "w") as f:
    f.write("{} {}\n".format(count_guong * 9, count_tuong * 9))
