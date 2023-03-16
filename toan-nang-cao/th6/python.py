def naive_search(pattern, text):
    m = len(pattern)
    n = len(text)
    # Duyệt qua từng ký tự của chuỗi đầu vào
    for i in range(n - m + 1):
        j = 0
        # So sánh từng ký tự của chuỗi mẫu với chuỗi đầu vào từ vị trí i
        while j < m and text[i + j] == pattern[j]:
            j += 1
        # Nếu chuỗi mẫu được tìm thấy, trả về vị trí đầu tiên của nó trong chuỗi đầu vào
        if j == m:
            return i
    # Nếu không tìm thấy chuỗi mẫu, trả về giá trị -1
    return -1

def build_T(pattern):
    m = len(pattern)
    T = [0] * m
    T[0] = -1
    j = -1
    # Xây dựng bảng T
    for i in range(1, m):
        while j >= 0 and pattern[i] != pattern[j + 1]:
            j = T[j]
        if pattern[i] == pattern[j + 1]:
            j += 1
        T[i] = j
    return T

def kmp_search(pattern, text):
    m = len(pattern)
    n = len(text)
    T = build_T(pattern)
    j = -1
    # Duyệt qua từng ký tự của chuỗi đầu vào
    for i in range(n):
        while j >= 0 and text[i] != pattern[j + 1]:
            j = T[j]
        if text[i] == pattern[j + 1]:
            j += 1
        # Nếu chuỗi mẫu được tìm thấy, trả về vị trí đầu tiên của nó trong chuỗi đầu vào
        if j == m - 1:
            return i - m + 1
    # Nếu không tìm thấy chuỗi mẫu, trả về giá trị -1
    return -1

def main():
    pattern = "ABCDABD"
    text = "ABCABCDABABCDABCDABDE"
    print("Using Naive algorithm")
    index = naive_search(pattern, text)
    if index == -1:
        print("Pattern not found in text")
    else:
        print("Pattern found at index", index)
    print("Using KMP algorithm:")
    index = kmp_search(pattern, text)
    if index == -1:
        print("Pattern not found in text")
    else:
        print("Pattern found at index:", index)

if __name__ == '__main__':
    main()
