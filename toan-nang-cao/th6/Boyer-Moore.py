def boyer_moore_search(text, pattern):
    m = len(pattern)
    n = len(text)
    if m > n:
        return -1

    # Tạo bảng băm cho bad character rule
    skip = {}
    for k in range(m-1):
        skip[pattern[k]] = m - k - 1

    # Tạo bảng băm cho good suffix rule
    bmGs, bmBc = boyer_moore_table(pattern)
    
    # Tìm kiếm từ phải sang trái
    i = m - 1
    j = m - 1
    while j >= 0 and i < n:
        if text[i] == pattern[j]:
            i -= 1
            j -= 1
        else:
            # Sử dụng bad character rule và good suffix rule để xác định vị trí tiếp theo cần kiểm tra
            if text[i] in skip:
                i += skip[text[i]]
            else:
                i += m
            j = m - 1

    if j == -1:
        return i + 1
    else:
        return -1

def boyer_moore_table(pattern):
    m = len(pattern)
    bmGs = [0] * m
    bmBc = [-1] * 256
    last = 0

    # Tạo bảng băm cho bad character rule
    for i in range(m):
        bmBc[ord(pattern[i])] = i

    # Tạo bảng băm cho good suffix rule
    for i in range(m-1, -1, -1):
        if is_prefix(pattern, i+1):
            last = i + 1
        bmGs[i] = last - i + m - 1

    for i in range(m):
        slen = suffix_length(pattern, i)
        bmGs[m-1-slen] = m-1-i+slen

    return bmGs, bmBc

def is_prefix(pattern, p):
    m = len(pattern)
    suffix = pattern[p:]
    return suffix == pattern[:m-p]

def suffix_length(pattern, p):
    m = len(pattern)
    i = 0
    while pattern[p-i] == pattern[m-1-i] and i < p:
        i += 1
    return i

# Sử dụng hàm boyer_moore_search để tìm kiếm chuỗi con "pattern" trong chuỗi "text"
text = "This is a sample text"
pattern = "sample"
result = boyer_moore_search(text, pattern)
print(result)