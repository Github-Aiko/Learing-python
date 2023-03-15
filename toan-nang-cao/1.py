def quick_sort(arr):
    # Nếu mảng chỉ có 1 phần tử thì trả về mảng
    if len(arr) <= 1:
        return arr
    # Chọn phần tử pivot
    pivot = arr[len(arr) // 2]
    # Tạo mảng left chứa các phần tử bé hơn pivot
    left = [x for x in arr if x < pivot]
    # Tạo mảng middle chứa các phần tử bằng pivot
    middle = [x for x in arr if x == pivot]
    # Tạo mảng right chứa các phần tử lớn hơn pivot
    right = [x for x in arr if x > pivot]
    # Gọi đệ quy lại hàm quick_sort để sắp xếp các phần tử trong mảng left, right
    return quick_sort(left) + middle + quick_sort(right)


def selection_sort(arr):
    # Lấy số phần tử của mảng
    n = len(arr)
    # Lặp qua các phần tử của mảng
    for i in range(n):
        # Gán phần tử nhỏ nhất là phần tử đầu tiên
        min_idx = i
        # Lặp qua các phần tử còn lại của mảng
        for j in range(i + 1, n):
            # Nếu có phần tử nào nhỏ hơn phần tử đầu tiên thì gán phần tử đó là phần tử nhỏ nhất
            if arr[j] < arr[min_idx]:
                min_idx = j
        # Đổi chỗ phần tử đầu tiên và phần tử nhỏ nhất
        arr[i], arr[min_idx] = arr[min_idx], arr[i]
    # Trả về mảng đã sắp xếp
    return arr


def find_first_rank(S, T):
    # Sắp xếp mảng S
    sorted_S = selection_sort(S)
    # Gán biến low là 0
    low, high = 0, len(sorted_S) - 1
    # Lặp khi low <= high
    while low <= high:
        # Tìm giá trị mid
        mid = (low + high) // 2
        # Nếu phần tử ở vị trí mid nhỏ hơn phần tử đầu tiên của T thì gán low = mid + 1
        if sorted_S[mid] < T[0]:
            low = mid + 1
        # Nếu phần tử ở vị trí mid lớn hơn phần tử đầu tiên của T thì gán high = mid - 1
        elif sorted_S[mid] > T[0]:
            high = mid - 1
        # Nếu phần tử ở vị trí mid bằng phần tử đầu tiên của T
        else:
            # Nếu mảng con của mảng S bắt đầu từ mid và có độ dài bằng độ dài của T thì trả về mid
            if sorted_S[mid : mid + len(T)] == T:
                return mid
            # Nếu không thì gán low = mid + 1
            else:
                low = mid + 1
    # Trả về -1 nếu không tìm thấy
    return -1


def main():
    S = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    T = [5, 6, 7]
    print(f"Xếp hạng đầu tiên: {find_first_rank(S, T)}")


if __name__ == "__main__":
    main()
