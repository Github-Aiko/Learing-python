import numpy as np
from scipy.optimize import linear_sum_assignment

## I ) Dãy con đơn điệu dài nhất
def find_lis(seq):
    n = len(seq)
    # 1. dp[i] is the length of the longest increasing subsequence in seq[:i+1]
    dp = [1] * n
    # 2. prev[i] stores the index of the previous element in the LIS ending at seq[i]
    prev = [-1] * n
    for i in range(n):
        for j in range(i):
            # 3. If seq[j] < seq[i] and there is a subsequence ending at seq[j] with
            #    length dp[j] which is longer than any other subsequence ending at seq[i]
            #    then update dp[i] and prev[i]
            if seq[j] < seq[i] and dp[j] + 1 > dp[i]:
                dp[i] = dp[j] + 1
                prev[i] = j
    # 4. The length of the longest increasing subsequence is the maximum value in dp
    lis_len = max(dp)
    # 5. Find the index of the last element in the LIS
    end_idx = dp.index(lis_len)
    # 6. Backtrack to find the actual elements of the LIS
    lis = []
    while end_idx != -1:
        lis.append(seq[end_idx])
        end_idx = prev[end_idx]
    return lis[::-1]

# Bố trí phòng họp( mất tính thứ tự so với dãy ban đầu)
def Meeting_room_arrangement(n, meetings):
    n = len(meetings)
    
    # sort the meetings by their end times
    meetings = sorted(meetings, key=lambda x: x[1])
    
    # initialize the memo table
    memo = [0] * n
    
    # fill in the memo table using dynamic programming
    for i in range(n):
        memo[i] = 1
        for j in range(i):
            if meetings[j][1] <= meetings[i][0]:
                memo[i] = max(memo[i], memo[j] + 1)
    
    # find the maximum number of meetings that can be arranged
    max_meetings = max(memo)
    
    # find the longest sequence of meetings that can be arranged
    longest_seq = []
    for i in range(n-1, -1, -1):
        if memo[i] == max_meetings:
            longest_seq.append(meetings[i])
            max_meetings -= 1
            if max_meetings == 0:
                break
    
    # reverse the longest sequence and return it
    longest_seq.reverse()
    return longest_seq

# Cho thuê máy
def Machine_rental(n, orders):
    # Sort orders by end time
    orders.sort(key=lambda x: x[1])
    
    # Initialize L array to the fees of each order
    L = [order[2] for order in orders]
    
    # Iterate through each order and find the maximum revenue
    for i in range(1, n):
        for j in range(i):
            if orders[j][1] <= orders[i][0] and L[i] < L[j] + orders[i][2]:
                L[i] = L[j] + orders[i][2]
    
    # Return the maximum revenue
    return max(L)

# Dãy tam giác bao nhau
def many_triangles(triangles):
    # Sort triangles by area
    triangles.sort(key=lambda t: t[2])
    n = len(triangles)
    # Initialize array to store longest subsequence ending at each triangle
    L = [1] * n
    # Initialize array to store previous triangle in longest subsequence
    prev = [-1] * n
    # Find longest increasing subsequence
    for i in range(1, n):
        for j in range(i):
            # Check if triangle j can be added to the subsequence ending at i
            if is_contained(triangles[j], triangles[i]) and L[j] + 1 > L[i]:
                L[i] = L[j] + 1
                prev[i] = j
    # Find index of triangle with maximum length subsequence
    max_index = 0
    for i in range(1, n):
        if L[i] > L[max_index]:
            max_index = i
    # Construct longest increasing subsequence
    subsequence = []
    while max_index != -1:
        subsequence.append(triangles[max_index])
        max_index = prev[max_index]
    subsequence.reverse()
    return subsequence

def is_contained(triangle1, triangle2):
    # Check if all vertices of triangle2 are contained within triangle1
    for vertex in triangle2[:3]:
        if not is_inside(vertex, triangle1):
            return False
    return True

def is_inside(point, triangle):
    # Check if point is inside triangle using barycentric coordinates
    # Get the coordinates of the triangle vertices
    A, B, C = triangle[:3]
    x, y = point
    # Compute denominator of the barycentric coordinates
    denominator = (B[1] - C[1]) * (A[0] - C[0]) + (C[0] - B[0]) * (A[1] - C[1])
    # Compute the barycentric coordinates of the point
    alpha = ((B[1] - C[1]) * (x - C[0]) + (C[0] - B[0]) * (y - C[1])) / denominator
    beta = ((C[1] - A[1]) * (x - C[0]) + (A[0] - C[0]) * (y - C[1])) / denominator
    gamma = 1 - alpha - beta
    # Check if the point is inside the triangle
    return 0 <= alpha <= 1 and 0 <= beta <= 1 and 0 <= gamma <= 1

# Dãy đổi dấu
def Sign_change_sequence(a, L, U):
    n = len(a) # length of the array
    L_seq = [1] * n # create a list of 1s with length n
    P_seq = [1] * n # create a list of 1s with length n
    max_len = 1 # initialize the maximum length of the sequence
    
    for i in range(1, n): # for each element in the array
        for j in range(i-L-1, -1, -1): # for each element before the current element
            if a[i] - U <= a[j] < a[i]: # if the number is in the upper range
                L_seq[i] = max(L_seq[i], P_seq[j] + 1) # add 1 to the length of the sequence
                max_len = max(max_len, L_seq[i]) # update the maximum length
            if a[i] < a[j] <= a[i] + U: # if the number is in the lower range
                P_seq[i] = max(P_seq[i], L_seq[j] + 1) # add 1 to the length of the sequence
                max_len = max(max_len, P_seq[i]) # update the maximum length
    
    return max_len

# Dãy số WAVIO:
def Wavio_Numeric_Sequence(sequence):
    n = len(sequence)
    # Create two arrays L1 and L2 and fill them with 1s. The value L1[i] is going to store length of the longest increasing subsequence ending with sequence[i]. Similarly, L2[i] is going to store length of the longest decreasing subsequence starting from sequence[i].
    L1 = [1] * n
    L2 = [1] * n
 
    # Compute LIS values from left to right.
    for i in range(1, n):
        for j in range(i):
            if sequence[j] < sequence[i]:
                L1[i] = max(L1[i], L1[j]+1)
    
    # Compute LIS values from right to left.
    for i in range(n-2, -1, -1):
        for j in range(n-1, i, -1):
            if sequence[j] < sequence[i]:
                L2[i] = max(L2[i], L2[j]+1)
 
    max_len = 0
    # Find the maximum value of L1[i] + L2[i] - 1.
    for j in range(n):
        max_len = max(max_len, L1[j]+L2[j]-1)
 
    return max_len

# Tháp Babilon
def tower_of_babylon(blocks):
    """
    Finds the height of the tallest tower that can be built using the given blocks.

    :param blocks: A list of tuples (l, w, h) representing the length, width, and height of each block.
    :return: The height of the tallest tower that can be built.
    """
    # Sort the blocks in non-increasing order of base area
    sorted_blocks = sorted(blocks, key=lambda x: x[0] * x[1], reverse=True)

    # Initialize an array to store the maximum height that can be achieved using each block as the top block
    max_heights = [block[2] for block in sorted_blocks]

    # Iterate over all blocks in non-increasing order of base area
    for i in range(len(sorted_blocks)):
        for j in range(i):
            if sorted_blocks[j][0] > sorted_blocks[i][0] and sorted_blocks[j][1] > sorted_blocks[i][1]:
                # If the base of block j is larger than the base of block i, update the maximum height that can be
                # achieved using block i as the top block
                max_heights[i] = max(max_heights[i], max_heights[j] + sorted_blocks[i][2])

    # Return the maximum height that can be achieved using any block as the top block
    return max(max_heights)

# Xếp các khối đá :
def Stack_the_blocks(input_file, output_file1, output_file2):
    with open(input_file, 'r') as f:
        n = int(f.readline().strip())
        blocks = []
        for i in range(n):
            d, r, c = map(int, f.readline().strip().split())
            blocks.append((i+1, d, r, c))

    # Sort blocks in descending order of height
    blocks.sort(key=lambda x: min(x[1:]), reverse=True)

    # Initialize dynamic programming tables
    dp_count = [1] * n
    dp_height = [blocks[i][1] for i in range(n)]
    dp_prev = [-1] * n

    # Compute dynamic programming tables
    for i in range(n):
        for j in range(i):
            if blocks[j][1] > blocks[i][1] and blocks[j][2] > blocks[i][2]:
                if dp_count[j] + 1 > dp_count[i]:
                    dp_count[i] = dp_count[j] + 1
                    dp_height[i] = dp_height[j] + blocks[i][1]
                    dp_prev[i] = j
                elif dp_count[j] + 1 == dp_count[i] and dp_height[j] + blocks[i][1] > dp_height[i]:
                    dp_height[i] = dp_height[j] + blocks[i][1]
                    dp_prev[i] = j

    # Write output for maximum count
    with open(output_file1, 'w') as f:
        f.write(str(dp_count[-1]) + '\n')
        curr = n-1
        while curr != -1:
            t, d, r, c = blocks[curr]
            f.write(str(t) + ' ' + str(d) + ' ' + str(r) + ' ' + str(c) + '\n')
            curr = dp_prev[curr]

    # Write output for maximum height
    with open(output_file2, 'w') as f:
        f.write(str(dp_height[-1]) + '\n')
        curr = n-1
        while curr != -1:
            t, d, r, c = blocks[curr]
            f.write(str(t) + ' ' + str(d) + ' ' + str(r) + ' ' + str(c) + '\n')
            curr = dp_prev[curr]
            
## II) Vali (B)
def knapsack(n, W, a, b):
    """
    Solves the knapsack problem for n objects with weights a and values b,
    given a maximum weight W for the knapsack.
    Returns the maximum value that can be obtained.
    """
    # dp[i][j] represents the maximum value of the first i items when the total weight is j.
    dp = [[0 for j in range(W+1)] for i in range(n+1)]
    
    for i in range(1, n+1):
        for j in range(1, W+1):
            if a[i-1] > j:
                # If item i's weight is larger than the total weight, we cannot put item i into the knapsack. 
                # So the maximum value of the first i items is the same as the maximum value of the first i-1 items.
                dp[i][j] = dp[i-1][j]
            else:
                # If item i's weight is smaller than the total weight, we can put item i into the knapsack.
                # There are two cases: 
                # 1. We put item i into the knapsack. 
                #    In this case, the maximum value of the first i items is the value of item i plus the maximum value of the first i-1 items 
                #    when the total weight is the total weight minus item i's weight.
                # 2. We do not put item i into the knapsack.
                #    In this case, the maximum value of the first i items is the same as the maximum value of the first i-1 items.
                dp[i][j] = max(dp[i-1][j], b[i-1] + dp[i-1][j-a[i-1]])
    
    return dp[n][W]

# Dãy con có tổng bằng S:
def subset_sum(seq, target):
    n = len(seq)
    L = [[0] * (target + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        L[i][0] = 1
    for i in range(1, n + 1):
        for j in range(1, target + 1):
            if j < seq[i-1]:
                L[i][j] = L[i-1][j]
            else:
                L[i][j] = L[i-1][j] or L[i-1][j-seq[i-1]]
    return L[n][target]

# Chia kẹo
def divide_candy_packages(packages):
    total_candy = sum(packages)
    target_sum = total_candy // 2
    n = len(packages)
    
    # create a table to store if we can get sum t from first i candies
    table = [[False for t in range(target_sum + 1)] for i in range(n+1)]
    
    # set first column to True for base case
    for i in range(n+1):
        table[i][0] = True
    
    # fill the rest of the table
    for i in range(1, n+1):
        for t in range(1, target_sum+1):
            if t < packages[i-1]:
                table[i][t] = table[i-1][t]
            else:
                table[i][t] = table[i-1][t] or table[i-1][t-packages[i-1]]
    
    # find the largest sum we can get from the first i candies
    s = target_sum
    while not table[n][s]:
        s -= 1
    
    # find the two parts of the candy packages
    part1 = []
    part2 = []
    i = n
    while i > 0:
        if s - packages[i-1] >= 0 and table[i-1][s-packages[i-1]]:
            part1.append(packages[i-1])
            s -= packages[i-1]
        else:
            part2.append(packages[i-1])
        i -= 1
    
    return (part1, part2)

# Market (Olympic Balkan 2000)
def Market(a):
    n = len(a)
    T = sum(a)
    half_T = T // 2
    
    # initialize the L array
    L = [False] * (half_T + 1)
    L[0] = True
    
    # fill in the L array
    for i in range(n):
        for j in range(half_T, a[i]-1, -1):
            L[j] |= L[j-a[i]]
    
    # find the maximum weight that can be chosen
    for s in range(half_T, -1, -1):
        if L[s]:
            return s
    
    return 0

# Điền dấu 
def can_evaluate_to_S(nums, S):
    n = len(nums)
    T = sum(nums)
    L = [[0] * (2*T+1) for _ in range(n+1)]
    
    # Base case: L(1,a[1]) = 1
    L[1][nums[0]+T] = 1
    L[1][-nums[0]+T] = 1
    
    # Fill in the rest of the table using the recurrence relation
    for i in range(2, n+1):
        for t in range(-T, T+1):
            if L[i-1][t+nums[i-1]] or L[i-1][t-nums[i-1]]:
                L[i][t+T] = 1
    
    # Return the answer: can we evaluate to S?
    return bool(L[n][S+T])

def can_be_divisible_by_k(nums, k):
    n = len(nums)
    T = sum(nums)
    L = [0] * k
    
    # Base case: L(1,a[1]%k) = 1
    L[nums[0] % k] = 1
    
    # Fill in the rest of the table using the recurrence relation
    for i in range(2, n+1):
        new_L = [0] * k
        for j in range(k):
            if L[j]:
                new_L[(j+nums[i-1]) % k] = 1
                new_L[(j-nums[i-1]) % k] = 1
        L = new_L
    
    # Return the answer: can we get a result that's divisible by k?
    return bool(L[0])

def Fill_in(nums, target):
    # Determine the range of possible sums
    max_sum = sum(nums)
    min_sum = -max_sum

    # Initialize the L matrix
    L = [[False] * (max_sum - min_sum + 1) for _ in range(len(nums) + 1)]

    # Set the base case
    L[0][0 - min_sum] = True

    # Fill in the rest of the matrix using the recurrence relation
    for i in range(1, len(nums) + 1):
        for t in range(min_sum, max_sum + 1):
            if t - nums[i - 1] >= min_sum and L[i - 1][t - nums[i - 1] - min_sum]:
                L[i][t - min_sum] = True
            elif t + nums[i - 1] <= max_sum and L[i - 1][t + nums[i - 1] - min_sum]:
                L[i][t - min_sum] = True

    # Return True if there exists a solution, else False
    return L[-1][target - min_sum]

# Expression (ACM 10690)
def Expression(nums):
    n = len(nums)
    T = sum(nums)
    max_sum = T // 2
    L = [0] * (max_sum + 1)
    L[0] = 1
    for i in range(n):
        for j in range(max_sum, nums[i] - 1, -1):
            L[j] |= L[j - nums[i]]
    max_product = 0
    for S in range(max_sum, 0, -1):
        if L[S]:
            max_product = S * (T - S)
            break
    return max_product

## III) Biến đổi xâu:
def transform_x_to_f(X, F):
    n = len(X)
    m = len(F)
    
    # Initialize the memoization table
    dp = [[0 for j in range(m+1)] for i in range(n+1)]
    
    # Initialize the base cases
    for i in range(1, n+1):
        dp[i][0] = i
    for j in range(1, m+1):
        dp[0][j] = j
        
    # Fill in the memoization table
    for i in range(1, n+1):
        for j in range(1, m+1):
            if X[i-1] == F[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]) + 1
    
    # Backtrack to get the transformation steps
    i, j = n, m
    steps = []
    while i > 0 and j > 0:
        if X[i-1] == F[j-1]:
            i -= 1
            j -= 1
        elif dp[i][j] == dp[i-1][j-1] + 1:
            steps.append(('replace', i, F[j-1]))
            i -= 1
            j -= 1
        elif dp[i][j] == dp[i-1][j] + 1:
            steps.append(('delete', i))
            i -= 1
        else:
            steps.append(('insert', i+1, F[j-1]))
            j -= 1
    
    # Handle remaining characters in X or F
    while i > 0:
        steps.append(('delete', i))
        i -= 1
    while j > 0:
        steps.append(('insert', 1, F[j-1]))
        j -= 1
    
    # Reverse the steps and return the result
    steps.reverse()
    return steps

# Xâu con chung dài nhất
def longest_common_substring(X, Y):
    m = len(X)
    n = len(Y)
    # Initialize the table with 0 values
    L = [[0] * (n+1) for i in range(m+1)]
    # Fill the table with the dynamic programming algorithm
    for i in range(1, m+1):
        for j in range(1, n+1):
            if X[i-1] == Y[j-1]:
                L[i][j] = L[i-1][j-1] + 1
            else:
                L[i][j] = max(L[i-1][j], L[i][j-1])
    # Extract the longest common substring from the table
    longest_substring = ""
    i = m
    j = n
    while i > 0 and j > 0:
        if X[i-1] == Y[j-1]:
            longest_substring = X[i-1] + longest_substring
            i -= 1
            j -= 1
        elif L[i-1][j] > L[i][j-1]:
            i -= 1
        else:
            j -= 1
    return longest_substring

# Bắc cầu
def find_max_bridges(m, n, a, b):
    # create a table to store the length of the longest common subsequence of a and b
    table = [[0] * (n + 1) for _ in range(m + 1)]

    # fill the table using dynamic programming
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                table[i][j] = table[i - 1][j - 1] + 1
            else:
                table[i][j] = max(table[i][j - 1], table[i - 1][j])

    # backtrack to find the indices of the common subsequence
    i = m
    j = n
    indices = []
    while i > 0 and j > 0:
        if a[i - 1] == b[j - 1]:
            indices.append((i, j))
            i -= 1
            j -= 1
        elif table[i - 1][j] > table[i][j - 1]:
            i -= 1
        else:
            j -= 1

    # return the indices of the common subsequence in reverse order
    return indices[::-1]

# Palindrom (IOI 2000)
def min_chars_to_palindrome(S):
    n = len(S)
    P = S[::-1]
    L = [[0]*(n+1) for _ in range(n+1)]
    max_len = 0
    for i in range(1, n+1):
        for j in range(1, n+1):
            if S[i-1] == P[j-1]:
                L[i][j] = L[i-1][j-1] + 1
                if L[i][j] > max_len:
                    max_len = L[i][j]
            else:
                L[i][j] = max(L[i-1][j], L[i][j-1])
    return n - max_len

## IV) Vali (A)
def unbounded_knapsack(n, a, b, W):
    dp = [0] * (W + 1)
    for i in range(W + 1):
        for j in range(n):
            if a[j] <= i:
                dp[i] = max(dp[i], dp[i-a[j]] + b[j])
    return dp[W]

# Farmer (IOI 2004)
def Farmer(N, M, a, b, Q):
    # Create the items list
    items = []
    for i in range(N):
        items.append((a[i], a[i]))  # Land plot i has ai weight and ai value
    for j in range(M):
        items.append((b[j], b[j]-1))  # Strip of land j has bj weight and bj-1 value

    # Implement the knapsack algorithm
    dp = [0] * (Q+1)
    for i in range(len(items)):
        for j in range(items[i][0], Q+1):
            dp[j] = max(dp[j], dp[j-items[i][0]] + items[i][1])
    return dp[Q]


# Đổi tiền
def coin_change(N, M, coins):
    # initialize the dynamic programming table
    INF = float('inf')
    dp = [[INF] * (M+1) for _ in range(N+1)]
    for i in range(N+1):
        dp[i][0] = 0

    # fill the dynamic programming table
    for i in range(1, N+1):
        for j in range(1, M+1):
            if j < coins[i-1]:
                dp[i][j] = dp[i-1][j]
            else:
                dp[i][j] = min(dp[i-1][j], dp[i][j-coins[i-1]] + 1)

    # extract the solution
    if dp[N][M] == INF:
        return -1
    else:
        ans = []
        i, j = N, M
        while j > 0:
            if dp[i][j] == dp[i-1][j]:
                i -= 1
            else:
                ans.append(coins[i-1])
                j -= coins[i-1]
        return ans

## V) Nhân ma trận
def multiply_matrices(A, B):
    """
    Multiplies two matrices A and B and returns the resulting matrix.
    Assumes the matrices have compatible dimensions for multiplication.
    """
    # Get dimensions of matrices
    m = len(A)
    n = len(A[0])
    p = len(B[0])

    # Initialize result matrix with zeros
    result = [[0 for j in range(p)] for i in range(m)]

    # Multiply matrices
    for i in range(m):
        for j in range(p):
            for k in range(n):
                result[i][j] += A[i][k] * B[k][j]

    return result

# Chia đa giác
def Divide_polygon(dims):
    n = len(dims) - 1
    # Initialize the memoization table
    memo = [[0] * n for _ in range(n)]
    # Initialize the backtracking table
    backtrack = [[0] * n for _ in range(n)]

    # Compute the minimum number of multiplications for all subproblems
    for L in range(2, n+1):
        for i in range(n-L+1):
            j = i + L - 1
            memo[i][j] = float('inf')
            for k in range(i, j):
                cost = memo[i][k] + memo[k+1][j] + dims[i] * dims[k+1] * dims[j+1]
                if cost < memo[i][j]:
                    memo[i][j] = cost
                    backtrack[i][j] = k

    # Construct the optimal multiplication order
    def construct_order(i, j):
        if i == j:
            return 'A{}'.format(i+1)
        k = backtrack[i][j]
        return '({} {})'.format(construct_order(i, k), construct_order(k+1, j))

    return memo[0][n-1], construct_order(0, n-1)

# Biểu thức số học (IOI 1999)
def Arithmetic_expression(nums, ops):
    n = len(nums)
    dp_min = [[float('inf')] * n for _ in range(n)]
    dp_max = [[-float('inf')] * n for _ in range(n)]
    
    for i in range(n):
        dp_min[i][i] = nums[i]
        dp_max[i][i] = nums[i]
        
    for l in range(2, n+1):
        for i in range(n-l+1):
            j = i + l - 1
            for k in range(i, j):
                if ops[k] == '+':
                    dp_min[i][j] = min(dp_min[i][j], dp_min[i][k] + dp_min[k+1][j])
                    dp_max[i][j] = max(dp_max[i][j], dp_max[i][k] + dp_max[k+1][j])
                elif ops[k] == '*':
                    dp_min[i][j] = min(dp_min[i][j], dp_min[i][k] * dp_min[k+1][j])
                    dp_max[i][j] = max(dp_max[i][j], dp_max[i][k] * dp_max[k+1][j])
                    
    return dp_max[0][n-1]

## VI) Ghép cặp
def maximize_aesthetic(n, k, v):
    dp = [[0 for j in range(n+1)] for i in range(k+1)]
    for i in range(1, k+1):
        for j in range(i, n-k+i+1):
            dp[i][j] = dp[i][j-1]
            for p in range(i-1, j):
                dp[i][j] = max(dp[i][j], dp[i-1][p] + sum(v[q][j-1] for q in range(p, i-1, -1)))
    return dp[k][n]

# Câu lạc bộ:
def Club(n, k, a, b):
    # Initialize a 2D array to store the differences between the number of seats in each room and the number of students in each group
    v = [[0] * n for i in range(k)]
    for i in range(k):
        for j in range(n):
            v[i][j] = abs(a[j] - b[i])
    # Initialize a 2D array to store the minimum number of seat transfers
    f = [[float('inf')] * n for i in range(k)]
    for j in range(n):
        f[0][j] = v[0][j]
    # Calculate the minimum number of seat transfers for each group and room
    for i in range(1, k):
        for j in range(i, n):
            for m in range(i-1, j):
                f[i][j] = min(f[i][j], f[i-1][m] + v[i][j])
    # Return the minimum number of seat transfers required to arrange all groups in all rooms
    return f[k-1][n-1]

# Mua giày
def Buy_shoes(h, s):
    n = len(h)
    k = len(s)
    cost_matrix = np.zeros((k, n))
    for i in range(k):
        for j in range(n):
            cost_matrix[i, j] = abs(h[j] - s[i])
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    total_cost = cost_matrix[row_ind, col_ind].sum()
    return row_ind, col_ind, total_cost

## VII) Di chuyển
def move(A):
    M, N = len(A), len(A[0])
    dp = [[0 for j in range(N)] for i in range(M)]
    for j in range(N):
        dp[0][j] = A[0][j]
    for i in range(1, M):
        for j in range(N):
            if j == 0:
                dp[i][j] = A[i][j] + max(dp[i-1][j], dp[i-1][j+1])
            elif j == N-1:
                dp[i][j] = A[i][j] + max(dp[i-1][j], dp[i-1][j-1])
            else:
                dp[i][j] = A[i][j] + max(dp[i-1][j], dp[i-1][j-1], dp[i-1][j+1])
    return max(dp[-1])

# Tam giác (IOI 1994)
def Triangle(triangle):
    """
    Calculates the maximum sum of a path from the top of a triangle to the bottom.
    
    :param triangle: a list of lists representing the triangle
    :return: the maximum sum of a path from the top to the bottom of the triangle
    """
    n = len(triangle)
    # initialize the first row of F
    F = [triangle[0]]
    # calculate the values of F for the remaining rows
    for i in range(1, n):
        row = triangle[i]
        F.append([])
        for j in range(i+1):
            if j == 0:
                F[i].append(F[i-1][0] + row[0])
            elif j == i:
                F[i].append(F[i-1][i-1] + row[i])
            else:
                F[i].append(max(F[i-1][j-1], F[i-1][j]) + row[j])
    # return the maximum value in the last row of F
    return max(F[n-1])

# Con kiến
def Ant(food_grid):
    M = len(food_grid)
    N = len(food_grid[0])
    
    # add extra rows to handle wrapping around the cylinder
    food_grid = [[0] + row + [0] for row in food_grid]
    food_grid = [[food_grid[-1][j]]*(N+2)] + food_grid + [[food_grid[1][j]]*(N+2)]
    
    # initialize the memoization table
    memo = [[0]*(N+2) for _ in range(M+2)]
    
    # compute the maximum food the ant can collect at each position
    for j in range(1, N+1):
        for i in range(1, M+1):
            memo[i][j] = max(memo[i-1][j-1], memo[i][j-1], memo[i+1][j-1]) + food_grid[i][j]
    
    # find the maximum amount of food the ant can collect
    max_food = 0
    for i in range(1, M+1):
        max_food = max(max_food, memo[i][N])
    
    return max_food

def main():
    # Test def find_lis(seq)
    seq = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    print(find_lis(seq))
    
    # test def Meeting_room_arrangement(n, meetings)
    n = 5
    meetings = [(1, 3), (2, 4), (3, 5), (4, 6), (5, 7)]
    print(Meeting_room_arrangement(n, meetings))
    
    # test def Machine_rental(n, orders)
    n = 3
    orders = [(1, 3), (2, 4), (3, 5), (4, 6), (5, 7)]
    print(Machine_rental(n, orders))
    
    # test def many_triangles(triangles)
    triangles = [[(0, 0), (0, 1), (1, 0)], [(0, 0), (0, 1), (1, 1)], [(0, 0), (1, 0), (1, 1)], [(0, 1), (1, 0), (1, 1)]]
    print(many_triangles(triangles))
    
    # test def Sign_change_sequence(a, L, U)
    a = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    L = 3
    U = 5
    print(Sign_change_sequence(a, L, U))
    
    # test def Wavio_Numeric_Sequence(sequence)
    sequence = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    print(Wavio_Numeric_Sequence(sequence))

    # test def tower_of_babylon(blocks)
    blocks = [(1, 2, 3), (2, 3, 4), (3, 4, 5), (4, 5, 6), (5, 6, 7)]
    print(tower_of_babylon(blocks))
    
    # test def Stack_the_blocks(input_file, output_file1, output_file2)
    input_file = 'input.txt'
    output_file1 = 'output1.txt'
    output_file2 = 'output2.txt'
    Stack_the_blocks(input_file, output_file1, output_file2)

    # test def knapsack(n, W, a, b)
    n = 3
    W = 5
    a = [1, 2, 3]
    b = [2, 3, 4]
    print(knapsack(n, W, a, b))
    
    # test def subset_sum(seq, target)
    seq = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    target = 10
    print(subset_sum(seq, target))
    
    # test def divide_candy_packages(packages)
    packages = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    print(divide_candy_packages(packages))
    
    # test def Market(a)
    a = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    print(Market(a))
    
    # test def can_evaluate_to_S(nums, S)
    nums = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    S = 10
    print(can_evaluate_to_S(nums, S))
    
    # test def can_be_divisible_by_k(nums, k)
    nums = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    k = 10
    print(can_be_divisible_by_k(nums, k))
    
    # test def Fill_in(nums, target)
    nums = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    target = 10
    print(Fill_in(nums, target))
    
    # test def Expression(nums)
    nums = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    print(Expression(nums))
    
    # test def transform_x_to_f(X, F)
    X = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    F = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    print(transform_x_to_f(X, F))
    
    # def longest_common_substring(X, Y)
    X = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    Y = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    print(longest_common_substring(X, Y))

    # test def find_max_bridges(m, n, a, b)
    m = 3
    n = 3
    a = [1, 2, 3]
    b = [1, 2, 3]
    print(find_max_bridges(m, n, a, b))
    
    # test def min_chars_to_palindrome(S)
    S = 'abab'
    print(min_chars_to_palindrome(S))
    
    # test def unbounded_knapsack(n, a, b, W)
    n = 3
    a = [1, 2, 3]
    b = [2, 3, 4]
    W = 5
    print(unbounded_knapsack(n, a, b, W))
    
    # def Farmer(N, M, a, b, Q)
    N = 3
    M = 3
    a = [1, 2, 3]
    b = [1, 2, 3]
    Q = 3
    print(Farmer(N, M, a, b, Q))
    
    # test def coin_change(N, M, coins)
    N = 3
    M = 3
    coins = [1, 2, 3]
    print(coin_change(N, M, coins))
    
    # def multiply_matrices(A, B)
    A = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    B = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    print(multiply_matrices(A, B))
    
    # def Divide_polygon(dims)
    dims = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    print(Divide_polygon(dims))
    
    # def Arithmetic_expression(nums, ops)
    nums = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    ops = ['+', '-', '*', '/']
    print(Arithmetic_expression(nums, ops))
    
    # def maximize_aesthetic(n, k, v)
    n = 3
    k = 3
    v = [1, 2, 3]
    print(maximize_aesthetic(n, k, v))
    
    # def Club(n, k, a, b)
    n = 3
    k = 3
    a = [1, 2, 3]
    b = [1, 2, 3]
    print(Club(n, k, a, b))
    
    # def Buy_shoes(h, s)
    h = 3
    s = 3
    print("h is " + str(h))
    print("s is " + str(s))
    print(Buy_shoes(h, s))

    # def move(A)
    A = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    print("A: ", A)
    print("A[0]: ", A[0])
    print("A[1]: ", A[1])
    print("A[2]: ", A[2])
    print("A[3]: ", A[3])
    print("A[4]: ", A[4])
    print("A[5]: ", A[5])
    print("A[6]: ", A[6])
    print("A[7]: ", A[7])
    print("A[8]: ", A[8])
    print(move(A))

    # def Triangle(triangle):
    triangle = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    print("The triangle is: ", triangle)
    print("The perimeter is: ", Triangle(triangle))
    
    # def Ant(food_grid):
    food_grid = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    print(Ant(food_grid))
    print("End of program")


    
if __name__ == '__main__':
    main()

    
