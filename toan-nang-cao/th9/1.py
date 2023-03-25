def is_prime(n):
    if n <= 1:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True

def approximate_pi(n):
    pi = 0
    sign = 1
    for i in range(1, n+1, 2):
        pi += sign * 4/i
        sign *= -1
    return pi

def integrate(f, a, b, n):
    dx = (b - a) / n
    x = a
    integral = 0
    for i in range(n):
        integral += f(x) * dx
        x += dx
    return integral
