
def newton_raphson(func, derivative, initial_guess, *args, tolerance=1e-15, max_iterations=1000):

    x_n = initial_guess
    for iteration in range(max_iterations):
        f_x_n = func(x_n, *args)
        f_prime_x_n = derivative(x_n, *args)

        if f_prime_x_n == 0:
            raise ValueError("Derivative is zero. No solution.")

        x_n1 = x_n - f_x_n / f_prime_x_n

        # convergence
        if abs(x_n1 - x_n) < tolerance:
            return x_n1, iteration + 1

        x_n = x_n1

    raise ValueError("Maximum iterations without solution.")
    