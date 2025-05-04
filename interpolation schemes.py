
def linear_interpolation(x_values, y_values, step=0.01):
    # Create the new x-values array with the specified step
    x_min = min(x_values)
    x_max = max(x_values)
    new_x_values = np.arange(x_min, x_max + step, step)
    
    interpolated_y_values = []

    for x in new_x_values:
        # Find the interval [x_i, x_i+1] such that x_i <= x < x_i+1
        for i in range(len(x_values) - 1):
            if x_values[i] <= x < x_values[i + 1]:
                # Linear interpolation formula
                x_i = x_values[i]
                x_i1 = x_values[i + 1]
                y_i = y_values[i]
                y_i1 = y_values[i + 1]
                
                y = y_i + (y_i1 - y_i) * (x - x_i) / (x_i1 - x_i)
                interpolated_y_values.append(y)
                break
        else:
            # If x is exactly the last element of the x_values array
            if x == x_values[-1]:
                interpolated_y_values.append(y_values[-1])
    
    return new_x_values, interpolated_y_values


def natural_cubic_spline_interpolation(x_values, y_values, step=0.01):
    n = len(x_values) - 1
    h = np.diff(x_values)
    alpha = np.zeros(n)
    
    for i in range(1, n):
        alpha[i] = (3/h[i]) * (y_values[i + 1] - y_values[i]) - (3/h[i - 1]) * (y_values[i] - y_values[i - 1])
    
    l = np.ones(n + 1)
    mu = np.zeros(n)
    z = np.zeros(n + 1)
    
    for i in range(1, n):
        l[i] = 2 * (x_values[i + 1] - x_values[i - 1]) - h[i - 1] * mu[i - 1]
        mu[i] = h[i] / l[i]
        z[i] = (alpha[i] - h[i - 1] * z[i - 1]) / l[i]
    
    l[n] = 1
    z[n] = 0
    c = np.zeros(n + 1)
    b = np.zeros(n)
    d = np.zeros(n)
    
    for j in range(n - 1, -1, -1):
        c[j] = z[j] - mu[j] * c[j + 1]
        b[j] = (y_values[j + 1] - y_values[j]) / h[j] - h[j] * (c[j + 1] + 2 * c[j]) / 3
        d[j] = (c[j + 1] - c[j]) / (3 * h[j])
    
    new_x_values = np.arange(min(x_values), max(x_values) + step, step)
    interpolated_y_values = []
    
    for x in new_x_values:
        for i in range(n):
            if x_values[i] <= x <= x_values[i + 1]:
                delta_x = x - x_values[i]
                y = y_values[i] + b[i] * delta_x + c[i] * delta_x**2 + d[i] * delta_x**3
                interpolated_y_values.append(y)
                break
                
    #Add additional last value to make dimensions match
    #interpolated_y_values.append(interpolated_y_values[-1]) 
    
    return new_x_values, interpolated_y_values
