import numpy as np


def calc_integral_one_node(a, b, tk, n, alpha_func, K_func):
    aa = a
    res = 0.0
    for i in range(1, n+1):
        bb = min(alpha_func(i, tk), b)
        if aa>bb:
            continue
        integral = (bb-aa)*K_func(0.5*(aa+bb), i, tk)
        res += integral
        aa = bb
    return res


def solve_system_of_linear_volterra_1st_order(f_func, x_exact_func, alpha_func, K_func, n, T=1.0, N_MAX=100):
    t = np.arange(0, N_MAX+1)

    f = f_func(t)
    x_exact = x_exact_func(t)
    x = np.zeros(x_exact.shape)

    x[0] = x_exact_func(0.0)
    for k in range(1, t.size):
        temp_sum = 0
        for j in range(1, k):
            temp_sum += calc_integral_one_node(t[j - 1], t[j], t[k], n, alpha_func, K_func).dot(x[j])
        c = calc_integral_one_node(t[k - 1], t[k], t[k], n, alpha_func, K_func)
        x[k] = np.linalg.inv(c).dot(f[k] - temp_sum)
    err = np.abs(np.add(x_exact, -x))
    return x, err


if __name__ == "__main__":
    import example_svmo1 as ex

    print('Linear_Volterra_main')

    N, T = 500, 1.0
    xh, errh = solve_system_of_linear_volterra_1st_order(ex.f_func, ex.x_exact_func, ex.alpha_func, ex.K_func, ex.n, T, N)
    print(np.amax(errh, axis=0))
    print(xh)
