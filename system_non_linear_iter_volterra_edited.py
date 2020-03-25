import numpy as np
from scipy import integrate
import example_svmo3_snlv21 as ex


def int_1node(a, b, tk, n, j, x_mas, f_alpha, f_K, f_G, with_s=False):
    aa = a
    res = 0.0
    for i in range(1, n + 1):
        bb = min(f_alpha(i, tk), b)
        if aa > bb or aa == bb:
            continue
        s = 0.5 * (aa + bb)
        x_s = xh_apprx(x_mas, j, a, b, s)
        if with_s:
            var_s = a
            integral = (bb - aa) * (s - var_s) * f_K(s, i, tk).dot(f_G(s, i, x_s))
        else:
            integral = (bb - aa) * f_K(s, i, tk).dot(f_G(s, i, x_s))
        res += integral
        aa = bb
    return res


def int_rp(a, b, tk, n, j, x_0, x_m, f_alpha, f_K, f_G, f_dG):
    aa = a
    res = 0.0
    for i in range(1, n+1):
        bb = min(f_alpha(i, tk), b)
        if aa > bb or aa == bb:
            continue
        s = 0.5 * (aa + bb)
        x0_s = xh_apprx(x_0, j, a, b, s)
        xm_s = xh_apprx(x_m, j, a, b, s)
        # f_dG_xm = f_dG(s, i, x0_s).dot(xm_s)
        # integral = (bb - aa) * f_K(s, i, tk).dot(f_dG_xm - f_G(s, i, xm_s))
        f_dG_xm = f_dG(s, i, x0_s)*(xm_s)
        integral = (bb - aa) * f_K(s, i, tk).dot(f_dG_xm - f_G(s, i, xm_s))
        res += integral
        aa = bb
    return res


def xh_apprx(xh, j, a, b, s=None):
    if s is None:
        s = 0.5 * (a + b)
    if approximation == PIECEWISE_CONST:
        return xh[j]
    elif approximation == PIECEWISE_LINEAR:
        return xh[j-1] + ((xh[j] - xh[j - 1]) / (b - a)) * (s - a)
    else:
        print("Error get_piece_of_x", xh, j, approximation)
        return 0


def xk_apprx(t, k, n, x_0, x, f_alpha, f_K, f_dG, fk):
    if approximation == PIECEWISE_CONST:
        left_part_sum = 0   # left part of linearized equation
        for j in range(1, k):
            left_part_sum += int_1node(t[j - 1], t[j], t[k], n, j, x_0, f_alpha, f_K, f_dG)*x[j]
        int_tk = int_1node(t[k - 1], t[k], t[k], n, k, x_0, f_alpha, f_K, f_dG)
        if int_tk.size == 1:
            res = (fk - left_part_sum) / int_tk
        else:
            res = np.linalg.inv(int_tk).dot(fk - left_part_sum)
    elif approximation == PIECEWISE_LINEAR:
        sum_1k = 0
        for j in range(1, k):
            int_tj = int_1node(t[j - 1], t[j], t[k], n, j, x_0, f_alpha, f_K, f_dG)
            int_s_tj = int_1node(t[j - 1], t[j], t[k], n, j, x_0, f_alpha, f_K, f_dG, True)
            sum_1k += int_tj.dot(x[j - 1]) + int_s_tj.dot((x[j] - x[j - 1]) / (t[j] - t[j - 1]))
        int_s_tk = int_1node(t[k - 1], t[k], t[k], n, k, x_0, f_alpha, f_K, f_dG, True)
        I = (1.0 / (t[k] - t[k - 1])) * int_s_tk
        if int_s_tk.size == 1:
            res = x[k - 1] + ((fk - x[k - 1] * int_1node(t[k - 1], t[k], t[k], n, k, x_0, f_alpha, f_K, f_dG) - sum_1k) / I)
        else:
            res = x[k - 1] + np.linalg.inv(I).dot(fk - int_1node(t[k - 1], t[k], t[k], n, k, x_0, f_alpha, f_K, f_dG).dot(x[k - 1]) - sum_1k)
        if I == 0.0:
            print("Error KusLin, X1[", k, "], I==0")
    else:
        print("Error get_piece_of_xk", x, k, approximation)
        res = 0
    return res


def solve_non_linear_volterra_1st_order(f_func, x_exact_func, f_alpha, f_K, f_G, f_dG, n, M_ITER=2, T=2.0, N_MAX=32):
    t = np.linspace(0, T, N_MAX+1)    # mesh nodes

    x_exact = x_exact_func(t)   # exact solution of x(s)
    x_0 = 0.6 * np.ones(x_exact.shape)     # x_0 - first approximation for x(s)
    x_m = np.copy(x_0)      # current approximation for x(s) on m-iteration
    x = np.zeros(x_exact.shape)     # next approximation for x(s) on m-iteration, we have calculate it
    x[0] = x_exact_func(0.0)    # we don't calculate initial value for t_0=0

    for i in range(1, M_ITER+1):
        f = f_func(t)   # f[k] contains right part of linearized equation: f(t)+sum(integrals)
        for k in range(1, t.size):
            for j in range(1, k+1):
                f[k] += int_rp(t[j - 1], t[j], t[k], n, j, x_0, x_m, f_alpha, f_K, f_G, f_dG)
            x[k] = xk_apprx(t, k, n, x_0, x, f_alpha, f_K, f_dG, f[k])
        x_m = np.copy(x)
        print("Погрешность итерации", i, "равна", np.amax(np.abs(np.add(x_exact, -x))))

    err = np.abs(np.add(x_exact, -x))
    return x_m, err


PIECEWISE_CONST = 0
PIECEWISE_LINEAR = 1

if __name__=="__main__":
    print('Non_linear_Volterra_main\n')
    # approximation = PIECEWISE_CONST
    approximation = PIECEWISE_LINEAR

    x_sol, err_sol = solve_non_linear_volterra_1st_order(ex.f_func, ex.x_exact_func, ex.alpha_func, ex.K_func, ex.G_func,
                                                     ex.dG_func, ex.n, M_ITER=10, N_MAX=64)
    print("Погрешность равна", np.amax(np.abs(err_sol)))
