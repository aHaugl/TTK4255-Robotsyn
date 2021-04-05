import numpy as np
# This is just a suggestion for how you might
# structure your implementation. Feel free to
# make changes e.g. taking in other arguments.


def finite_jacobian(func, x, stepsize=1e-5):
    # TODO discuss why not using equation from book
    def gradient(diff):
        return (func(x+diff) - func(x-diff)) / (2*stepsize)
    jac = np.apply_along_axis(
        gradient, axis=0, arr=np.eye(x.shape[0])*stepsize)
    return jac


def squaresum(residuals):
    return np.sum(residuals**2)


def gauss_newton(func, func_diff, p0, step_size=0.25,
                 num_iterations=100):
    # See the comment in part1.py regarding the 'residualsfun' argument.

    p = p0.copy()
    for iteration in range(num_iterations):
        # 1: Compute the Jacobian matrix J, using e.g.
        #    finite differences with the given epsilon.
        res_p = func(p)
        jac = func_diff(p)
        # 2: Form the normal equation terms JTJ and JTr.
        norm_eq = jac.T@jac
        # 3: Solve for the step delta and update p as
        delta = np.linalg.solve(norm_eq, -jac.T@res_p)
        p += step_size*delta
    return p

# Implement Levenberg-Marquardt here. Feel free to
# modify the function to take additional arguments,
# e.g. the termination condition tolerance.


def levenberg_marquardt(func, func_diff, p0, max_iterations=100,
                        cost_logger=None, min_stepsize=1e-3):
    """
    see wikipedia for mu initialization
    """
    p = p0.copy()
    mu = None
    if cost_logger is not None:
        cost_logger.append([0, p.copy(), np.nan, np.nan])
    for iteration in range(1, max_iterations):
        res_p = func(p)
        jac = func_diff(p)

        norm_eq = jac.T@jac

        mu = (mu if mu is not None else 1e-3 * np.amax(jac**2))

        delta = np.linalg.solve(norm_eq + mu*np.eye(p0.shape[0]), -jac.T@res_p)
        if squaresum(func(p + delta)) < squaresum(res_p):
            p += delta
            mu /= 3
        else:
            mu *= 2

        stepsize = np.linalg.norm(delta)

        if cost_logger is not None:
            cost_logger.append([iteration, p.copy(), mu, stepsize])

        if stepsize <= min_stepsize:
            break

    return p  # Placeholder, remove me!
