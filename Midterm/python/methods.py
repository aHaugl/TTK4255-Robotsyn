import numpy as np
# This is just a suggestion for how you might
# structure your implementation. Feel free to
# make changes e.g. taking in other arguments.


def fast_finite_jacobian(func, x, func_of_x, stepsize):
    # TODO discuss why not using equation from book
    def gradient(diff):
        return (func(x+diff) - func_of_x) / stepsize
    jac = np.apply_along_axis(
        gradient, axis=0, arr=np.eye(3)*stepsize)
    return jac


def cost(residuals):
    return np.sum(residuals**2)


def gauss_newton(residualsfun, p0, step_size=0.25,
                 num_iterations=100, finite_difference_epsilon=1e-5):
    # See the comment in part1.py regarding the 'residualsfun' argument.

    p = p0.copy()
    for iteration in range(num_iterations):
        # 1: Compute the Jacobian matrix J, using e.g.
        #    finite differences with the given epsilon.
        res_p = residualsfun(p)
        jac = fast_finite_jacobian(
            residualsfun, p, res_p, finite_difference_epsilon)
        # 2: Form the normal equation terms JTJ and JTr.
        norm_eq = jac.T@jac
        # 3: Solve for the step delta and update p as
        delta = np.linalg.solve(norm_eq, -jac.T@res_p)
        p += step_size*delta
    return p

# Implement Levenberg-Marquardt here. Feel free to
# modify the function to take additional arguments,
# e.g. the termination condition tolerance.


def levenberg_marquardt(residualsfun, p0, max_iterations=100,
                        finite_difference_epsilon=1e-5, cost_logger=None):
    p = p0.copy()
    mu = None
    for iteration in range(max_iterations):
        res_p = residualsfun(p)
        jac = fast_finite_jacobian(
            residualsfun, p, res_p, finite_difference_epsilon)
        norm_eq = jac.T@jac

        mu = (mu if mu is not None else 1e-6 * np.amax(jac**2))
        delta = np.linalg.solve(norm_eq + mu*np.eye(3), -jac.T@res_p)
        if cost(residualsfun(p + delta)) < cost(residualsfun(p)):
            p += delta
            mu /= 3
        else:
            mu *= 2

        stepsize = np.linalg.norm(delta)

        if cost_logger is not None:
            cost_logger.append([iteration, cost(res_p), stepsize])

        if stepsize <= 1e-3:
            break

    return p  # Placeholder, remove me!
