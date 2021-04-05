
# Midterm project

The files part1.py, part2.py and part3.py correspond to the different tasks.

As we used symbolic differenciation in task3, eval_part3.py is used instead of part1.py to run with optimized parameters.

---

Most of the functions are decorated with @numba.njit for speedup.

By default we have disabled JIT, as it might fail on some systems.

To enable it, and get that sweet, sweet speedup, change the first line in *./src/.numba_config.yaml* to:

`disable_jit: 0`