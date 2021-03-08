import sympy as sp
from sympy.printing.pycode import NumPyPrinter
import autopep8


def numbafy(name, expression):
    cse = sp.cse(expression, optimizations='basic')
    np_printer = NumPyPrinter()
    variables = expression.free_symbols
    arg1 = 'array'

    code = ''
    # code += '@numba.jit\n'
    code += f'def {name}(array):\n'

    for i, var in enumerate(variables):
        code += f'\t{var} = {arg1}[{i}]\n'
    code += '\n'
    for assignment in cse[0]:
        code += f'\t{np_printer.doprint(*assignment[::-1])}\n'

    code += '\n'

    code += '\treturn ('
    code += ',\n'.join([np_printer.doprint(retval).replace(',', ',\n')
                        for retval in cse[1]])
    code += ')'
    code = code.replace('numpy', 'np')
    return autopep8.fix_code(code)
