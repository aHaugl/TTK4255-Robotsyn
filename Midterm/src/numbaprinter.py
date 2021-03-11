import sympy as sp
from sympy.codegen.ast import Variable
from sympy.printing.pycode import NumPyPrinter
import re
import autopep8


def numbafy(name, expression, variables=None,
            signature=None):
    signature = signature or 'float64[:,::1](float64[:])'
    cse = sp.cse(expression, optimizations='basic')
    np_printer = NumPyPrinter()
    variables = variables or expression.free_symbols
    arg1 = 'array'

    code = ''
    dec = f"@nb.njit('{signature}', cache=True, parallel=True)\n"
    code += dec
    code += f'def {name}(array):\n'
    var_string = ', '.join([str(var) for var in variables])
    code += f'\t"""\n\t{var_string}\n\t"""\n\n'
    for i, var in enumerate(variables):
        code += f'\t{var} = {arg1}[{i}]\n'
    code += '\n'
    for assignment in cse[0]:
        code += f'\t{np_printer.doprint(*assignment[::-1])}\n'

    code += '\n'

    retval = np_printer.doprint(cse[1][0]).replace('], [', '],\n[')
    retval = re.sub(r'(?<=[^\.\w])(\d+)(?=[^\.\w])', r'\g<1>.', retval)

    code += '\treturn '
    code += retval
    code = code.replace('numpy', 'np')
    return autopep8.fix_code(code)


def create_numba_file(functions):
    code = ''
    code += 'import numpy as np\n'
    code += 'import numba as nb\n'

    for func in functions:
        name, expression, variables = func[:3]
        signature = func[3] if len(func) > 3 else None

        code += '\n\n'
        code += numbafy(name, expression, variables, signature)

    with open('accelerated.py', 'w') as file:
        file.write(code)
# def numbafy(name, expression, variables=None):
#     np_printer = NumPyPrinter()
#     variables = variables or expression.free_symbols
#     arg1 = 'array'

#     code = ''
#     # code += '@numba.jit\n'
#     code += f'def {name}(array):\n'

#     for i, var in enumerate(variables):
#         code += f'\t{var} = {arg1}[{i}]\n'
#     code += '\n'

#     code += '\treturn (\n'
#     code += sp.pycode(expression).replace(',', ',\n')
#     code += '\n)'
#     code = code.replace('numpy', 'np')
#     code = code.replace('ImmutableDenseMatrix', 'np.array')
#     code = code.replace('math', 'np')

#     return autopep8.fix_code(code)
