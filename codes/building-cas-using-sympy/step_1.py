from operator  import mul, add
from functools import reduce

from sympy.core               import Basic
from sympy                    import Symbol
from sympy.core               import Expr
from sympy                    import Function
from sympy.core.singleton     import S
from sympy.core               import Add, Mul, Pow
from sympy                    import sympify

#==============================================================================
class Scalar(Symbol):
    """
    Represents a scalar symbol.
    """
    _label = ''
    is_number = True
    def __new__(cls, *args, **kwargs):
        label = kwargs.pop('label', '')

        obj = Symbol.__new__(cls, *args, **kwargs)
        obj._label = label
        return obj

    @property
    def label(self):
        return self.label

#==============================================================================
class ScalarFunctionSpace(Basic):
    """
    An abstract Space of Scalar Functions
    """
    def __new__(cls, name):

        obj = Basic.__new__(cls)
        obj._name  = name

        return obj

    @property
    def name(self):
        return self._name

    def _sympystr(self, printer):
        sstr = printer.doprint
        return sstr(self.name)

    def __mul__(self, other):
        raise NotImplementedError('TODO')

    def __hash__(self):
        return hash((self.name, ))

#==============================================================================
class ScalarFunction(Symbol):
    """
    class descriving an element of a ScalarFunctionSpace.
    """
    def __new__(cls, name, space):
        if not isinstance(space, ScalarFunctionSpace):
            raise ValueError('Expecting a ScalarFunctionSpace')

        obj = Expr.__new__(cls)
        obj._name = name
        obj._space = space

        return obj

    @property
    def name(self):
        return self._name

    @property
    def space(self):
        return self._space

    def _sympystr(self, printer):
        sstr = printer.doprint
        return sstr(self.name)

    def __hash__(self):
        return hash((self.name, self.space))

#==============================================================================
class Grad(Function):
    """
    Represents a generic Grad operator, without knowledge of the dimension.
    This operator implements the properties of addition and multiplication
    """
    def __new__(cls, expr, **options):
        # (Try to) sympify args first

        if options.pop('evaluate', True):
            r = cls.eval(sympify(expr))
        else:
            r = None

        if r is None:
            return Basic.__new__(cls, expr, **options)
        else:
            return r

    @classmethod
    def eval(cls, expr):

        if isinstance(expr, Add):
            args = [i for i in expr.args]
            args = [cls(i) for i in args]
            return reduce(add, args)

        if isinstance(expr, Mul):
            left = expr.args[0]
            right = expr.args[1:]
            right = reduce(mul, right)

            d_left = cls(left, evaluate=True)
            d_right = cls(right, evaluate=True)

            return left * d_right + right * d_left

        if isinstance(expr, Pow):
            b = expr.base
            e = expr.exp
            a = cls(b)
            expr = expr.func(b, e-1)
            if isinstance(a, Add):
                expr = reduce(add, [e*expr*i for i in a.args])
            else:
                expr = e*a*expr
            return expr

        if not isinstance(expr, ScalarFunction):
            if expr.is_number:
                return S.Zero
            return cls(expr, evaluate=False)

#==============================================================================
def test_scalar_function_spaces():
    V = ScalarFunctionSpace('V')
    u = ScalarFunction('u', V)
    v = ScalarFunction('v', V)

    alpha = Scalar('alpha')
    beta  = Scalar('beta')

    # add two scalar functions
    expr = u + v
    # multiply two scalar functions
    expr = u*v
    # a linear expression
    expr = alpha * u + beta * v

#==============================================================================
def test_grad_1():
    V = ScalarFunctionSpace('V')
    u = ScalarFunction('u', V)
    v = ScalarFunction('v', V)

    alpha = Scalar('alpha')
    beta  = Scalar('beta')

    # Scalars
    assert(Grad(3) == 0)
    assert(Grad(alpha) == 0)

    # Distributive property
    assert(Grad(u+v) == Grad(u) + Grad(v))
    assert(Grad(u+3) == Grad(u))
    assert(Grad(u+alpha) == Grad(u))

    # Product rule for multiplication by a scalar
    assert(Grad(3*u) == 3*Grad(u))
    assert(Grad(alpha*u) == alpha*Grad(u))
    assert(Grad(u*v) == v*Grad(u) + u*Grad(v))
    assert(Grad(u/v) == -u*Grad(v)*v**(-2) + v**(-1)*Grad(u))

#    expr = Grad(u/v)
#    print(expr)

################################################################################
if __name__ == '__main__':
    test_scalar_function_spaces()
    test_grad_1()
