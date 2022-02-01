from operator  import mul, add
from functools import reduce

from sympy.core               import Basic
from sympy                    import Symbol
from sympy.core               import Expr
from sympy                    import Function
from sympy.core.singleton     import S
from sympy.core               import Add, Mul, Pow
from sympy                    import sympify
from sympy.tensor             import Indexed, IndexedBase

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
class BasicFunctionSpace(Basic):
    """
    Base class for Space of Functions
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
class ScalarFunctionSpace(BasicFunctionSpace):
    """
    An abstract Space of Scalar Functions
    """
    pass

#==============================================================================
class VectorFunctionSpace(BasicFunctionSpace):
    """
    An abstract Space of Vector Functions
    """
    pass

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
class IndexedVectorFunction(Indexed):
    """
    Represents a mathematical object with indices.
    """
    def __new__(cls, base, *args, **kw_args):

        if not args:
            raise IndexException("Indexed needs at least one index.")

        return Expr.__new__(cls, base, *args, **kw_args)

    def __hash__(self):
        return hash(self._args)

#==============================================================================
class VectorFunction(Symbol, IndexedBase):
    """
    Represents a vector function.
    """
    def __new__(cls, name, space):
        if not isinstance(space, VectorFunctionSpace):
            raise ValueError('Expecting a VectorFunctionSpace')
        obj        = Expr.__new__(cls)
        obj._space = space
        obj._name = name
        return obj

    @property
    def space(self):
        return self._space

    @property
    def name(self):
        return self._name

    def __getitem__(self, *args):

        if not(len(args) == 1):
            raise ValueError('expecting exactly one argument')

        return IndexedVectorFunction(self, *args)

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
class Dot(Function):
    """
    Represents a generic Dot operator, without knowledge of the dimension.
    """
    def __new__(cls, arg1, arg2, **options):
        # (Try to) sympify args first

        if options.pop('evaluate', True):
            r = cls.eval(sympify(arg1), sympify(arg2))
        else:
            r = None

        if r is None:
            return Basic.__new__(cls, arg1, arg2, **options)
        else:
            return r

    @classmethod
    def eval(cls, arg1, arg2):

        if isinstance(arg1, Add):
            args = [i for i in arg1.args]
            args = [cls(i, arg2) for i in args]
            return reduce(add, args)

        if isinstance(arg2, Add):
            args = [i for i in arg2.args]
            args = [cls(arg1, i) for i in args]
            return reduce(add, args)

        if isinstance(arg1, Mul):
            a = arg1.args
        else:
            a = [arg1]

        if isinstance(arg2, Mul):
            b = arg2.args
        else:
            b = [arg2]

        args_1 = [i for i in a if isinstance(i, VectorFunction)]
        c1     = [i for i in a if not i in args_1]
        args_2 = [i for i in b if isinstance(i, VectorFunction)]
        c2     = [i for i in b if not i in args_2]

        a = reduce(mul, args_1)
        b = reduce(mul, args_2)
        c = Mul(*c1)*Mul(*c2)

        if str(a) > str(b):
            a,b = b,a

        obj = Basic.__new__(cls, a, b)
        obj = Basic.__new__(cls, a, b)

        return c*obj

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
def test_vector_function_spaces():
    V = VectorFunctionSpace('V')
    u = VectorFunction('u', V)
    v = VectorFunction('v', V)

    alpha = Scalar('alpha')
    beta  = Scalar('beta')

    expr = u[0] * v[1]

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

#==============================================================================
def test_dot_1():
    V = VectorFunctionSpace('V')
    u1 = VectorFunction('u1', V)
    u2 = VectorFunction('u2', V)
    v1 = VectorFunction('v1', V)
    v2 = VectorFunction('v2', V)

    alpha = Scalar('alpha')
    a1 = Scalar('a1')
    a2 = Scalar('a2')
    b1 = Scalar('b1')
    b2 = Scalar('b2')

    #
    assert(Dot(u1+u2, v1) == Dot(u1,v1) + Dot(u2,v1))
    assert(Dot(u1, v1+v2) == Dot(u1,v1) + Dot(u1,v2))
    assert(Dot(alpha*u1,v1) == alpha*Dot(u1, v1))
    assert(Dot(u1,alpha*v1) == alpha*Dot(u1, v1))
    assert(Dot(a1*u1+a2*u2,b1*v1+b2*v2) == a1*b1*Dot(u1, v1) + a1*b2*Dot(u1, v2) + a2*b1*Dot(u2, v1) + a2*b2*Dot(u2, v2))

    assert(Dot(alpha*u1*u2,v1) == alpha*Dot(u1*u2, v1))

    assert(Dot(u1,v1) == Dot(v1,u1))

    expr = Dot(u1,v1)
    print(expr)

################################################################################
if __name__ == '__main__':
    test_scalar_function_spaces()
    test_vector_function_spaces()
    test_grad_1()
    test_dot_1()
