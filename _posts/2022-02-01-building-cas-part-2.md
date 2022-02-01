---
layout: post
title: Building a Computer Algebra System (CAS) - Part (2) 
---

In this post, we will add the following things

- Vector functions
- the differential operators **curl** and **div**
- the algebraic operation **dot**

Finally, we will see what are the additional calculus identities that we will be able to cover.

## Space of vector functions

Since we will have now a new space of functions, it is worth making both of them extend a basic class **BasicFunctionSpace**: 

```python
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
```

Therefor, scalar and vector function spaces are defined as:


```python
class ScalarFunctionSpace(BasicFunctionSpace):
    """
    An abstract Space of Scalar Functions
    """
    pass

class VectorFunctionSpace(BasicFunctionSpace):
    """
    An abstract Space of Vector Functions
    """
    pass
```

In opposition to **ScalarFunction**, the case of **VectorFunction** must be treated carefully. In fact, we must be able to get a component of a vector function.

In sympy, in order to have an object that can be indexed, we use **IndexedBase**. Here's an example :  

```python
from sympy.tensor import Indexed, IndexedBase

A = IndexedBase('A') 
``` 

Now, if you print the type of **A[0]** you'll find that it is an **Indexed**:

```python
> type(A[0])
sympy.tensor.indexed.Indexed
```

This is exactly what we need to have in our CAS system. This means we need to:
- implement an object describing vector functions, that can be indexed
- if **a** is a vector function, then we should be able to access its components, example: **a[0]**

These objects will be
- **VectorFunction**
- **IndexedVectorFunction**

Next is the implementation of **VectorFunction**:

```python
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
```

A first attempt for **IndexedVectorFunction** is given here

```python
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
```

Now it is possible to do something like this:
```python
V = VectorFunctionSpace('V')
u = VectorFunction('u', V)
v = VectorFunction('v', V)

expr = u[0] * v[1]
```

The next thing to do now, is to add the algebraic operator **Dot** for the scalar product.

## The Algebraic Operator Dot

While the **grad** is an unary operator, the **dot** is a binary one.

The next rule we are interested in, is the linearity of the **Dot** operator.

If $$u, u_1, u_2, v, v_1, v_2$$ are vector functions and $$\alpha$$ is a scalar, then:
 
> 
  $$
  \newcommand{\dot}{\mathrm{dot}}
  \begin{equation}
    \dot(u_1+u_2, v) = \dot(u_1, v) + \dot(u_2, v) \\
    \dot(u, v_1+v_2) = \dot(u, v_1) + \dot(u, v_2) 
  \end{equation}
  $$
> 
  $$
  \newcommand{\dot}{\mathrm{dot}}
  \begin{equation}
    \dot(\alpha~u, v) = \alpha~\dot(u,v) \\
    \dot(u, \alpha~v) = \alpha~\dot(u,v)
  \end{equation}
  $$

We already encountered these kind of rules in the case of the **grad** operator. However in this case, we need to deal with a bianry operator. But it is the same idea! 

The implementation of **Dot** is given below
```python
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

        obj = Basic.__new__(cls, a, b)

        return c*obj
```
With this, we ensure the following things
```python
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
```
This is pretty cool! The problem now, is that our **dot** operator is not **commutative**! Here is what we get right now
```python
> assert(Dot(u1,v1) == Dot(v1,u1))
AssertionError
```
A simple trick to fix this, is to add the following statement, just before the creation of **obj**
```python
    @classmethod
    def eval(cls, arg1, arg2):
        ...
        if str(a) > str(b):
            a,b = b,a

        obj = Basic.__new__(cls, a, b)
```
This will ensure that a given order is always preserved. You can test it.
