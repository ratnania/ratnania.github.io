---
layout: post
title: Building a Computer Algebra System (CAS) - Part (1) 
---

In this post, we will learn how to implement the **Grad** operator using **Sympy**.

Let's first, state some of the properties we want to ensure:

If $$u$$ and $$v$$ are elements of a **FunctionSpace**, and $$\alpha$$ is a scalar, then:
 
> 
  $$
  \newcommand{\grad}{\mathrm{grad}}
  \begin{equation}
    \grad(u+v) = \grad(u) + \grad(v)
  \end{equation}
  $$
> 
  $$
  \newcommand{\grad}{\mathrm{grad}}
  \begin{equation}
    \grad(\alpha~u) = \alpha~\grad(u)
  \end{equation}
  $$

More generally, we would like to preserve every [calculus identity](https://en.wikipedia.org/wiki/Vector_calculus_identities#Gradient).

But wait a minute, there are two kinds of function spaces, one for scalar functions and the other one for vector functions. 

For the moment, we will restrict ourselves to the case of scalar functions.

In the sequel, every object that we will create, will be an extension of some existent object in sympy. For the moment, we will need the class **Basic** and **Symbol**

```python
from sympy.core import Basic
from sympy import Symbol
```

## Space of scalar functions

We shall use the following implementation for our **ScalarFunctionSpace**

```python
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

    def __hash__(self):
        return hash((self.name, ))
```

#### Some explainations :

- **_sympystr** TODO
- **__hash__** TODO

Next step, is to be able to create a **ScalarFunction** :

```python
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
```

#### Some explainations :
- In order to allow to write expressions with our scalar functions, we extended the **Symbol** class, and create the object using the **Expr** constructor.

Now we are able to things like these:
```python
V = ScalarFunctionSpace('V')
u = ScalarFunction('u', V)
v = ScalarFunction('v', V)

# add two scalar functions
expr = u + v

# multiply two scalar functions
expr = u*v
```

## Scalars

Finally, we will also add the notion of a scalar (in sympde, it is called **Constant**) to represent any scalar number:

```python
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
```

We can write an expression like this:

```python
V = ScalarFunctionSpace('V')
u = ScalarFunction('u', V)
v = ScalarFunction('v', V)

alpha = Scalar('alpha')
beta  = Scalar('beta')

# a linear expression
expr = alpha * u + beta * v
```

We are now ready to go!

## The Grad operator

The first thing is to find the appropriate class to extand from sympy. Here we shall be using the **Function** object.

```python
from sympy import Function 
```
Let us first take a look at how the class will be, before starting an appropriate implementation:
```python
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
        pass
```

The **__new__** method is special here; in fact, what we want from the **Grad** operator is to be able to perform an early evaluation of the given expression, when possible.

For this reason, we see that if **Grad** is called with **evaluate=False**, then no evaluation is performed. Actually, the default behavior here is to automatically evaluate the expression, by calling the **eval** method.

### Grad on scalars

The first rule we will implement is 

>
  $$
  \newcommand{\grad}{\mathrm{grad}}
  \begin{equation}
  \grad(\alpha) = 0, \quad \mbox{if} ~\alpha~ \mbox{is a scalar}
  \end{equation}
  $$

Implementing this rule is quite simple:
```python
    @classmethod
    def eval(cls, expr):
        ...
        if not isinstance(expr, ScalarFunction):
            if expr.is_number:
                return S.Zero
            return cls(expr, evaluate=False)
```

Since we are considering scalars here, normally, we must check if **expr** is of type **Scalar** then return **S.Zero** (a singleton describing the Zero object in sympy). However, since we want our rule to apply also on any **sympy** number, we use a different approach.

In fact, now we ensure that 
```python
assert(Grad(3) == 0)
assert(Grad(alpha) == 0)
```

### Distributive property

The second rule we will implement is 

>
  $$
  \newcommand{\grad}{\mathrm{grad}}
  \begin{equation}
    \grad(u+v) = \grad(u) + \grad(v)
  \end{equation}
  $$

Implementing this rule is the following:
```python
    @classmethod
    def eval(cls, expr):

        if isinstance(expr, Add):
            args = [i for i in expr.args]
            args = [cls(i) for i in args]
            return reduce(add, args)

        ...
```
This rule is placed at the begining of the **eval** method.

Now we ensure that 
```python
assert(Grad(u+3) == Grad(u))
assert(Grad(u+alpha) == Grad(u))
assert(Grad(u+v) == Grad(u) + Grad(v))
```

### Product rule for multiplication by a scalar 

The third rule we will implement is 

>
  $$
  \newcommand{\grad}{\mathrm{grad}}
  \begin{equation}
    \grad(uv) = u~\grad(v) + v~\grad(u)
  \end{equation}
  $$

Implementing this rule is quite simple:
```python
    @classmethod
    def eval(cls, expr):

        ...
        if isinstance(expr, Mul):
            left = expr.args[0]
            right = expr.args[1:]
            right = reduce(mul, right)

            d_left = cls(left, evaluate=True)
            d_right = cls(right, evaluate=True)

            return left * d_right + right * d_left

        ...
```
This rule is placed after the **Add** rule in the **eval** method.

Now we ensure that 
```python
assert(Grad(3*u) == 3*Grad(u))
assert(Grad(alpha*u) == alpha*Grad(u))
assert(Grad(u*v) == v*Grad(u) + u*Grad(v))
```

