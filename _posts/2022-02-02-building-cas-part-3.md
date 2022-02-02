---
layout: post
title: Building a Computer Algebra System (CAS) - Part (3) 
---

In this post, we will treat the following things

- update the **grad** operator for vector functions

With the code with developped in the previous post, we have a bug to fix, when taking the **grad** of the product between a scalar and a vector function:

```python
> W = ScalarFunctionSpace('W')
> V = VectorFunctionSpace('V')
> phi = ScalarFunction('phi', W)
> u = VectorFunction('u', V)

> Grad(u*phi)
u*Grad(phi) + Grad(u)*phi
```
which is absolutely wrong! 
Therefor, we have to implement the following rule; if $\phi$ is a scalar function and $u$ a vector function, then
>
  $$
  \newcommand{\grad}{\mathrm{grad}}
  \newcommand{\dot}{\mathrm{dot}}
  \begin{equation}
  \grad(\phi~u) = \dot(\grad(\phi), u) + \phi ~\grad(u) 
  \end{equation}
  $$



## Getting the code

You can find the code associated to this post [here](https://github.com/ratnania/ratnania.github.io/blob/gh-pages/codes/building-cas-using-sympy/step_3.py).
