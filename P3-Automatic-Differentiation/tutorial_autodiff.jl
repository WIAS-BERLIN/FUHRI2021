### A Pluto.jl notebook ###
# v0.14.2

using Markdown
using InteractiveUtils

# ╔═╡ 5cb08a46-a0f0-11eb-3aaa-03f9763dcb75
begin
	using ForwardDiff
end

# ╔═╡ d4c4e2f4-ba74-400d-aee1-d5ac2c63ee19
md"""

# Automatic Differentation with Julia

This tutorial gives some first insight into the automatic differentation features of Julia. To run it properly you need to have installed the following packages:

- [ForwardDiff](https://github.com/JuliaDiff/ForwardDiff.jl)
- [DiffResults](https://github.com/JuliaDiff/DiffResults.jl)


**Remark**: 
Here, we will focus on the forward differentiation technique based on dual numbers. There is also a ReverseDiff package that is based on iteratively reversed application of the cain rule. It might be the better alternative for high-dimensional gradients or functions with more input parameters than the ouput dimension.

"""

# ╔═╡ b19235be-0338-4212-9d0a-5e45a1c866ea
md"""

### Dual Numbers

Forward differentiation is based on dual numbers. Dual numbers look like complex numbers as there are written as

$ z = a + b \epsilon \quad \text{where} \quad \epsilon^2 = 0 $

and have a 'real part' $a$ and a 'dual part' $b$ beeing the coefficient for the 'dual unit' $\epsilon$.

You can add and multiply them commutatively by

$ (a + b \epsilon) + (x + y \epsilon) = (a + x) + \epsilon (b + y) $

$ (a + b \epsilon) \cdot (x + y \epsilon) = (ax + 0) + \epsilon (ay + bx) $

"""

# ╔═╡ 688958d3-c39d-488c-9950-2c5fc83dee40
md"""

### So what? (Look at Taylor series)

Consider a function $f : \mathbb{R} \rightarrow \mathbb{R}$ and its Taylor series expansion at some value $a$:

$ f(a+h) = f(a) + \sum_{n=1}^\infty \frac{f^{(n)}(a)}{n!} h^n $

Now, for $h = \epsilon$, due to $\epsilon^2 = 0$ one immediately obtains

$ f(a+h) = f(a) + f^{(1)}(a) \epsilon $

In other words: Evaluating the function in the dual number $(a + 1\epsilon)$ gives the dual number $(f(a) + f^{(1)}(a) \epsilon)$ as a result.

**Key point**: If the function $f$ can be also evaluated in dual numbers, one can obtain its derivatives 'automatically' on basis of the dual number operations above!
Due to Julia's on demand compilation and type dispatching, there will be two compiled versions of f(x<:Real), one for e.g. x::Float64 and one for the corressponding dual numbers (which is still a subtype of Real).

Details on the actual implementation of the Dual numbers can be found in the [Documentation of ForwardDiff](https://juliadiff.org/ForwardDiff.jl/stable/dev/how_it_works/)

"""

# ╔═╡ 2029b8bd-408c-4e2f-a628-ee2a56eed43a
md"""

#### Example 1 : 1D differentiation
Let's define some scalar-valued function $f : \mathbb{R} \rightarrow \mathbb{R}$ to test this:

"""

# ╔═╡ 605fa750-99c8-4ea4-928b-2e13110de9a8
function f(z::Real)
	return abs(z)
end

# ╔═╡ 9153b452-f073-402d-ac68-a257cb8e7740
md"""

The derivative of our function f can be generated by the following line.

"""

# ╔═╡ 624a73e1-3ee0-4e7e-a7ec-8d9146d2ac29
df(z) = ForwardDiff.derivative(f,z)

# ╔═╡ 53a4910c-5e18-4d66-bc34-5712bd0934a9
md"""

Let's evaluate f and df at some points.

"""

# ╔═╡ d0bdbfaf-9a8a-408c-aace-5ed53052ab4f
z = 1e-16

# ╔═╡ e1bbb076-1ec4-4005-a69c-667e7fd0a35b
"f(z) = $(f(z)) | df(z) = $(df(z))"

# ╔═╡ 271511b3-c5e5-4ca2-ab79-5372b9f9a616
md"""

#### Exercise Block 1

**Exercise 1.1** : Compute the derivative of the function $f(z) = (z+1)\exp(-z^2)$ at z = 1. (You just have to manipulate the function f and z above.)

**Exercise 1.2** : Try discontinuous functions like f(z) = abs(z) to see what happens there.

**Exercise 1.3** : Write a Newton method to solve $f(z) = 0$ where you compute the gradient of $f$ by ForwardDiff. Test your implementtion with $f(z) = z^2 - 2$. (You can reveal the Solution by pressing on the eye symbol on the left side of the next hidden box.)

"""

# ╔═╡ 2a54d947-04e7-4b77-8c82-d2ebddadff4a
begin
function newton(f::Function; init = 1, maxits = 100, tol = 1e-12)
	zn::Float64 = init
	fzn::Float64 = 0
	dfzn::Float64 = 0
	it::Int = 0
	while (true)
		it += 1
		fzn = f(zn)
		if abs(fzn) < tol
			return "arrived at [z,f(z)] = [$zn, $fzn] after $it iterations"
		elseif it >= maxits
			return "arrived at [z,f(z)] = [$zn, $fzn] after $it iterations (=maxits)"
		else
			dfzn = ForwardDiff.derivative(f,zn)
			if abs(dfzn) < tol
				return "zero derivative at z = $z in iteration $it (try different init)"
			end
			zn = zn - fzn/dfzn
		end
	end
end
newton(f; init = z)
end

# ╔═╡ 1c7324e7-73d6-41f3-be47-fd794756b842
md"""

#### Example 2 : Derivatives of vector-valued functions

Let's have a look at some two-dimensionally parametrized vector-field $F : \mathbb{R}^2 \rightarrow \mathbb{R}^3$

"""

# ╔═╡ b86165a5-894a-4f51-ad45-2155ac6bf041
function F(a::Vector{<:Real})
	return [cos(a[1]), sin(a[2]), a[1]^2+a[2]]
end

# ╔═╡ 525771d7-0e1c-4841-adcc-7faf01d72172
md"""

A partial derivative of F with respect to the parameter $y$ could be obtained by

"""

# ╔═╡ 5bc6a8bb-ab37-4bec-82fc-5546cdd2325d
 DF(a) = ForwardDiff.jacobian(F,a)

# ╔═╡ b3f45169-7dc0-4d6f-8d81-22fbe9a91e7e
md"""

Let's evaluate F and dFdy at some points.

"""

# ╔═╡ be36c18c-9e06-41a6-b76d-22fb2ee359f9
a = [0.0, 0.0]

# ╔═╡ 62a2e422-3ed3-44bf-9d67-11bc899a4b38
"F(a) = $(F(a)) | DF(a) = $(DF(a))"

# ╔═╡ aa74e4e8-8e9f-4c6d-bb2f-4dbf73f88a7e
md"""

Note: $DF(a) \in \mathbb{R}^{3 \times 2}$, the entry
at position $[j,k]$ is the partial derivative of $\partial F_j / \partial a_k$.

"""

# ╔═╡ Cell order:
# ╟─d4c4e2f4-ba74-400d-aee1-d5ac2c63ee19
# ╟─5cb08a46-a0f0-11eb-3aaa-03f9763dcb75
# ╟─b19235be-0338-4212-9d0a-5e45a1c866ea
# ╟─688958d3-c39d-488c-9950-2c5fc83dee40
# ╟─2029b8bd-408c-4e2f-a628-ee2a56eed43a
# ╠═605fa750-99c8-4ea4-928b-2e13110de9a8
# ╟─9153b452-f073-402d-ac68-a257cb8e7740
# ╠═624a73e1-3ee0-4e7e-a7ec-8d9146d2ac29
# ╟─53a4910c-5e18-4d66-bc34-5712bd0934a9
# ╠═d0bdbfaf-9a8a-408c-aace-5ed53052ab4f
# ╟─e1bbb076-1ec4-4005-a69c-667e7fd0a35b
# ╟─271511b3-c5e5-4ca2-ab79-5372b9f9a616
# ╟─2a54d947-04e7-4b77-8c82-d2ebddadff4a
# ╟─1c7324e7-73d6-41f3-be47-fd794756b842
# ╠═b86165a5-894a-4f51-ad45-2155ac6bf041
# ╟─525771d7-0e1c-4841-adcc-7faf01d72172
# ╠═5bc6a8bb-ab37-4bec-82fc-5546cdd2325d
# ╟─b3f45169-7dc0-4d6f-8d81-22fbe9a91e7e
# ╠═be36c18c-9e06-41a6-b76d-22fb2ee359f9
# ╟─62a2e422-3ed3-44bf-9d67-11bc899a4b38
# ╟─aa74e4e8-8e9f-4c6d-bb2f-4dbf73f88a7e
