### A Pluto.jl notebook ###
# v0.14.3

using Markdown
using InteractiveUtils

# ╔═╡ 00aee08e-a674-11eb-0cc7-c17b0e738c81
begin
	ENV["MPLBACKEND"]="Agg"
	
	using Plots
	using PlutoUI
	
	using OrdinaryDiffEq
	using LinearAlgebra
	using ForwardDiff

	using BenchmarkTools
	using CPUTime
	
	gr()  # Using the Plotly Backend
	
	using DifferentialEquations
end

# ╔═╡ 3ac23f84-6a1e-456f-96cd-d131b1363000
md"""Why Julia:
* open source
* developed by **computer scientists** and **mathematicians** **simultaneously**
* wants to close gap between **scripting languages** (high productivity) and **scientific computing** languages like FORTRAN (high speed)
* engagement, e.g., by Intel
* **innovative** language features
"""

# ╔═╡ cf2164d6-fac7-41be-8422-191524b2782f
md"""**Challenges addressed by Julia**:
* *Readability* of mathematical software
* *Efficiency* (do not lose any performance, **nowhere**!)
* *Sustainability* in the sense of software engineering (big programs)
* *Reproducibility*
* *Parallelization* (towards automatic parallelization & vectorization)
* *Real world simulations* (setup, parameter variation, …)"""

# ╔═╡ 5ee03d5f-6f59-4fc3-923a-117ee25191bd
md"""Some appetizers in the beginning, Julia enables to solve a
couple of problems which can typically solved by **MATHEMATICA**:

**Example 1**, solve the **Lotka-Volterra ODE**:
$\begin{align*}
 \dot{x} & = a x - b x y \\
\dot{y} & = -c y + d x y
\end{align*}$
"""

# ╔═╡ b58b60cd-5db6-4b17-a7cb-b1b1e59b0fcf
begin
	# Define a problem:
	p = (1.0, 2.0, 1.5, 1.25) # a, b, c, d

	f = function(du, u, p, t) # Define f as an in-place update into du
    	a, b, c, d = p
		
    	du[1] = a * u[1] - b * u[1] * u[2]
    	du[2] = -c * u[2] + d * u[1] * u[2]
	end

	u0 = [1.0; 1.0]
	tspan = (0.0, 10.0)
	prob = ODEProblem(f, u0, tspan, p)
end

# ╔═╡ b132f218-4e88-4822-baf7-35ed8e923a6a
# Solve the problem:
sol = solve(prob);

# ╔═╡ 36edbe4a-362d-4469-bfb9-6eb853acf27a
# Make some plots:
plot(sol, title = "functions")

# ╔═╡ 7bfe7361-27bc-47b7-8809-ed1e3b7bbe42
# Further plots:
plot(sol, title  ="Phase Diagram", vars=(1,2))

# ╔═╡ 826f10d8-fdde-47b7-ab23-67cda4963245
md"""**Example 2**, the Kepler problem:
$\begin{align*}
   H & = \frac{1}{2} \left (p_1^2 + p_2^2 \right ) - \frac{1}{\sqrt{q_1^2 + q_2^2}} \\
 \frac{\mathrm{d} p}{\mathrm{dt}} & = -\frac{\mathrm{d} H}{dq} \\
 \frac{\mathrm{d} q}{\mathrm{dt}} & = \frac{\mathrm{d} H}{dp}.
\end{align*}$ """

# ╔═╡ 13690565-6e6f-4fea-8cdc-b314c90149f7
begin
	H(q,p) = norm(p)^2/2 - inv(norm(q))
	L(q,p) = q[1]*p[2] - p[1]*q[2]

	pdot(dp, p, q, params, t) = ForwardDiff.gradient!(dp, q->-H(q, p), q)
	qdot(dq, p, q, params, t) = ForwardDiff.gradient!(dq, p-> H(q, p), p)

	initial_position = [0.4, 0]
	initial_velocity = [0.0, 2.0]
	initial_cond = (initial_position, initial_velocity)
	
	initial_first_integrals = (H(initial_cond...), L(initial_cond...))
	
	
	tspan_kepler = (0, 20.0)
	prob_kepler = DynamicalODEProblem(pdot, qdot, initial_velocity, initial_position, tspan_kepler)
	sol_kepler = solve(prob_kepler, KahanLi6(), dt=1//10)
end

# ╔═╡ fe31f078-f602-4973-a4c2-18118e23659c
begin
	# Define plot functions:
	plot_orbit(sol) = plot(sol, vars=(3,4), lab="Orbit", title="Kepler Problem Solution")

function plot_first_integrals(sol, H, L)
	plot(initial_first_integrals[1].-map(u->H(u[2,:], u[1,:]), sol.u), lab="Energy variation", title="First Integrals")
    	plot!(initial_first_integrals[2].-map(u->L(u[2,:], u[1,:]), sol.u), lab="Angular momentum variation")
	end
	
	analysis_plot(sol, H, L) = plot(plot_orbit(sol), plot_first_integrals(sol, H, L))
end

# ╔═╡ 0055fb9d-20e4-492d-95f6-e9e9ca161fa0
analysis_plot(sol_kepler, H, L)

# ╔═╡ ddf94698-8761-48f1-af37-80bfdece505f
md"""**Example 3**, rational matrices:"""

# ╔═╡ 790fbea2-6b28-4d71-a86f-979f2b195253
m = [3 // 2  3//4; 5 // 6  7 // 8]

# ╔═╡ 8635bea0-53a1-4981-b0fe-a387d0ee93a2
# invert the matrix
mi = inv(m)

# ╔═╡ aed7c6ea-3447-44a1-985b-a74b440690eb
md"""check $m * m_i = I$:"""

# ╔═╡ 930affd2-ecdd-4d28-9807-827b132c31e3
m * mi

# ╔═╡ 87d26378-7d08-42b6-a214-be7d4ca6fbd0
# Some experiments with Julia, a function definition:
f1(n) = 5n - 1

# ╔═╡ 29fc1ae0-e14e-4b52-975b-99dddb1ca277
# And a more complex function:
function g(n)   # g(n) = f1(f1(f1(f1(f1(f1(f1(f1(f1(f1(n))))))))))
  for i=1:10
    n = f1(n)
  end
  return n
end

# ╔═╡ 91731e6e-249f-474a-8fc2-3506e958a1d8
# We can use different data types, e.g., Int64:
typeof(1)

# ╔═╡ c4ae2170-db32-4dc8-a02f-9da2a8df0bea
# f1 for Int64(1) yields:
f1(1)

# ╔═╡ e18f3b51-355d-43b6-b898-0dd2d07b4d3c
# g for Int64(1) yields:
g(1)

# ╔═╡ b8867a7c-a475-4b9c-9f22-c2090cd59af4
# Another type, Float64:
typeof(1.0)

# ╔═╡ abc7e1bf-a07d-4e46-9b66-bf81d9d3de53
md"""But there are also **high precision float numbers**:"""

# ╔═╡ dc1646ea-5873-461c-b532-2ad8a1d61267
BigFloat(π)

# ╔═╡ 23357ed7-9673-44b5-9bd8-f423364aec6d
# g for BigFloat(π) yields:
g(BigFloat(π))

# ╔═╡ 49c6b16b-f93e-4975-bac5-e28400cb5642
md"""Another interesting type is BigInt for **very big integers**.
Example: $n! = \prod_{i=1}^n i$"""

# ╔═╡ 0fcdfb3d-442b-49bb-bb05-16c9fe97bafe
# n! grows very fast:
factorial(4)

# ╔═╡ c5c042a0-383d-4a6b-bbe4-725d4e9f5f5f
# 20! is fine for Int64:
factorial(20)

# ╔═╡ 3f89fc28-5c49-4dce-9b8d-aad23d1a1382
begin
	# But 21! is not:
	21 * factorial(20)
end

# ╔═╡ c004221e-0ad4-4e03-b3bb-45c4534ff776
# Here is the reason:
typemax(Int64)

# ╔═╡ 72bf6902-285d-476d-b991-b9abd052fad4
# However:
factorial(BigInt(21))

# ╔═╡ 3ade8c02-45ea-463d-b08d-8e3d2136c153
# And even:
factorial(BigInt(100))

# ╔═╡ 91bcc05d-da7e-4268-a15f-21c7dd9918c5
md"""Is Julia just another script language like Python, ...? **No!**

Behind Julia, there is a powerful **Just-In-Time (JIT)** compiler, based on
**LLVM (low level virtual machine)**:"""

# ╔═╡ 1f0e070c-8244-40de-88b0-bc402fac1417
with_terminal() do
	@code_llvm g(5)
end

# ╔═╡ 1220beed-75cb-44a3-97ca-3d70231c40bb
md"""One can even get the compiled code of the routine:"""

# ╔═╡ f7642343-5a7c-4789-9c69-d8fbabe6241d
with_terminal() do
	@code_native g(5)
end

# ╔═╡ bf7387a8-da56-4b74-91f1-25d6b5815239
md"""Actually, the compiled code tells us:
$\begin{align*}
   g(n) & = 9765625 n - 2441406
\end{align*}$"""

# ╔═╡ 2aa105ec-00a3-4ae2-8afc-1060d9e1a1f9
md"""Why is this the case?
$\begin{align*}
   g(n) & = 𝑓^{(10)}(n) \\
        & = f(f(f(f(f(f(f(f(f(f(n)))))))))) \\
        & = 5^{10} n - \sum_{i=0}^{9} 5^i \\
         & = 5^{10} n - \frac{5^{10} - 1}{5-1} \\
         & =  9765625 n - 2441406
\end{align*}$
Thus, the code is **highly optimized** by **LLVM**!"""

# ╔═╡ 7911ae8c-9f7c-400e-84c5-9d9efda20742
md"""The performance can be measured with a **Julia macro**:"""

# ╔═╡ e3988925-78b2-4b4f-8841-faac90905285
@benchmark g(5)

# ╔═╡ 4fc27d39-f274-4cd8-86f2-1fb71a817532
md"""What does happen for calling g with a **Float64** argument?"""

# ╔═╡ 9a478fe3-9400-448b-8593-2e2c9adc88d9
with_terminal() do
	@code_llvm g(5.0)
end

# ╔═╡ c214e680-904d-4d0e-97e7-45caef769dfb
md"""Thus, the code is not so much optimized, the **loop is unrolled**!"""

# ╔═╡ 92da1d01-280a-401f-a313-23ad6913bacb
md"""What does happen for calling g with a **BigFloat** argument?"""

# ╔═╡ 09b168d8-65de-4e2b-8ebb-ce6a6d971729
with_terminal() do
	@code_llvm g(BigFloat(5.0))
end

# ╔═╡ 3cf1953a-1efe-465a-8214-bec780971a25
md"""Here, the code is completely different and **not very well readable**."""

# ╔═╡ 8086163e-e193-4dfe-b8a4-beb0031c3471
md"""Also the time measurements are different:"""

# ╔═╡ 74260633-5671-402f-b816-d841b32cbfe0
@benchmark g(5.0)

# ╔═╡ 25454ac0-9648-4c7d-8600-dd2075c879ae
md"""Also the time measurements are different:"""

# ╔═╡ 774e38c8-df48-4b72-bfa5-840824a2c16e
@benchmark g(BigFloat(5))

# ╔═╡ dafb671c-d34e-4ec6-a259-9853f111e1e3
md"""Behind all this is a basic concept of Julia called **multiple dispatch**.

A function called with **different types** has different representations
in the computer - with **different performance**."""

# ╔═╡ d4499be5-21bd-4518-a645-1f479546d202
md"""However, we can **speed up** the function **g** by adding a **specialized method**:"""

# ╔═╡ 0d2a4dd6-1209-4486-bb83-6a4b91381b34
# function g(n::Union{Float64, BigFloat})
#	return 9765625n - 2441406
# end

# ╔═╡ 08a104b0-1f0a-4951-b71f-4f9e4747de98
md"""The **runtime** is now better:"""

# ╔═╡ c87a0597-d349-43c4-ba55-0e272058b5c5
@benchmark g(5.0)

# ╔═╡ bb873b6d-4f07-4ee7-9c2b-d5e61b69addc
md"""And for **g(BigFloat)**:"""

# ╔═╡ 3b5c401c-1490-41ce-9ad0-7504ef2ce8e3
@benchmark g(BigFloat(5))

# ╔═╡ fbe296df-cd13-4dbd-8920-989a3bfbe6a9
md"""Now we have **two methods** for **one function (g)**:
methods(g)
This is again **multiple dispatch**, a kind of **generic programming**."""

# ╔═╡ c76ccc04-1d49-4963-9246-99623346dd0c
md"""The compiled code reads now:"""

# ╔═╡ 4b5b9079-a937-4156-b978-3fdd13329b0f
with_terminal() do
	@code_llvm g(5.0)
end

# ╔═╡ 496c3dec-11e2-4653-ab95-b85fad21cebf
md"""There is another question: why was Julia able to optimize g(5), but not g(5.0)?
The answer is **reproducibility**.
Julia only makes (**automatically**) optimizations that do not **change the numerical results**. But **addition of Float64 numbers** is **not associative**!"""

# ╔═╡ d94298ba-8f67-4899-840f-a868ae44cf94
@show (1.0 + 0.5 * eps(1.0)) - 1.0

# ╔═╡ 4a354f49-0531-45f3-8b45-262b430f954a
@show 1.0 + (0.5 * eps(1.0) - 1.0)

# ╔═╡ 87e42e8b-5da0-46de-a071-212b07ba2fdf
md"""eps(x) is a very interesting function for **reproducibility** of algorithms. It gives us the next Float64 number **next to x**:"""

# ╔═╡ f42eef0c-2a66-4257-b12f-d186abf413ba
@show eps(0.0)

# ╔═╡ 71e1b5b1-b2bf-4d17-bfe4-5bd40a70d92f
@show eps(1.0)

# ╔═╡ 3f9ed07c-fbdb-4e78-b3d4-5b328aa904dd
@show eps(10.0)

# ╔═╡ de52691b-249d-46ab-9e35-76aefba74202
md"""This can be used for **sensible stopping criteria** in numerical algorithms, e.g., for the Newton method to compute
the non-negative solution of $x^2 - c = $ via the iteration:
$\begin{align*}
   x_{n+1} & = \frac{1}{2} x_n + \frac{c}{2 x_n}
\end{align*}$"""

# ╔═╡ 68d80321-4714-4504-a6f8-324b09d53b51
function my_sqrt(c::T) where T <: Real
  if(c < 0)
    error("Argument of my_sqrt is negative!")
  end

  its::Int = 0

  my_float_type = promote_type(Float64, T)

  x_old::my_float_type = c
  x_new::my_float_type = c / 2

  while abs(x_new - x_old) >= eps(x_old)
    x_old, its = x_new, its + 1
    x_new = x_old / 2 + c / 2x_old
  end

  return x_new, its
end

# ╔═╡ a01e9fa9-87e0-46ac-a1d8-07473958f1a1
md"""execute the function:"""

# ╔═╡ fc3ac93b-1b7b-4724-9a2f-ba0030265588
my_sqrt(2)

# ╔═╡ ba278362-aa82-44bd-9085-75881480e122
md"""Many Julia functions similar to Matlab (dense/sparse, symmetric, diagonal, …) matrices:"""

# ╔═╡ 876abb6e-a2a6-4fbf-924f-750ad9dea7fb
a = rand(10, 10)

# ╔═╡ 215cd6de-9096-4480-8352-60d8b8ff0983
md"""Compute the inverse of a matrix:"""

# ╔═╡ b3e6cd23-0f81-469b-861c-fab86942335a
inv(a)

# ╔═╡ 450bf245-fd80-4dc7-812a-7cdfd4a6f12c
eigvals(a)

# ╔═╡ 2b4e40db-1250-4054-8dcb-a95244bb5266
md"""There are **three different ways** to **parallelize** your code. Parallelization
 is important, because **arithmetic operations are fast**, but **memomry access is usually slow**:
 * **SIMD** (single instruction, multiple data; vectorization)
 * **shared memory** parallelization via threads (experimental)
 * **distributed memory** parallelization via message passing (exploratory)"""

# ╔═╡ c3eca11f-e452-4351-b6f5-3fecdea2f228
md"""**SIMD** parallelization, resp., **vectorization**:"""

# ╔═╡ 7e46f851-5db5-44e7-90e9-1709e06d462c
function normal_sum(a::Vector)
    total = zero(eltype(a))
    for x in a
        total += x^2
    end
    return total
end

# ╔═╡ c6a0ff1d-15be-4149-a202-1069f7012bf6
md"""Here the **vectorized** version:"""

# ╔═╡ 22f8baf4-832e-4a6c-89b4-6373ddfb1d1c
function simd_sum(a::Vector)
    total = zero(eltype(a))
    @simd for x in a
        total += x^2
    end
    return total
end

# ╔═╡ f4fdbc85-554c-4dda-8c04-7193f1fd995b
md"""Compare the performance with a random vector:"""

# ╔═╡ 68345345-ed11-4d16-b3fe-c6b138e10cd1
b = rand(10_000)

# ╔═╡ abc31df9-f9f1-43de-a63f-3626f7810cc9
md"""Without *SMD*:"""

# ╔═╡ 6eb308a8-e307-47e1-8b54-7b0e7750cf40
@benchmark normal_sum(b)

# ╔═╡ 66514cd5-bf25-499c-b27e-40748a6ee70b
md"""With *SMD*:"""

# ╔═╡ 388b37d4-d2ee-442c-91fd-f6705c3bee84
@benchmark simd_sum(b)

# ╔═╡ a67cdd42-791e-4f06-8ec9-bfb8bb9ebc71
md"""A last remark, how to realize **structured programming** in Julia?
Use **modules**:
In a file "HelloWorld.jl" in the current path you could find the code:"""

# ╔═╡ ab7bc993-76c2-47cb-a5d5-ec75c380f6e0
module HelloWorld
  export main

  function main()
    local_function()
  end

  function local_function()
    @show("Hello, world!")
  end
end

# ╔═╡ fa7bb4e5-62ed-41f8-ba87-e40b90d9a041
md"""The content of the file can be made available by:"""

# ╔═╡ 37950ac1-cc37-4417-a9de-712edc6ce9f0
begin
	# import HelloWorld
	HelloWorld.main()
	# local_function() # will not work, scope of local_function() is hidden
end

# ╔═╡ Cell order:
# ╠═00aee08e-a674-11eb-0cc7-c17b0e738c81
# ╟─3ac23f84-6a1e-456f-96cd-d131b1363000
# ╟─cf2164d6-fac7-41be-8422-191524b2782f
# ╟─5ee03d5f-6f59-4fc3-923a-117ee25191bd
# ╠═b58b60cd-5db6-4b17-a7cb-b1b1e59b0fcf
# ╠═b132f218-4e88-4822-baf7-35ed8e923a6a
# ╠═36edbe4a-362d-4469-bfb9-6eb853acf27a
# ╠═7bfe7361-27bc-47b7-8809-ed1e3b7bbe42
# ╟─826f10d8-fdde-47b7-ab23-67cda4963245
# ╠═13690565-6e6f-4fea-8cdc-b314c90149f7
# ╠═fe31f078-f602-4973-a4c2-18118e23659c
# ╠═0055fb9d-20e4-492d-95f6-e9e9ca161fa0
# ╟─ddf94698-8761-48f1-af37-80bfdece505f
# ╠═790fbea2-6b28-4d71-a86f-979f2b195253
# ╠═8635bea0-53a1-4981-b0fe-a387d0ee93a2
# ╟─aed7c6ea-3447-44a1-985b-a74b440690eb
# ╠═930affd2-ecdd-4d28-9807-827b132c31e3
# ╠═87d26378-7d08-42b6-a214-be7d4ca6fbd0
# ╠═29fc1ae0-e14e-4b52-975b-99dddb1ca277
# ╠═91731e6e-249f-474a-8fc2-3506e958a1d8
# ╠═c4ae2170-db32-4dc8-a02f-9da2a8df0bea
# ╠═e18f3b51-355d-43b6-b898-0dd2d07b4d3c
# ╠═b8867a7c-a475-4b9c-9f22-c2090cd59af4
# ╟─abc7e1bf-a07d-4e46-9b66-bf81d9d3de53
# ╠═dc1646ea-5873-461c-b532-2ad8a1d61267
# ╠═23357ed7-9673-44b5-9bd8-f423364aec6d
# ╟─49c6b16b-f93e-4975-bac5-e28400cb5642
# ╠═0fcdfb3d-442b-49bb-bb05-16c9fe97bafe
# ╠═c5c042a0-383d-4a6b-bbe4-725d4e9f5f5f
# ╠═3f89fc28-5c49-4dce-9b8d-aad23d1a1382
# ╠═c004221e-0ad4-4e03-b3bb-45c4534ff776
# ╠═72bf6902-285d-476d-b991-b9abd052fad4
# ╠═3ade8c02-45ea-463d-b08d-8e3d2136c153
# ╟─91bcc05d-da7e-4268-a15f-21c7dd9918c5
# ╠═1f0e070c-8244-40de-88b0-bc402fac1417
# ╟─1220beed-75cb-44a3-97ca-3d70231c40bb
# ╠═f7642343-5a7c-4789-9c69-d8fbabe6241d
# ╟─bf7387a8-da56-4b74-91f1-25d6b5815239
# ╟─2aa105ec-00a3-4ae2-8afc-1060d9e1a1f9
# ╟─7911ae8c-9f7c-400e-84c5-9d9efda20742
# ╠═e3988925-78b2-4b4f-8841-faac90905285
# ╟─4fc27d39-f274-4cd8-86f2-1fb71a817532
# ╠═9a478fe3-9400-448b-8593-2e2c9adc88d9
# ╟─c214e680-904d-4d0e-97e7-45caef769dfb
# ╟─92da1d01-280a-401f-a313-23ad6913bacb
# ╠═09b168d8-65de-4e2b-8ebb-ce6a6d971729
# ╟─3cf1953a-1efe-465a-8214-bec780971a25
# ╟─8086163e-e193-4dfe-b8a4-beb0031c3471
# ╠═74260633-5671-402f-b816-d841b32cbfe0
# ╟─25454ac0-9648-4c7d-8600-dd2075c879ae
# ╠═774e38c8-df48-4b72-bfa5-840824a2c16e
# ╟─dafb671c-d34e-4ec6-a259-9853f111e1e3
# ╟─d4499be5-21bd-4518-a645-1f479546d202
# ╠═0d2a4dd6-1209-4486-bb83-6a4b91381b34
# ╟─08a104b0-1f0a-4951-b71f-4f9e4747de98
# ╠═c87a0597-d349-43c4-ba55-0e272058b5c5
# ╟─bb873b6d-4f07-4ee7-9c2b-d5e61b69addc
# ╠═3b5c401c-1490-41ce-9ad0-7504ef2ce8e3
# ╟─fbe296df-cd13-4dbd-8920-989a3bfbe6a9
# ╟─c76ccc04-1d49-4963-9246-99623346dd0c
# ╠═4b5b9079-a937-4156-b978-3fdd13329b0f
# ╟─496c3dec-11e2-4653-ab95-b85fad21cebf
# ╠═d94298ba-8f67-4899-840f-a868ae44cf94
# ╠═4a354f49-0531-45f3-8b45-262b430f954a
# ╟─87e42e8b-5da0-46de-a071-212b07ba2fdf
# ╠═f42eef0c-2a66-4257-b12f-d186abf413ba
# ╠═71e1b5b1-b2bf-4d17-bfe4-5bd40a70d92f
# ╠═3f9ed07c-fbdb-4e78-b3d4-5b328aa904dd
# ╟─de52691b-249d-46ab-9e35-76aefba74202
# ╠═68d80321-4714-4504-a6f8-324b09d53b51
# ╟─a01e9fa9-87e0-46ac-a1d8-07473958f1a1
# ╠═fc3ac93b-1b7b-4724-9a2f-ba0030265588
# ╟─ba278362-aa82-44bd-9085-75881480e122
# ╠═876abb6e-a2a6-4fbf-924f-750ad9dea7fb
# ╟─215cd6de-9096-4480-8352-60d8b8ff0983
# ╠═b3e6cd23-0f81-469b-861c-fab86942335a
# ╠═450bf245-fd80-4dc7-812a-7cdfd4a6f12c
# ╟─2b4e40db-1250-4054-8dcb-a95244bb5266
# ╟─c3eca11f-e452-4351-b6f5-3fecdea2f228
# ╠═7e46f851-5db5-44e7-90e9-1709e06d462c
# ╟─c6a0ff1d-15be-4149-a202-1069f7012bf6
# ╠═22f8baf4-832e-4a6c-89b4-6373ddfb1d1c
# ╟─f4fdbc85-554c-4dda-8c04-7193f1fd995b
# ╠═68345345-ed11-4d16-b3fe-c6b138e10cd1
# ╟─abc31df9-f9f1-43de-a63f-3626f7810cc9
# ╠═6eb308a8-e307-47e1-8b54-7b0e7750cf40
# ╟─66514cd5-bf25-499c-b27e-40748a6ee70b
# ╠═388b37d4-d2ee-442c-91fd-f6705c3bee84
# ╟─a67cdd42-791e-4f06-8ec9-bfb8bb9ebc71
# ╠═ab7bc993-76c2-47cb-a5d5-ec75c380f6e0
# ╟─fa7bb4e5-62ed-41f8-ba87-e40b90d9a041
# ╠═37950ac1-cc37-4417-a9de-712edc6ce9f0
