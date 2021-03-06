### A Pluto.jl notebook ###
# v0.14.3

using Markdown
using InteractiveUtils

# ╔═╡ fdb7be1f-94ab-4c3f-a5fb-05d8433d8e16
begin
	using Pkg
	Pkg.activate(mktempdir())
	Pkg.add(["OrdinaryDiffEq", 
			 "DataDrivenDiffEq", 
			 "ModelingToolkit", 
			 "DiffEqFlux", 
			 "Flux", 
			 "Plots", 
			 "DiffEqSensitivity", 
			 "Optim",
			 "GalacticOptim",
			 "PlutoUI"]) # it is a good idea to test if CUDA is available

	using Plots
	using Markdown
	
	using OrdinaryDiffEq
	using Statistics
	using DiffEqFlux
	using Flux
	using DiffEqSensitivity
	using ModelingToolkit
	using DataDrivenDiffEq
	using Optim
	using GalacticOptim
	using LinearAlgebra
	using PlutoUI
	gr()   # use Plotly graphics backend
end

# ╔═╡ 631a008c-a667-11eb-3af0-e99874435c33
md"""# Machine learning and PDEs with Julia
In this tutorial, we will discuss examples from the paper ["Universal Differential Equations for Scientific Machine Learning"](https://arxiv.org/abs/2001.04385) by Christopher Rackauckas, Yingbo Ma, Julius Martensen, Collin Warner, Kirill Zubov, Rohit Supekar, Dominic Skinner, Ali Ramadhan, Alan Edelman.


Run the following code to install all required packages (this will take a while).
"""

# ╔═╡ 56747210-79d6-4862-a5b2-1a707d713c93
md"""
## 0. Brief background on neural networks

In very simplified terms, a (deep) neural network attempts to 
approximate an unkown function $F:\mathbb{R}^M\to \mathbb{R}^N$.
Typically such an NN has the form 

$\begin{equation}
F_\theta(X) = W_K\sigma_{K-1}\big(W_{K-1}\sigma_{K-2}(\ldots W_1\sigma_0(W_0X+b_0)+b_1\ldots) + b_{K-1}\big) + b_K,
\end{equation}$

where $\theta=\{(W_k,b_k)\}_{k=0}^K$ are the parameters consisting of weight matrices
$W_k$ and biases $b_k$, $\sigma_k$ are the (nonlinear) activation functions (e.g. sigmoid, ReLU). Each part of this composition is called a layer.

In supervised learning the parameters $\theta$ are computed
by minimizing a cost function that depends on some given data.
This process is called training and the data is called training set.


## 1. Lotka-Volterra ODE example
As a first example, we will study the wellknown Lotka-Volterra system given in the form 


$\begin{align*}
\dot u_1 &= +\alpha \,u_1 -\beta\, u_1 \cdot u_2\\
\dot u_2 &=   + \gamma\, u_1\cdot u_2 - \delta\, u_2
\end{align*}$


Subject to initial conditions $u_1(0) = u_1^0$ and $u_2(0)=u_2^0$. We can solve the system easily in Julia with the following code.
"""

# ╔═╡ c7475b08-ebbd-4d33-9320-24f67b72b9c5
function lotka_volterra!(du, u, p, t)
    α, β, γ, δ = p
    du[1] = α*u[1] - β*u[2]*u[1]
    du[2] = γ*u[1]*u[2]  - δ*u[2]
end

# ╔═╡ 8bd1dec7-a592-4980-8f2f-f35cdb829096
md"""This defines the right-hand side in the equation above, `du` is a placeholder for the time derivative, `p=(α,β,γ,δ)` denotes the parameters and `t` is the time (not used here).

Next, we specify the time interval on which we want to solve the ODE, set the initial conditions $u(0)= u_0$, and fix the parameters for the problem."""

# ╔═╡ e352104e-d543-4b01-a905-06f14728fed6
begin
	tspan = (0.0f0, 3.0f0)              # Time interval (0, 3)
	u0 = Float32[0.44249296,4.6280594]  # initial condition
	p_ = Float32[1.3, 0.9, 0.8, 1.8]    # parameters α, β, γ, δ
	nothing                             # suppress output
end

# ╔═╡ a4ab0f0a-8cfd-458a-807d-f59f982cbcd5
md"""To define the full ODE problem, we use `ODEProblem` from the `DifferentialEquations.jl` package (in fact we only use the package `OrdinaryDifferentialeq.jl` since we are "only" dealing with ODEs). The constructor for the problem takes the nonlinearity, the intial condition, the time interval, and the parameter array as arguments."""

# ╔═╡ 59461d05-df8f-487d-98dc-46e524a49603
prob = ODEProblem(lotka_volterra!, u0, tspan, p_);

# ╔═╡ 696e727c-33ef-4f07-b5f1-5e90330eef57
md"""Finally, we solve the Lotka-Volterra system using Verner's "Most Efficient" 7/6 Runge-Kutta method. We save the solution at timesteps $\Delta t = 0.1$. The pairs $\{(t_i,u(t_i))\}_{i=1}^N$ will be used as "measurements" to train our network. However, to make things more interesting we will add some noise to $\{(t_i,u(t_i))\}_{i=1}^N$. Note that we put `;` at the end of the command to suppress the output."""

# ╔═╡ 5602c03b-ee48-4073-990d-2be18b873da7
solution = solve(prob, Vern7(), saveat = 0.1);

# ╔═╡ c29da13f-9d85-484e-9617-9f9dc732d0c3
md"""The solve command returns a complicated object. For easier use later on we convert it to a simple array."""

# ╔═╡ 7fa06933-f018-4b48-bdac-e2bd21b37ced
begin
	U = Array(solution)
	t = solution.t
	nothing # suppresses any output
end

# ╔═╡ ab36c055-a97a-4204-81f8-26a5152b56e5
md"""
### 1.1 Creating noisy measurements
"""

# ╔═╡ 7efe7664-302b-4cbf-a276-098fdf7ef4b4
md"""Next, we will add some noise to the solution. We use the mean values of $u_1(t_i)$ and $u_2(t_i)$ for $i=1,\ldots,N$ to construct noise with a magnitude proportional to the solution."""

# ╔═╡ d4c5df84-3ed4-41df-b651-f5e4605bc5de
begin
	noise_factor = Float32(5e-2);
	Uₘ = mean(U, dims=2) # vector of means
end

# ╔═╡ d1ae256f-091b-4459-b759-825b18549128
md"""The following vector defines the noisy data $(t_i,(\tilde u_1^i, \tilde u_2^i))$ that will be used to train the network. (The notation `.+` and `.*`, etc. means that the operations are componentwise.)"""

# ╔═╡ 1d291e76-3d2f-4cae-9c2f-5009a4d58e53
Ũ = U .+ (noise_factor*Uₘ) .* rand(eltype(U), size(U)); # Gaussian

# ╔═╡ 63667f95-0431-4cd6-874d-701ae57bbb1d
md"""Let us plot the solution of the "true" system together with the noisy data."""

# ╔═╡ 0ede66e5-4c24-488c-85f0-9d4b69cabe68
begin
	plot(solution, alpha = 0.75, color = :black, label = ["True Data" nothing], lw = 2)
	scatter!(t,transpose(Ũ), color= :red, label = ["Noisy data" nothing])
end

# ╔═╡ 4802fc3e-e9ce-4980-862f-e1c8fc0f3147
md"""### 1.2 Automated Identification of Nonlinear Interactions
Now, we introduce neural networks to help us identifying nonlinear interactions. In particular, we assume that we are give the noisy data $(t_i,\tilde u^i)$ and know that the governing equations are of the following form

$\begin{align*}
\dot u_1 &= +\alpha \,u_1 + f_1(u_1, u_2)\\
\dot u_2 &=  - \delta\, u_2 + f_2(u_1,u_2)
\end{align*}$

The aim is now to approximate the __unknown__ nonlinear interaction map $F(u_1,u_2)=(f_1(u_1,u_2),f_2(u_1,u_2))^\top$ using a neural network and the data $(t_i,\tilde u^i)$ to train it.

For scientific machine learning it has proven advantageous to use __Gaussian Radial Basis Funcions__ as activation functions. We define
"""

# ╔═╡ bd090e78-7ebd-43c5-b338-603746b5bccd
rbf(x) = exp.(-(x.^2)); 

# ╔═╡ d5886faf-9563-46c3-bfc2-d70f017adfc5
md"""Next, we create the neural network giving us a map $\tilde F_\theta:\mathbb{R}^2\to\mathbb{R}^2$, where $\theta$ denotes the parameters of network. The latter has four layers. 

The first layer has $2\times5+5 = 15$ parameters while the second and third layer have each $5\times5+5 = 30$ parameters, the first, second, and third use the Gaussian RBF activation function. The last layer is linear and  has $5\times2+2 =12$ parameters. In total, our neural network has $15+30+30+12 = 87$ parameters. We use the package `DiffEqFlux.jl` to construct the network. It is based on the `Flux.jl` package and adds functionality and improvements for scientific machine learning."""

# ╔═╡ a348c3f6-de44-47d3-9926-d27e0e4e17e3
F̃ = FastChain(
	FastDense(2,5,rbf), FastDense(5,5, rbf), FastDense(5,5, rbf), FastDense(5,2)
	);

# ╔═╡ b7fe8879-efc2-4057-85c2-a433f00ce52c
md"""We initialize the parameters $\theta$ for the network with some (random) values."""

# ╔═╡ ca308769-6eb7-4b87-b848-6ea3b153918c
	θ = initial_params(F̃)

# ╔═╡ 13246e42-961a-4365-a6da-dabf447b5ffe
md"""The size of $\theta$ is indeed 87."""

# ╔═╡ 4184b520-5771-40c6-abda-8d87a8225057
size(θ)

# ╔═╡ 2d80d2b5-07ea-4afd-b8f8-71f8cb79d3da
md"""In the next step, we define the nonlinear ODE system with our network giving the nonlinear interaction. Note that we also have the original parameters as arguments, however, we only use the first and last parameters $\alpha$ and $\delta$ that are supposed to be known. This is analogous to the definition of the `lotka_volterra!` function."""

# ╔═╡ 0e6d311d-d1bd-4baf-b2aa-13b50920facf
function ude_dynamics!(du, u, θ, t, p_true) # UDE = Universal Differential Equation
	f̃ = F̃(u,θ)
	α, β, γ, δ = p_true    # p_true contains the original ODE parameters
	du[1] = +α*u[1] + f̃[1]
	du[2] = -δ*u[2] + f̃[2]
end

# ╔═╡ 727f8eea-d300-4afb-90e7-e7187cdbec83
md"""Since the original ODE parameters $\alpha, \beta, \gamma, \delta$ are fixed, we create a so-called closure by binding the fixed parameters. Note that only $\alpha$ and $\delta$ are supposed to be known."""

# ╔═╡ c1088a4c-72f9-4a63-bb81-793927c67ebf
nn_dynamics!(du,u,θ,t) = ude_dynamics!(du,u,θ,t, p_)

# ╔═╡ 358fbffd-fc7a-4348-a70e-212414134112
md"""As before we create an ODE system with the nonlinearity given by our network, the initial condition is now given by the original initial condition $u^0$ plus noise. The parameters for the system are the parameters of the neural network."""

# ╔═╡ ea6ec827-5090-443c-8030-39dad523b42d
prob_nn = ODEProblem(nn_dynamics!, Ũ[:,1], tspan, θ); # Ũ contains the noisy data

# ╔═╡ 9ab39442-95ca-496a-a624-8d62a66975a7
md"""In order to train the network, we have to define a loss function. This is done by solving the system with the current parameters. Note that we have to differentiate the ODE solver for the optimization method. Thus, we specify that we use `ForwardDiffSensetivity`."""

# ╔═╡ ad5af2aa-fe1b-4eb1-86cf-49edf21d2d51
function predict(θ̂, X = Ũ[:,1], T = t) # X and T have default values
	Array(solve(prob_nn, Vern7(), u0 = X, p = θ̂,
			tspan = (T[1], T[end]), saveat = T,
			abstol=1e-6, reltol=1e-6,
			sensealg = ForwardDiffSensitivity()
			))
end

# ╔═╡ 268584ef-2cb7-4939-8f13-d60ab3fddb07
md"""The actual loss function solves the ODE system and computes the squared $\ell^2$ norm of the solution to the neural network problem and the "measurements" $\tilde u^i$."""

# ╔═╡ bd0876c3-b0be-461e-b5f9-58322bf6357b
function loss(θ̂)
	X̂ = predict(θ̂)
	sum(abs2, Ũ .- X̂)
end

# ╔═╡ 90e688eb-82be-4f83-a3de-000209ca3dc3
md"""The following array keeps track of the computed losses along the training."""

# ╔═╡ cc9ebd90-e970-4bf7-a3ed-bedf8308a6e0
losses = Float32[]; # an empty float32 array, to be filled during the training

# ╔═╡ 1cf5e475-63af-4699-961b-a66349a6c8af
md"""We define a callback function that is called during the training to fill the `losses` array and to output some information. We could also do some visualization, e.g. show the intermediate solutions."""

# ╔═╡ 3d796ca9-243e-4b72-b912-fe18856d2847
callback(θ̂, ℓ) = begin
	push!(losses, ℓ) # append to losses array
	if length(losses)%50 == 0
		println("Current loss after $(length(losses)) iterations: $(losses[end])")
	end
	false
end

# ╔═╡ 9d49309e-6a0d-4805-9769-5c35585ffb38
md"""Let's start the actually training, we first use the [ADAM algorithm](https://en.wikipedia.org/wiki/Stochastic_gradient_descent#Adam) (Adaptive Moment Estimation) and then use the minimizer for the [BFGS method](https://en.wikipedia.org/wiki/Broyden%E2%80%93Fletcher%E2%80%93Goldfarb%E2%80%93Shanno_algorithm) for better convergence. """

# ╔═╡ c4628747-9420-4c22-922a-9600614935fd
res1 = DiffEqFlux.sciml_train(loss, θ, ADAM(0.1f0), cb=callback, maxiters = 200);

# ╔═╡ 2eaa0094-6b7d-4c52-8100-dc36cbcabc8b
res2 = DiffEqFlux.sciml_train(loss, res1.minimizer, BFGS(initial_stepnorm=0.01f0), cb=callback, maxiters = 10000);

# ╔═╡ 09983a1a-1148-4336-b4f5-6607400b4a49
md"""### 1.3 Results
We first output the losses during training.
"""

# ╔═╡ c19bd911-39dd-49fa-b04e-ee699c499ac5
begin
	pl_losses = plot(1:200, losses[1:200], yaxis = :log10, xaxis = :log10, xlabel = "Iterations", ylabel = "Loss", label = "ADAM", color = :blue, lw = 3)
plot!(201:length(losses), losses[201:end], yaxis = :log10, xaxis = :log10, xlabel = "Iterations", ylabel = "Loss", label = "BFGS", color = :red, lw = 3)
end

# ╔═╡ 5fbe1674-1a6c-48bb-9975-451e486701e0
md"""Next, we use the final minimizer $\theta_\text{trained}$ and compare the respective solution $\hat u$ to the measurements $\tilde u$."""

# ╔═╡ 7a8a0a11-1ae2-4cb2-9002-9b7078f0a537
begin
	θ_trained = res2.minimizer

	Û = predict(θ_trained, Ũ[:,1], t[1]:0.05f0:t[end]) # solve the NN ODE
	pl_trajectory = plot(t[1]:0.05f0:t[end], transpose(Û), xlabel = "t", ylabel ="u₁(t), u₂(t)",  label = ["UDE Approximation" nothing])
	scatter!(t, transpose(Ũ), color = :black, label = ["Measurements" nothing])
end

# ╔═╡ 82ebd361-8c2b-4d5d-b4e1-22320381a192
md"""Let us see how well the original nonlinear function $F$ is approximated by our network $\tilde F$ along the predicted solution $\hat u$, i.e., we plot $F(\hat u) = (-\beta\hat u_1\cdot \hat u_2, \gamma \hat u_1\cdot\hat u_2)$ and $\tilde F_{\theta_\text{trained}}(\hat u)$"""

# ╔═╡ de0884a5-e609-4890-ad35-3f93a0d60c5a
begin
	# original interactions of the predictor
	α, β, γ, δ = p_
	G = [-β*(Û[1,:].*Û[2,:])'; γ*(Û[1,:].*Û[2,:])']
	# Neural network guess
	G̃ = F̃(Û,θ_trained)

	pl_reconstruction = plot(t[1]:0.05f0:t[end], transpose(G̃), xlabel = "t", ylabel ="F̃(x,y)", color = :red, label = ["UDE Approximation" nothing], lw = 3)
	plot!(t[1]:0.05f0:t[end], transpose(G), color = :black, label = ["True Interaction" nothing], lw = 3)
end

# ╔═╡ f8451a8e-14db-483c-83ee-8f4483584522
md"""We trained our network on observations on the time interval $[0,3]. How does our network perform for a longer time interval?"""

# ╔═╡ 19ea9b2b-fdf8-4fdb-a9d0-9fb0b03e1dee
begin
	T = 5.0f0
	prob_nn_longterm = ODEProblem(nn_dynamics!, Ũ[:,1], (0,T), θ_trained);
	# compute true solution
	true_solution = solve(remake(prob, tspan=(0.0,T),p=p_),Tsit5(),saveat=0.1)
	# compute predicted solution
	predicted_solution = solve(prob_nn_longterm, Vern7())
	plot(true_solution, label= ["True solution" nothing], linestyle = :dot, lw=4)
	plot!(predicted_solution, lw=4, label = ["Predicted solution" nothing])
end

# ╔═╡ dfbf1deb-4d88-4b57-99ec-34566edfd64d
md"""That is very disappointing and shows that NN are not the magical tools that can solve any problem we throw at them. But we can improve the result with the __Sparse Identification of Nonlinear Dynamics method__ (SINDy).

The method assumes that the right-hand side in the equation system can be describe by a sparse basis of "simple" functions in a provided library $\boldsymbol\Phi$ of functions, e.g. monomials. The coefficients $\boldsymbol\Lambda$ with respect to this basis are computed via a minimization problem of the form

$\begin{equation}
\boldsymbol\Lambda_*\in\mathrm{Arg\,min}_{\boldsymbol{\Lambda}}\Big\{ \big\|\boldsymbol{\dot U} -\boldsymbol\Lambda\boldsymbol\Phi(\boldsymbol{U})\big\|_2 + \kappa\|\boldsymbol\Lambda\|_1 \Big\}
\end{equation}$

where $\boldsymbol U$ are the solutions predicted by our NN system.
However, since we have partial knowledge about the ODE system (given by the terms multiplied by $\alpha$ and $\delta$), we can use $\tilde F_\theta$ from our trained NN
instead of the time derivative $\boldsymbol{ \dot U}$.

We first create a basis consisting of monomials in $u=(u_1,u_2)$ with maximal order 5.
For fun we also throw the sine function into the mix.
"""

# ╔═╡ 56ea05c8-6a95-4b27-9a32-efbdda498ae8
# Create a Basis
begin
	@variables u[1:2]
	b = [polynomial_basis(u, 5); sin.(u)]
	basis = Basis(b, u)
end

# ╔═╡ 333ee57a-1fb4-4e16-a765-002ef9421c10
md""" The output reads:

${\begin{align} \varphi{_1} =& 1 \\ \varphi{_2} =& u{_1} \\ \varphi{_3} =& u{_1}^{2} \\ \varphi{_4} =& u{_1}^{3} \\ \varphi{_5} =& u{_1}^{4} \\ \varphi{_6} =& u{_1}^{5} \\ \varphi{_7} =& u{_2} \\ \varphi{_8} =& u{_1} u{_2} \\ \varphi{_9} =& u{_1}^{2} u{_2} \\ \varphi{_{10}} =& u{_1}^{3} u{_2} \\ \varphi{_{11}} =& u{_1}^{4} u{_2} \\ \varphi{_{12}} =& u{_2}^{2} \\ \varphi{_{13}} =& u{_2}^{2} u{_1} \\ \varphi{_{14}} =& u{_2}^{2} u{_1}^{2} \\ \varphi{_{15}} =& u{_2}^{2} u{_1}^{3} \\ \varphi{_{16}} =& u{_2}^{3} \\ \varphi{_{17}} =& u{_2}^{3} u{_1} \\ \varphi{_{18}} =& u{_2}^{3} u{_1}^{2} \\ \varphi{_{19}} =& u{_2}^{4} \\ \varphi{_{20}} =& u{_2}^{4} u{_1} \\ \varphi{_{21}} =& u{_2}^{5} \\ \varphi{_{22}} =& \sin\left( u{_1} \right) \\ \varphi{_{23}} =& \sin\left( u{_2} \right) \end{align} }$
"""

# ╔═╡ 858c513c-1164-462e-90ce-24dcc79745dc
md"""To determine the coefficients for the basis, we use the SR3 (Sparse Relaxed Regularization Regression) algorithm to solve the minimization problem above."""

# ╔═╡ ba9d5329-ad43-4e0f-b60a-d25fc37fa58d
opt = SR3(Float32(1e-2), Float32(0.1));

# ╔═╡ e729c42c-805b-46fd-b6bb-d858cf4c1f20
md"""We have to define thresholds, which should be used in the search process."""

# ╔═╡ 1f124224-1f7b-4699-8dbc-eaa1c9241675
λ = Float32.(exp10.(-7:0.1:5))

# ╔═╡ 40318de3-c53f-4f28-8e78-5d9e2bd98172
md"""We also define a target function to choose the results from: L2-Error of the model.
"""

# ╔═╡ 65d6e1c8-17ec-4e86-9651-ca1230e05a61
g(x) = x[1] < 1 ? Inf : norm(x, 2);

# ╔═╡ 8a8a601b-4b38-439e-b2ee-c41d380278e4
md"""We run the optimization problem first on the true nonlinear map $F$ to find the optimal coefficients. This is just to verify that the method does what it should."""

# ╔═╡ 39ce4bad-44e3-427f-8f5f-3936209dde78
begin
	Ψ = SINDy(Û, G, basis, λ, opt, g = g, maxiter = 10000)
	Ψ.coeff[:]
end

# ╔═╡ 66652f86-da03-41d3-a31c-1e710af3c1eb
md"""Now let us do the same for the predicted nonlinear interaction $\tilde F_{\theta_\text{trained}}$."""

# ╔═╡ 3b53ee1b-743a-4228-baaf-b21b0d40ccbf
begin
	Ψ̃ = SINDy(Û, G̃, basis, λ,  opt, g = g, maxiter = 50000, normalize = true, denoise = true, convergence_error = Float32(1e-10)) # Succeed
	Ψ̃.coeff[:]
end

# ╔═╡ 3c8fba9d-e896-43d7-b7ea-8d18ef66df7c
md"""We save the nonzero parameters."""

# ╔═╡ 19df7f30-1c7a-4e83-a04c-3dc11ba0cbfd
Λ = parameters(Ψ)

# ╔═╡ e7bdea03-9463-4312-b667-89e4a78b7d08
Λ̃ = parameters(Ψ̃)

# ╔═╡ e19f94db-8f3c-4d23-8a69-be8e6cd1aa5c
md"""We set a new basis by taking only the basis function in `b` into account with the two largest coefficients."""

# ╔═╡ 399c4ccc-1ca7-41ae-a04f-4f0d38043522
bnew = Basis((u, p, t)->Ψ̃(u, [1f0; 1f0], t), u)

# ╔═╡ bba7a594-7eb9-4fdf-8aec-9af91e333490
md"""We determine the coefficients with respect to the new basis."""

# ╔═╡ b462126d-65bb-4b91-86ee-85eab9fbeb26
Ψf = SINDy(Û, G̃, bnew, STRRidge(0.01f0), maxiter = 100, convergence_error = Float32(1e-18))

# ╔═╡ 51f89d8a-8960-44fb-8f64-082e87f86812
md"""What are the coefficients with respect to the new basis?"""

# ╔═╡ 651fff0c-d960-46ca-aeb9-4a5210b373bb
Λf = parameters(Ψf)

# ╔═╡ 8053a954-3f9b-4219-92ef-8154323df543
md"""That's better! We now use this result to construct an improved ODE system:"""

# ╔═╡ c53e290f-7a03-4b77-9535-c701404502a3
function recovered_dynamics!(du, u, Λ̂, t, p_true)
    û = Ψ̃(u, Λ̂) # Network prediction
    du[1] =  p_true[1]*u[1] + û[1]
    du[2] = -p_true[4]*u[2] + û[2]
end

# ╔═╡ 16318713-d8d2-4611-a3c6-8eff31d90375
md"""We define a closure to bind the fixed ODE parameters and solve the associated ODE system."""

# ╔═╡ 5766ef95-d5d5-43af-bd8a-4784dd439c4d
estimated_dynamics!(du,u,Λ̂,t) = recovered_dynamics!(du,u,Λ̂,t,p_)

# ╔═╡ 92be4059-19f8-4c3c-893c-29933d74959b
md"""Finally, we compare the solution of the estimated dynamics with the actual solution of the ODE system."""

# ╔═╡ c40053ec-964b-402f-b705-458d93361983
begin
	t_long = (0.0f0, 30.0f0)
	# solve estimated problem
	estimation_prob = ODEProblem(estimated_dynamics!, u0, t_long, Λf);
	estimate_long = solve(estimation_prob, Tsit5(), saveat = 0.1)
	# solve true problem on long time interval
	prob_long = ODEProblem(lotka_volterra!, u0, t_long, p_);
	solution_long = solve(prob_long, Tsit5(), saveat = 0.1)
	plot(estimate_long, lw = 2, label = ["estimated solution" nothing])
	plot!(solution_long, lw = 2, alpha = 0.5,color=:black, label = ["true solution" nothing], linestyle = :dash)
	scatter!(t,transpose(Ũ), label = ["Noisy data" nothing])
	
end

# ╔═╡ aee901dc-e5a6-4ed8-af0c-16b3924c989c
md"""
### 1.4 Conclusion
We used a small neural network to reconstruct the unknown nonlinear interaction in the Lotka--Volterra model based on noisy data on the time interval $[0,3]$. However, the neural network could not be used directly to predict the behavior of the system. Instead, we used the SINDy method to reconstruct the nonlinear interaction. 

While the SInDy method normally approximates derivatives using a spline over the data points or similar numerical techniques, here we have $\tilde F_\theta(u_1,u_2)$ as an estimator of the derivative for only the missing terms of the model and we can perform a sparse regression on samples from the trained $\tilde F_\theta(u_1,u_2)$ to reconstruct only the unknown interaction equations. Even though the original data
did not contain a full period of the cyclic solution, the resulting fit is then able to accurately extrapolate from the short time series data.

 Likewise,  when  attempting  to  learn  full  ODE  with  the original SInDy approach on the same trained data with the analytical derivative  values,  it seems to be not possible to  recover  the  exact  original  equations  from  the sparse  regression,  indicating  that  the  knowledge-enhanced  approach  increases the robustness equation discovery.

See [A. Bills,  S.  Sripad,  W.  L.  Fredericks,  M. Guttenberg,   D.   Charles,   E.   Frank,   and   V. Viswanathan.   Universal  Battery  Performance  and  Degradation  Model for Electric Aircraft](https://arxiv.org/abs/2008.01527) for a more practical example.

"""

# ╔═╡ b81b977e-5163-4a99-a44d-a4849ca5bff5
md"""
## 2. Fisher-KPP equation

Next, we consider a PDE example, namely, the famous Fischer-KPP equation:

$\begin{equation}
\partial_t \rho = D \partial_{xx} \rho + r \rho(1{-}\rho)
\end{equation}$

subject to periodic boundary conditions. In particular, we consider $x\in(0,1)$
and $t\in[0,T]$. Here, $\rho$ represents population density of a species, $r>0$ 
is the local growth rate and $D>0$ is the diffusion coefficient. 

We aim to reconstruct the (discrete) equation 
via neural networks for the reaction  and the diffusion term

$\begin{equation}
\partial_t \tilde\rho = \tilde D\tilde\Delta\tilde \rho +\tilde F_\theta(\tilde \rho).
\end{equation}$

### 2.1 Generating training data
First we set the diffusion coefficient and the reaction rate
"""

# ╔═╡ 41ae0ff3-c903-4b05-851d-7d6ed33cd34d
begin
	D = 0.01;
	r = 1.0;
	nothing
end

# ╔═╡ 8adbfa98-627c-4565-b1f0-cd68100286ab
md"""
For the spatial and temporal discretization, we set
"""

# ╔═╡ 03341e4f-5c58-473d-ac3d-071b99b2032f
begin
	X  = 1.0;   # domain
	Tf = 5;     # time horizon
	dx = 0.04;  # space discretization
	dt = Tf/10;  # time step
	x  = collect(0:dx:X);
	s  = collect(0:dt:Tf);
	Nx = Int64(X/dx+1);
	Nt = Int64(Tf/dt+1);
	nothing
end

# ╔═╡ a3155cee-d86a-457c-a929-9fede7aadaef
md"""
The initial condition is given by
"""

# ╔═╡ d03ae9c2-38c6-445a-baf0-13c5342caa6f
begin
	Amp = 1.0;
	Delta = 0.2
	rho0 = Amp*(tanh.((x .- (0.5 - Delta/2))/(Delta/10)) - tanh.((x .- (0.5 + Delta/2))/(Delta/10)))/2
	nothing
end

# ╔═╡ 76eebfce-1ea4-4fc9-bf4d-05ce1d5f4ec1
plot(x,rho0, lw = 3, label = "Initial condition", xlabel = "x", ylabel = "ρ₀")

# ╔═╡ a76a8dd8-3fc6-4d33-8c9e-71e67af431c4
md"""
Define reaction term and discrete Laplacian with periodic boundary conditions.
"""

# ╔═╡ 1d4634dc-8af1-4371-b36b-177863e05e6b
begin
	reaction(u) = r * u .* (1 .- u)
	lap = diagm(0 => -2.0 * ones(Nx), 1=> ones(Nx-1), -1 => ones(Nx-1)) ./ dx^2
	lap[1,end] = 1.0/dx^2
	lap[end,1] = 1.0/dx^2
	nothing
end

# ╔═╡ 52b605f9-cb28-4e3e-91c7-19ce54ff6105
md"Setup the equation"

# ╔═╡ 66224182-6c81-4ea3-be10-6eb73b9c218b
function rc_ode(rho, p, t)
    #finite difference
    D * lap * rho + reaction.(rho)
end

# ╔═╡ b40ef72c-f83f-48b5-a5c8-9da6380d6231
md"Solve the system"

# ╔═╡ 12cb668a-300e-4f98-8a75-86c3531b449b
begin
	prob_FKPP = ODEProblem(rc_ode, rho0, (0.0, Tf), saveat=dt)
	sol = solve(prob_FKPP, Tsit5());
	ode_data = Array(sol);
	
end

# ╔═╡ 46756164-f224-4ac7-9a20-ff6277bc0f31
md"Plot the solution"

# ╔═╡ 86ef0014-e401-4c08-ab5f-770cb845140f
contourf(s,x,ode_data, xlabel = "time s", ylabel = "x")

# ╔═╡ 1532c921-4f3e-4c97-8785-84a2da6fed99
md"""
### 2.2 Creating the neural network

The neural network for the reaction part is similar as for the Lotka-Volterra model but larger and uses the $\tanh$ function instead of the Gaussian radial basis function.

"""

# ╔═╡ 6a71bcf8-ca8f-4d4d-9451-a3e29b753d59
begin
	n_weights = 10

	rx_nn = Chain(Dense(1, n_weights, tanh),
                Dense(n_weights, 2*n_weights, tanh),
                Dense(2*n_weights, n_weights, tanh),
                Dense(n_weights, 1),
                x -> x[1])
end

# ╔═╡ 74445bb5-5c55-438d-b413-8056f6bd7c6a
md"The neural network for the diffusion part is of convolutional type, it is supposed to approximate the stencil $(1, -2, 1)$. Since our input is a two-dimensional array, the weights have to be reshaped into a 4-dimensional array. The bias is set to zero. We add padding to incoporate the periodic boundary conditions."

# ╔═╡ 78deaba4-ee72-4d33-bd8a-cac34ba8ac90
begin
	w_err = 0.0
	init_w = reshape([1.1 -2.5 1.0], (3, 1, 1, 1))
	diff_cnn_ = Conv(init_w, [0.], pad=(0,0,0,0))
end

# ╔═╡ 4a1d833b-6f4b-4eab-8709-5c820bbfbc8d
md"We set a initial guess for the diffusion coefficient. In particular, it should be close to `D/dx^2=6.25`."

# ╔═╡ d9c9e922-8281-46e5-bb43-47e80c750db2
D₀ = [6.5];

# ╔═╡ 2771c663-bee2-4763-94eb-08d22e430e21
md"""We have to glue the NNs for the reaction part, the diffusion stencil, and the diffusion coefficient together. This is achieved as follows:"""

# ╔═╡ c9a91eb0-fdd7-4804-b458-fd54ecc1ec3a
begin
	θ₁, re1 = Flux.destructure(rx_nn)
	θ₂, re2 = Flux.destructure(diff_cnn_)
	θₜ = [θ₁; θ₂; D₀]
	full_restructure(θₜ) = re1(θₜ[1:length(θ₁)]),re2(θₜ[(length(θ₁)+1):end-1]),θₜ[end]
end

# ╔═╡ 4d0ee20c-980d-41c9-8806-59ebb6c5d743
md"We construct the ODE system that is given by the combined NN. Note that we use `end` here for the parameter array `p` since the stencil is stored at the fourth, third,and second last position in the array."

# ╔═╡ a328db69-e3d7-4e67-be5d-303befeb4ecf
function nn_ode(u,p,t)
    rx_nn = re1(p[1:length(θ₁)])
	# periodic boundary condition left
    u_cnn_1   = [p[end-4] * u[end] + p[end-3] * u[1] + p[end-2] * u[2]]
	# discrete NN-Laplacian
    u_cnn     = [p[end-4] * u[i-1] + p[end-3] * u[i] + p[end-2] * u[i+1] for i in 2:Nx-1]
	# periodic boundary condition right
    u_cnn_end = [p[end-4] * u[end-1] + p[end-3] * u[end] + p[end-2] * u[1]]
	
	# sum of reaction and diffusion part, `p[end]` is the scaled diffusion coeff
    [rx_nn([u[i]])[1] for i in 1:Nx] + p[end] * vcat(u_cnn_1, u_cnn, u_cnn_end)
end

# ╔═╡ a8a43d16-6b3b-41bf-9abc-3c527687c891
md"We solve the NN ODE system. For some reason we have to use `concrete_solve` here"

# ╔═╡ 4127d2bc-29fc-494d-a1e6-1a5de5c75667
begin
	prob_nn_FKPP = ODEProblem(nn_ode, rho0, (0.0, Tf), θₜ, saveat=dt)
	sol_nn_FKPP1 = solve(prob_nn_FKPP,Tsit5())
	ode_data_nn = Array(sol_nn_FKPP1)
end

# ╔═╡ f1efc73f-d303-4e43-ab15-57d19eb333fa
contourf(s,x, ode_data_nn, xlabel = "time s", ylabel = "x")

# ╔═╡ e01e4c8d-2b28-4edc-ab6e-f9b50ee4953d
md"""It remains to define a loss function. This is given by the $\ell^2$ norm of the difference of NN solution and measurements. Moreover, we force the weights of the CNN (i.e. the stencil) to add up to zero."""

# ╔═╡ bb228f64-6436-44ee-8ec8-fb36da8af45e
begin
	function predict_rd(θ)
Array(solve(prob_nn_FKPP,Tsit5(),u0=rho0, p=θ, saveat=dt, 
				abstol=1e-6, reltol=1e-6,
				sensealg = ForwardDiffSensitivity()))
	end

	function loss_rd(p)
    	pred = predict_rd(p)
    	sum(abs2, ode_data .- pred) + 10^2 * abs(sum(p[end-4 : end-2])), pred
	end
end

# ╔═╡ ca0a9d61-d52b-4ca7-8f78-27206a3ce869
md"""We define a callback that saves the losses for each step."""

# ╔═╡ 32e57e24-8367-4f22-800e-25f6b4696fec
begin
	losses_FKPP = Float32[]
	cb_FKPP = function (p,l,pred)
		push!(losses_FKPP, l) # append to losses array
		if length(losses_FKPP)%5 == 0
			println("Current loss after $(length(losses_FKPP)) iterations: $(losses_FKPP[end])")
		end
		false
	end
end

# ╔═╡ cdd6c044-2f29-4643-aeb1-e513f08fa4a8
md"Do the actual training using again a combinition of ADAM and BFGS."

# ╔═╡ 8a5f9713-8076-45d0-b722-c55fa40251b2
begin
	println("Starting first run with ADAM")
	res1_FKPP = DiffEqFlux.sciml_train(loss_rd, θₜ, ADAM(0.001), cb=cb_FKPP, maxiters = 100);
	println("Finished first run with ADAM")
end

# ╔═╡ aafd511f-9d25-40d6-96d5-decc85eeb1b8
begin
		println("Starting second run with ADAM")
	res2_FKPP = DiffEqFlux.sciml_train(loss_rd, res1_FKPP.minimizer, ADAM(0.001), cb=cb_FKPP, maxiters = 300);
	println("Finished second run with ADAM")
end

# ╔═╡ 0712b202-916f-4392-92f6-706398b9c1a2
begin
	res3_FKPP = DiffEqFlux.sciml_train(loss_rd, res2_FKPP.minimizer, BFGS(initial_stepnorm=0.01f0), cb=cb_FKPP, maxiters = 1000)
end

# ╔═╡ e6da664d-b65a-4f3f-9fde-c3a7dacba893
begin
	plot(1:100, losses_FKPP[1:100], yaxis = :log10, xaxis = :log10, xlabel = "Iterations", ylabel = "Loss", label = "ADAM 1", color = :blue, lw = 3)
		plot!(101:400, losses_FKPP[101:400], yaxis = :log10, xaxis = :log10, xlabel = "Iterations", ylabel = "Loss", label = "ADAM 2", color = :green, lw = 3)
	plot!(401:length(losses_FKPP), losses_FKPP[401:end], yaxis = :log10, xaxis = :log10, xlabel = "Iterations", ylabel = "Loss", label = "BFGS", color = :red, lw = 3)
end

# ╔═╡ a94a2c7e-aadd-4ad1-bc3a-8a92615d80ee
md"""
### 2.3 Results
Let us first plot the predicted solution.
"""

# ╔═╡ 8f15c13f-e610-4947-bdcb-30c15cd4c340
begin
	pstar3=res3_FKPP.minimizer
	cur_pred3 = predict_rd(pstar3)
	contourf(s,x,cur_pred3, xlabel = "time s", ylabel = "x")
end


# ╔═╡ d0b2446d-909d-4ef5-aa2f-7af5633a536e
md"and the difference to the original solution."

# ╔═╡ de635870-9f54-4d99-83ce-5e8c763d4f1e
contourf(s,x,ode_data-cur_pred3, xlabel = "time s", ylabel = "x")

# ╔═╡ 85fee47e-d687-4591-ad2b-c51085be9315
md"Output the diffusion coefficient"

# ╔═╡ 03fe3e46-37ed-4190-9943-529e4b5eb629
D_pred = pstar3[end]

# ╔═╡ 8d8b34e8-11bc-4390-994a-2b5dc7d040cd
md"the predicted stencil"

# ╔═╡ 8af620ee-2d3b-446c-95ba-cc55a4a10cef
pstar3[end-4:end-2]

# ╔═╡ ae1dc823-05c5-4756-a70b-b4da12d474a7
md"And plot the predicted reaction term"

# ╔═╡ 889e55ac-17ba-4e51-ab66-a97781737b8f
begin
	ρ=collect(0:0.01:1)
	react3 = re1(pstar3[1:length(θ₁)])
	plot(ρ,react3.([[elem] for elem in ρ]), label= "predicted reaction term", lw = 3)
	plot!(ρ,reaction.(ρ), label = "true reaction term", lw = 3)
end

# ╔═╡ b90c266e-164a-431c-8aa4-fda02401f4c9
TableOfContents()

# ╔═╡ Cell order:
# ╟─631a008c-a667-11eb-3af0-e99874435c33
# ╠═fdb7be1f-94ab-4c3f-a5fb-05d8433d8e16
# ╟─56747210-79d6-4862-a5b2-1a707d713c93
# ╠═c7475b08-ebbd-4d33-9320-24f67b72b9c5
# ╟─8bd1dec7-a592-4980-8f2f-f35cdb829096
# ╠═e352104e-d543-4b01-a905-06f14728fed6
# ╟─a4ab0f0a-8cfd-458a-807d-f59f982cbcd5
# ╠═59461d05-df8f-487d-98dc-46e524a49603
# ╟─696e727c-33ef-4f07-b5f1-5e90330eef57
# ╠═5602c03b-ee48-4073-990d-2be18b873da7
# ╟─c29da13f-9d85-484e-9617-9f9dc732d0c3
# ╠═7fa06933-f018-4b48-bdac-e2bd21b37ced
# ╟─ab36c055-a97a-4204-81f8-26a5152b56e5
# ╟─7efe7664-302b-4cbf-a276-098fdf7ef4b4
# ╠═d4c5df84-3ed4-41df-b651-f5e4605bc5de
# ╟─d1ae256f-091b-4459-b759-825b18549128
# ╠═1d291e76-3d2f-4cae-9c2f-5009a4d58e53
# ╟─63667f95-0431-4cd6-874d-701ae57bbb1d
# ╠═0ede66e5-4c24-488c-85f0-9d4b69cabe68
# ╟─4802fc3e-e9ce-4980-862f-e1c8fc0f3147
# ╠═bd090e78-7ebd-43c5-b338-603746b5bccd
# ╟─d5886faf-9563-46c3-bfc2-d70f017adfc5
# ╠═a348c3f6-de44-47d3-9926-d27e0e4e17e3
# ╟─b7fe8879-efc2-4057-85c2-a433f00ce52c
# ╠═ca308769-6eb7-4b87-b848-6ea3b153918c
# ╟─13246e42-961a-4365-a6da-dabf447b5ffe
# ╠═4184b520-5771-40c6-abda-8d87a8225057
# ╟─2d80d2b5-07ea-4afd-b8f8-71f8cb79d3da
# ╠═0e6d311d-d1bd-4baf-b2aa-13b50920facf
# ╟─727f8eea-d300-4afb-90e7-e7187cdbec83
# ╠═c1088a4c-72f9-4a63-bb81-793927c67ebf
# ╟─358fbffd-fc7a-4348-a70e-212414134112
# ╠═ea6ec827-5090-443c-8030-39dad523b42d
# ╟─9ab39442-95ca-496a-a624-8d62a66975a7
# ╠═ad5af2aa-fe1b-4eb1-86cf-49edf21d2d51
# ╟─268584ef-2cb7-4939-8f13-d60ab3fddb07
# ╠═bd0876c3-b0be-461e-b5f9-58322bf6357b
# ╟─90e688eb-82be-4f83-a3de-000209ca3dc3
# ╠═cc9ebd90-e970-4bf7-a3ed-bedf8308a6e0
# ╟─1cf5e475-63af-4699-961b-a66349a6c8af
# ╠═3d796ca9-243e-4b72-b912-fe18856d2847
# ╟─9d49309e-6a0d-4805-9769-5c35585ffb38
# ╠═c4628747-9420-4c22-922a-9600614935fd
# ╠═2eaa0094-6b7d-4c52-8100-dc36cbcabc8b
# ╟─09983a1a-1148-4336-b4f5-6607400b4a49
# ╠═c19bd911-39dd-49fa-b04e-ee699c499ac5
# ╟─5fbe1674-1a6c-48bb-9975-451e486701e0
# ╠═7a8a0a11-1ae2-4cb2-9002-9b7078f0a537
# ╟─82ebd361-8c2b-4d5d-b4e1-22320381a192
# ╠═de0884a5-e609-4890-ad35-3f93a0d60c5a
# ╟─f8451a8e-14db-483c-83ee-8f4483584522
# ╠═19ea9b2b-fdf8-4fdb-a9d0-9fb0b03e1dee
# ╟─dfbf1deb-4d88-4b57-99ec-34566edfd64d
# ╠═56ea05c8-6a95-4b27-9a32-efbdda498ae8
# ╟─333ee57a-1fb4-4e16-a765-002ef9421c10
# ╟─858c513c-1164-462e-90ce-24dcc79745dc
# ╠═ba9d5329-ad43-4e0f-b60a-d25fc37fa58d
# ╟─e729c42c-805b-46fd-b6bb-d858cf4c1f20
# ╠═1f124224-1f7b-4699-8dbc-eaa1c9241675
# ╟─40318de3-c53f-4f28-8e78-5d9e2bd98172
# ╠═65d6e1c8-17ec-4e86-9651-ca1230e05a61
# ╟─8a8a601b-4b38-439e-b2ee-c41d380278e4
# ╠═39ce4bad-44e3-427f-8f5f-3936209dde78
# ╟─66652f86-da03-41d3-a31c-1e710af3c1eb
# ╠═3b53ee1b-743a-4228-baaf-b21b0d40ccbf
# ╟─3c8fba9d-e896-43d7-b7ea-8d18ef66df7c
# ╠═19df7f30-1c7a-4e83-a04c-3dc11ba0cbfd
# ╠═e7bdea03-9463-4312-b667-89e4a78b7d08
# ╟─e19f94db-8f3c-4d23-8a69-be8e6cd1aa5c
# ╠═399c4ccc-1ca7-41ae-a04f-4f0d38043522
# ╟─bba7a594-7eb9-4fdf-8aec-9af91e333490
# ╠═b462126d-65bb-4b91-86ee-85eab9fbeb26
# ╟─51f89d8a-8960-44fb-8f64-082e87f86812
# ╠═651fff0c-d960-46ca-aeb9-4a5210b373bb
# ╟─8053a954-3f9b-4219-92ef-8154323df543
# ╠═c53e290f-7a03-4b77-9535-c701404502a3
# ╟─16318713-d8d2-4611-a3c6-8eff31d90375
# ╠═5766ef95-d5d5-43af-bd8a-4784dd439c4d
# ╟─92be4059-19f8-4c3c-893c-29933d74959b
# ╠═c40053ec-964b-402f-b705-458d93361983
# ╟─aee901dc-e5a6-4ed8-af0c-16b3924c989c
# ╟─b81b977e-5163-4a99-a44d-a4849ca5bff5
# ╠═41ae0ff3-c903-4b05-851d-7d6ed33cd34d
# ╟─8adbfa98-627c-4565-b1f0-cd68100286ab
# ╠═03341e4f-5c58-473d-ac3d-071b99b2032f
# ╟─a3155cee-d86a-457c-a929-9fede7aadaef
# ╠═d03ae9c2-38c6-445a-baf0-13c5342caa6f
# ╠═76eebfce-1ea4-4fc9-bf4d-05ce1d5f4ec1
# ╟─a76a8dd8-3fc6-4d33-8c9e-71e67af431c4
# ╠═1d4634dc-8af1-4371-b36b-177863e05e6b
# ╟─52b605f9-cb28-4e3e-91c7-19ce54ff6105
# ╠═66224182-6c81-4ea3-be10-6eb73b9c218b
# ╟─b40ef72c-f83f-48b5-a5c8-9da6380d6231
# ╠═12cb668a-300e-4f98-8a75-86c3531b449b
# ╟─46756164-f224-4ac7-9a20-ff6277bc0f31
# ╠═86ef0014-e401-4c08-ab5f-770cb845140f
# ╟─1532c921-4f3e-4c97-8785-84a2da6fed99
# ╠═6a71bcf8-ca8f-4d4d-9451-a3e29b753d59
# ╟─74445bb5-5c55-438d-b413-8056f6bd7c6a
# ╠═78deaba4-ee72-4d33-bd8a-cac34ba8ac90
# ╟─4a1d833b-6f4b-4eab-8709-5c820bbfbc8d
# ╠═d9c9e922-8281-46e5-bb43-47e80c750db2
# ╟─2771c663-bee2-4763-94eb-08d22e430e21
# ╠═c9a91eb0-fdd7-4804-b458-fd54ecc1ec3a
# ╟─4d0ee20c-980d-41c9-8806-59ebb6c5d743
# ╠═a328db69-e3d7-4e67-be5d-303befeb4ecf
# ╟─a8a43d16-6b3b-41bf-9abc-3c527687c891
# ╠═4127d2bc-29fc-494d-a1e6-1a5de5c75667
# ╠═f1efc73f-d303-4e43-ab15-57d19eb333fa
# ╟─e01e4c8d-2b28-4edc-ab6e-f9b50ee4953d
# ╠═bb228f64-6436-44ee-8ec8-fb36da8af45e
# ╟─ca0a9d61-d52b-4ca7-8f78-27206a3ce869
# ╠═32e57e24-8367-4f22-800e-25f6b4696fec
# ╟─cdd6c044-2f29-4643-aeb1-e513f08fa4a8
# ╠═8a5f9713-8076-45d0-b722-c55fa40251b2
# ╠═aafd511f-9d25-40d6-96d5-decc85eeb1b8
# ╠═0712b202-916f-4392-92f6-706398b9c1a2
# ╟─e6da664d-b65a-4f3f-9fde-c3a7dacba893
# ╟─a94a2c7e-aadd-4ad1-bc3a-8a92615d80ee
# ╠═8f15c13f-e610-4947-bdcb-30c15cd4c340
# ╟─d0b2446d-909d-4ef5-aa2f-7af5633a536e
# ╟─de635870-9f54-4d99-83ce-5e8c763d4f1e
# ╟─85fee47e-d687-4591-ad2b-c51085be9315
# ╠═03fe3e46-37ed-4190-9943-529e4b5eb629
# ╟─8d8b34e8-11bc-4390-994a-2b5dc7d040cd
# ╠═8af620ee-2d3b-446c-95ba-cc55a4a10cef
# ╟─ae1dc823-05c5-4756-a70b-b4da12d474a7
# ╠═889e55ac-17ba-4e51-ab66-a97781737b8f
# ╟─b90c266e-164a-431c-8aa4-fda02401f4c9
