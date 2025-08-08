using Lux: Lux, Chain, Dense, LayerNorm, gelu
using Random: Random
using Optimisers: Optimisers
using Reactant: Reactant, @trace, Const
using Enzyme: Enzyme

const rdev = Lux.reactant_device()
const cdev = Lux.cpu_device()

"""
    create_model(; latent = 8, width = 64, rng = Random.default_rng())

Create a neural network model for the neural ODE. The model consists of several
dense layers with GELU activation functions, followed by a layer normalization,
and finally a dense layer that outputs the predicted state and latent variables.

The `latent` parameter specifies the number of latent variables, and `width`
specifies the width of the hidden layers. The `rng` parameter is used for
initializing the model parameters.
"""
function create_model(; latent = 8, width = 64, rng = Random.default_rng())
    model = Chain(
        Dense(2 + latent => width, gelu),
        Dense(width => width, gelu),
        Dense(width => width, gelu),
        LayerNorm(width),
        Dense(width => 1 + latent)
    )
    ps, st = Lux.setup(rng, model)
    return (; model, ps, st)
end

"""
    step(model, ps, st, x, u)

Perform a single step of the neural ODE model. The function takes the model, the
parameters `ps`, the model state `st` (hidden variables - empty in most cases),
the state vector `x`, and the control input `u`. It concatenates the control
input and state, performs a forward pass through the model, and returns the
updated state and the new state of the model. The time step `dt` is used to
update the state based on the predicted derivative.
"""
function step(model, ps, st, x, u, dt)
    # Concatenate control input and state
    input = [u; x]
    # Forward pass through the model
    (dxdt, st) = model(input, ps, st)
    # Euler step
    return (x + dt * dxdt), st
end

"""
    loss(model, ps, st, x0, u, dt)
"""
function loss(model, ps, st, x0, u, y, dt)
    x = x0
    loss = eltype(x0)(0)
    n_steps = size(u, 1)
    @trace for i in 1:n_steps
        x, st = step(model, ps, st, x, u[i, :]', dt)
        loss += sum((x[1, :] - y[i, :]) .^ 2)  # MSE
    end
    loss /= length(u)  # Average loss over all steps
    return loss, st
end

function gradient(model, ps, st, x0, u, y, dt)
    return Enzyme.gradient(
        Enzyme.Reverse, Const(loss), Const(model), ps, Const(st), Const(x0), Const(u), Const(y), Const(dt)
    )[2]
end

function dostuff()
    rng = Random.Xoshiro(123)
    (model, ps, st) = create_model(; rng)
    n_steps = 13
    n_batch = 7
    x0 = randn(rng, Float32, 9, n_batch)  # Initial state
    u = randn(rng, Float32, n_steps, n_batch)  # Control input
    y = randn(rng, Float32, n_steps, n_batch)  # Target output
    dt = 0.1f0  # Time step
    # Transfer to device
    ps_ra = rdev(ps)
    st_ra = rdev(st)
    x0_ra = rdev(x0)
    u_ra = rdev(u)
    y_ra = rdev(y)
    # Perform a forward pass
    (loss_value_ra, st_ra) = Reactant.@jit loss(model, ps_ra, st_ra, x0_ra, u_ra, y_ra, dt)
    println("Reactant loss: ", cdev(loss_value_ra))
    (loss_value, st) = loss(model, ps, st, x0, u, y, dt)
    println("Regular loss: ", loss_value)
    # Perform a backward pass
    grad = Reactant.@jit gradient(model, ps_ra, st_ra, x0_ra, u_ra, y_ra, dt)
    return loss_value_ra, loss_value, grad
end
