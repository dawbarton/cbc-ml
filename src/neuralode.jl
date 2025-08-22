using Lux: Lux, LuxCore, Training, AutoEnzyme, MSELoss, Chain, Dense, LayerNorm, AbstractLuxLayer, AbstractLuxWrapperLayer, gelu, @set!
using Random: Random
using Optimisers: Optimisers, Adam, AdamW
using Reactant: Reactant, @trace, Const
using Enzyme: Enzyme
using MLUtils: MLUtils, DataLoader
using Printf: @printf
using LinearAlgebra: LinearAlgebra, pinv, dot
using Statistics: mean, std

const rdev = Lux.reactant_device()
const cdev = Lux.cpu_device()

"""
    EulerStepper(network::N, dt::Float32)

A wrapper layer for a neural ODE that uses the Euler method for stepping through
the state. It takes a Lux layer `N` as the network and a time step `dt` as the
time increment for the Euler method. The `N` layer should output the derivative
of the state given the current state and control input.

Assumes that only the first output is observable.
"""
struct EulerStepper{N <: AbstractLuxLayer} <: AbstractLuxWrapperLayer{:network}
    network::N
    dt::Float32
end

function (euler::EulerStepper)((x0, u), ps, st::NamedTuple)
    # x0 has shape (n_states, n_batch)
    # u has shape (n_controls, n_steps, n_batch)
    y = similar(u, 1, size(u, 3), size(u, 2))  # y has shape (1, n_batch, n_steps) - will permute later (for speed)
    @trace for i in 1:size(u, 2)
        dxdt, st = Lux.apply(euler.network, vcat(x0, u[:, i, :]), ps, st)
        x0 = x0 + euler.dt * dxdt
        y[1, :, i] = x0[1, :]  # Only the first output is observable
    end
    return permutedims(y, (1, 3, 2)), st
end

"""
    create_model(; dt, latent = 8, width = 64, rng = Random.default_rng())

Create a neural network model for the neural ODE, wrapped inside an
`EulerStepper`. The model consists of several dense layers with GELU activation
functions, followed by a layer normalization, and finally a dense layer that
outputs the predicted state and latent variables.

The `dt` parameter specifies the time step for the Euler method. The `latent`
parameter specifies the number of latent variables in the neural network, and
`width` specifies the width of the hidden layers. The `rng` parameter is used
for initializing the model parameters.
"""
function create_model(; dt, latent = 8, width = 64, rng = Random.default_rng())
    model = EulerStepper(
        Chain(
            Dense(2 + latent => width, gelu),
            Dense(width => width, gelu),
            Dense(width => width, gelu),
            LayerNorm(width),
            Dense(width => 1 + latent)
        ),
        dt
    )
    ps, st = Lux.setup(rng, model)
    return (; model, ps, st)
end

function create_scalings(data::Vector{Measurement}; batchsize = 32, latent = 8, diff_window = 4)
    # Find the linear term of a least squares fit through the first diff_window points of x;
    # approximates the derivative at t=0 (NOTE: should be scaled by 1/dt to get the correct scaling)
    least_sqrs = pinv([ones(diff_window);; collect(1:diff_window)])[2, :]
    # Create the dataset
    x = reshape(reduce(hcat, [[pt.rand_perturb.x[1]; dot(least_sqrs, pt.rand_perturb.x[1:diff_window]); zeros(latent - 1)] for pt in data]), (1 + latent, length(data)))
    u = reshape(reduce(hcat, [pt.rand_perturb.out[1:(end - 1)] for pt in data]), (1, :, length(data)))
    y = reshape(reduce(hcat, [pt.rand_perturb.x[2:end] for pt in data]), (1, :, length(data)))
    dydt = diff(y; dims = 2)
    return (x, u, y, dydt)
end

function create_dataloader_rand(data::Vector{Measurement}; batchsize = 1, latent = 8, diff_window = 4, step = 1)
    # This code effectively assumes that the time step between each data point
    # is 1 unit of time; since the data (including time derivatives) get
    # standardised, the original (physical) time step is irrelevant

    # x is the initial condition, which is taken as [x(0), dx/dt(0), zeros for remaining latent variables]
    # u is the external input to the system
    # y is the full data series (ground truth)

    # Find the linear term of a least squares fit through the first diff_window
    # points of x; approximates the derivative at t=0 (NOTE: should be scaled by
    # SAMPLE_FREQ to get the correct physical scaling)
    least_sqrs = pinv([ones(diff_window);; collect(1:diff_window)])[2, :]
    # Create the dataset
    x = convert(Array{Float32}, reshape(reduce(hcat, [[pt.rand_perturb.x[1]; dot(least_sqrs, pt.rand_perturb.x[1:diff_window]); zeros(latent - 1)] for pt in data]), (1 + latent, length(data))))
    u = convert(Array{Float32}, reshape(reduce(hcat, [pt.rand_perturb.out[1:step:(end - 1)] for pt in data]), (1, :, length(data))))
    y = convert(Array{Float32}, reshape(reduce(hcat, [pt.rand_perturb.x[2:step:end] for pt in data]), (1, :, length(data))))
    # Standardise the data
    x_scale = (μ = mean(y), σ = std(y))  # use y because it contains (almost) the entire time series
    u_scale = (μ = mean(u), σ = std(u))
    u .= (u .- u_scale.μ) ./ u_scale.σ
    y .= (y .- x_scale.μ) ./ x_scale.σ
    dt = std(diff(y; dims = 2))
    x[1, :, :] .= (x[1, :, :] .- x_scale.μ) ./ x_scale.σ
    x[2, :, :] .= x[2, :, :] ./ (x_scale.σ * dt)
    scalings = (; x_scale, u_scale, dt)
    dataset = (; x, u, y)
    # Return the dataloader
    return DataLoader(dataset; batchsize, partial = true, shuffle = true), scalings
end

function train_model(model, ps, st, dataloader; epochs = 100, print_every = 10, lr = 0.001f0, timespan = :, opt_state = nothing)
    train_state = Training.TrainState(model, ps, st, AdamW(lr))
    if opt_state !== nothing
        @set! train_state.optimizer_state = opt_state
    end
    for iteration in 1:epochs
        for (i, (x_i, u_i, y_i)) in enumerate(dataloader)
            _, loss, _, train_state = Training.single_train_step!(
                AutoEnzyme(), MSELoss(), ((x_i, u_i[:, timespan, :]), y_i[:, timespan, :]), train_state
            )
            if (iteration % print_every == 0 || iteration == 1) && i == 1
                @printf("Iter: [%4d/%4d]\tLoss: %.8f\n", iteration, epochs, loss)
            end
        end
    end
    return train_state
end

function dostuff(dataloader, dt; latent = 8, skip = 1, kwargs...)
    # Create the model
    rng = Random.Xoshiro(1234)
    (model, ps, st) = create_model(; dt, rng, latent, kwargs...)
    dataloader_ra = rdev(dataloader)
    # Training schedules
    schedules = [
        (100, 1:(200 ÷ skip), 0.001f0),
        (100, 1:(400 ÷ skip), 0.001f0),
        (100, 1:(600 ÷ skip), 0.001f0),
        (100, 1:(800 ÷ skip), 0.001f0),
        (100, 1:(1200 ÷ skip), 0.001f0),
        (200, 1:(2400 ÷ skip), 0.0005f0),
        (200, 1:(4800 ÷ skip), 0.0005f0),
        (400, :, 0.0005f0),
    ]
    # Train the model
    sc, remaining = Iterators.peel(schedules)
    println(sc)
    ts = train_model(model, rdev(ps), rdev(st), dataloader_ra; epochs = sc[1], timespan = sc[2], lr = sc[3])
    for sc in remaining
        println(sc)
        ts = train_model(ts.model, ts.parameters, ts.states, dataloader_ra; opt_state = ts.optimizer_state, epochs = sc[1], timespan = sc[2], lr = sc[3])
    end
    return ts
end
