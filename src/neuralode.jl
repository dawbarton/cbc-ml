using Lux: Lux, LuxCore, Training, AutoEnzyme, MSELoss, Chain, Dense, LayerNorm, AbstractLuxLayer, AbstractLuxWrapperLayer, gelu
using Random: Random
using Optimisers: Optimisers, Adam
using Reactant: Reactant, @trace, Const
using Enzyme: Enzyme
using MLUtils: MLUtils, DataLoader
using Printf: @printf
using LinearAlgebra: LinearAlgebra, pinv, dot

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

function create_dataloader_rand(data::Vector{Measurement}; batchsize = 32, latent = 8, diff_window = 4)
    # Find the linear term of a least squares fit through the first diff_window points of x;
    # approximates the derivative at t=0 (NOTE: should be scaled by 1/dt to get the correct scaling)
    least_sqrs = pinv([ones(diff_window);; collect(1:diff_window)])[2, :]
    # Create the dataset
    x = reshape(reduce(hcat, [[pt.rand_perturb.x[1]; dot(least_sqrs, pt.rand_perturb.x[1:diff_window]); zeros(latent - 1)] for pt in data]), (1 + latent, length(data)))
    u = reshape(reduce(hcat, [pt.rand_perturb.out[1:(end - 1)] for pt in data]), (1, :, length(data)))
    y = reshape(reduce(hcat, [pt.rand_perturb.x[2:end] for pt in data]), (1, :, length(data)))
    dataset = (; x, u, y)
    # Return the dataloader
    return DataLoader(dataset; batchsize, partial = true, shuffle = true)
end

function train_model(model, ps, st, dataloader; epochs = 100, print_every = 10, lr = 0.001f0)
    train_state = Training.TrainState(model, ps, st, Adam(lr))
    for iteration in 1:epochs
        for (i, (x_i, u_i, y_i)) in enumerate(dataloader)
            _, loss, _, train_state = Training.single_train_step!(
                AutoEnzyme(), MSELoss(), ((x_i, u_i), y_i), train_state
            )
            if (iteration % print_every == 0 || iteration == 1) && i == 1
                @printf("Iter: [%4d/%4d]\tLoss: %.8f\n", iteration, epochs, loss)
            end
        end
    end
    return train_state
end

function dostuff(dataloader; latent = 8, kwargs...)
    # Create the model
    rng = Random.default_rng(1234)
    (model, ps, st) = create_model(; dt = Float32(1 / SAMPLE_FREQ), rng, latent)
    # Train the model
    return train_model(model, rdev(ps), rdev(st), rdev(dataloader); kwargs...)
end
