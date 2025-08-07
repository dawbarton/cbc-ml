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
    (model, ps, st) = create_model(rng = rng)
    n_steps = 13
    n_batch = 7
    x0 = Float32[-1.3832767 0.84985137 0.30427745 -0.044958733 -1.0677148 -1.5736556 1.2947402; -0.5162747 1.2465676 0.91887575 -0.40274197 -0.69026494 -0.124063306 0.5012879; -0.5455988 0.766296 -1.556192 -0.5312108 0.84583026 -1.9013535 -0.6548357; -1.3655096 1.5564576 1.6106234 0.5456047 -0.6347799 0.16984817 0.64862674; -3.014762 -2.504384 -0.7385977 0.29583368 -0.25673077 0.76431495 0.50688916; 0.78367597 -0.44305003 0.052315846 0.5057889 0.47063485 -0.8441043 1.1308262; 0.3242214 -0.4197961 -0.42178145 -0.17482775 1.6706173 1.5905248 -0.66165984; 0.10839289 -0.49617288 0.16236115 1.1391029 -0.45036948 0.1532558 -1.0371859; -1.0341535 0.2268608 -1.2679945 2.1738687 0.7698496 0.17265159 1.4181478] #randn(Float32, 9, n_batch)  # Initial state
    u = Float32[-1.1284686 0.7900919 -0.498859 0.22539303 -0.08533491 -0.52355987 -0.10178299; 1.5222852 -1.63973 0.03105152 0.97092587 -1.2178774 2.2799983 -0.34777302; 1.4705609 -0.008318167 1.1836609 0.79317 0.9692262 -2.0446692 -1.3079661; -0.4861697 -0.38908297 0.084916346 -0.36023903 0.23808777 1.4487302 -0.6775805; -1.5847149 -0.3450839 1.395907 0.33435023 0.7200655 0.08511576 0.7374191; 1.5771464 -0.4130214 0.54762304 -0.6991527 0.80424744 0.2164369 -1.3903517; 0.05446302 0.4077639 0.8282895 0.27634516 1.2738134 0.9801856 0.84951836; 0.4809817 -0.48814592 -1.0852027 1.002704 -0.3233871 1.5823575 -1.5423214; -0.32434502 0.5964641 0.24430856 -2.2393014 -0.33337092 0.36953163 0.21674192; -1.5551085 -0.11481965 0.41273212 -0.29578778 -0.79082835 0.01522871 0.29623997; -0.58503103 -0.5414124 -1.1277186 0.36162487 -2.660261 -0.19937904 -0.6502729; -1.078781 -0.02789837 1.350495 -0.397268 0.42840433 -0.550001 1.3907036; -0.41275057 -0.17665163 -0.05020652 -0.08793639 -0.21427597 1.697888 -1.7665641] #randn(Float32, n_steps, n_batch)  # Control input
    y = Float32[-1.8385237 0.5508405 0.32711846 0.78230643 1.9933405 1.5920227 -0.32292274; 1.3132579 -0.8423196 0.46991065 0.7074946 -0.14377636 -0.21102488 1.7059916; -0.43332806 0.11185128 -1.4896264 -0.40678978 -0.7435633 -0.96424913 -0.21775572; -1.7125106 -0.30497563 -0.5589103 0.7827464 -0.73202163 -1.4562876 -1.4190221; -0.8848837 0.5591459 0.7149639 -0.20375594 0.06458849 -1.526316 -0.7291689; -0.022509279 0.29973745 -0.7251164 -0.6326398 -0.6878459 0.94454354 0.116147205; -1.3888663 0.94991094 1.1144314 -0.87771404 -1.1587987 -0.37586585 0.49395272; -0.22658202 0.26913032 0.47758487 0.9705532 -0.5937554 0.119936295 -0.19108693; -0.67235273 1.2556098 -1.1867012 0.062171947 -0.905421 -1.941201 -0.99565786; 0.39366028 1.2393461 0.25356922 -1.4501189 0.71220404 0.001304453 0.64645386; 0.3913119 -0.70874846 -0.9805249 1.9651573 -0.60491323 -0.6364309 -1.1517092; 1.0968064 -0.0610148 0.86135745 0.4865784 1.1943206 2.164689 0.5325889; -0.75665003 1.4519044 0.28592956 0.42501855 1.2815912 0.15568873 -1.6439742] #randn(Float32, n_steps, n_batch)  # Target output
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
    println("Reactant gradient: ", cdev(grad))
    return loss_value_ra, loss_value
end
