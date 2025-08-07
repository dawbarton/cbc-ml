using GLMakie: Figure, Axis3, scatter!
using SavitzkyGolay: savitzky_golay

# # Notes
#
# When visualising with a scatter plot, there is often a white circle around
# each marker. These are artifacts of the rendering. Various solutions can be
# found at
# <https://docs.makie.org/stable/reference/plots/scatter.html#Dealing-with-outline-artifacts-in-GLMakie>.
# Setting `fxaa = true` in the `scatter!` function seems to work well, but it is
# not always the case. Sometimes the artifacts can be helpful in visualising the
# data because they give more structure to the plot.

DEFAULT_FIGURE_SIZE::Tuple{Int, Int} = (800, 600)

"""
    plot_forcing(data; size = DEFAULT_FIGURE_SIZE)

Plot the response surface of the data provided by plotting the triple (forcing
frequency, forcing amplitude, response amplitude).
"""
function plot_surface(data; size = DEFAULT_FIGURE_SIZE)
    forcing_freq = [pt.forcing_freq for run in data for pt in run]
    forcing_amp = [pt.out_amp for run in data for pt in run]
    response_amp = [pt.x_amp for run in data for pt in run]
    fig = Figure(; size)
    ax = Axis3(fig[1, 1], title = "Surface Plot", xlabel = "Frequency (Hz)", ylabel = "Forcing Amplitude (V)", zlabel = "Response Amplitude (mm)")
    scatter!(ax, forcing_freq, forcing_amp, response_amp)
    return fig
end

"""
    plot_manifold_dxdt(data; window = 21, rand = false, base = false)

Plot the manifold that all the data lies on; plots `x(t)` vs `dx/dt` vs `u(t)`,
where `dx/dt` is the derivative of `x(t)`. The derivative is calculated using a
Savitzky-Golay filter with a specified window size.

The `rand` argument specifies whether to use the random perturbation data, and
`base` specifies whether to use the base data instead of the output data. The
`window` argument specifies the size of the Savitzky-Golay filter window.
"""
function plot_manifold_dxdt(data; window = 21, rand = false, base = false, size = DEFAULT_FIGURE_SIZE)
    x = Float64[]
    dxdt = Float64[]
    u = Float64[]
    if data isa Vector{Measurement}
        data = [data]
    end
    half_window = (window - 1) ÷ 2
    for run in data
        for pt in run
            if rand
                x_pt = pt.rand_perturb.x
                u_pt = base ? pt.rand_perturb.base : pt.rand_perturb.out
            else
                x_pt = pt.x
                u_pt = base ? pt.base : pt.out
            end
            append!(x, @view x_pt[(half_window + 1):(end - half_window)])
            dxdt_pt = savitzky_golay(x_pt, window, 3; deriv = 1, rate = SAMPLE_FREQ)
            append!(dxdt, @view dxdt_pt.y[(half_window + 1):(end - half_window)])
            append!(u, @view u_pt[(half_window + 1):(end - half_window)])
        end
    end
    @show length(x)
    fig = Figure(; size)
    ax = Axis3(fig[1, 1], title = "Manifold Plot", xlabel = "x", ylabel = "dx/dt", zlabel = "u")
    scatter!(ax, x, dxdt, u; fxaa = true, alpha = 0.1)
    return fig
end

"""
    plot_manifold_delay(data; delay = 75, rand = false, base = false)

Plot the manifold that all the data lies on; plots `x(t)` vs `x(t-τ)` vs `u(t)`,
where `τ` is the delay.

The `rand` argument specifies whether to use the random perturbation data, and
`base` specifies whether to use the base data instead of the output data. The
`window` argument specifies the size of the Savitzky-Golay filter window.
"""
function plot_manifold_delay(data; delay = 75, rand = false, base = false, size = DEFAULT_FIGURE_SIZE)
    x = Float64[]
    xtau = Float64[]
    u = Float64[]
    if data isa Vector{Measurement}
        data = [data]
    end
    for run in data
        for pt in run
            if rand
                x_pt = pt.rand_perturb.x
                u_pt = base ? pt.rand_perturb.base : pt.rand_perturb.out
            else
                x_pt = pt.x
                u_pt = base ? pt.base : pt.out
            end
            append!(x, @view x_pt[(begin + delay):end])
            append!(xtau, @view x_pt[begin:(end - delay)])
            append!(u, @view u_pt[(begin + delay):end])
        end
    end
    @show length(x)
    fig = Figure(; size)
    ax = Axis3(fig[1, 1], title = "Manifold Plot", xlabel = "x(t)", ylabel = "x(t-τ)", zlabel = "u(t)")
    scatter!(ax, x, xtau, u; fxaa = true, alpha = 0.1)
    return fig
end
