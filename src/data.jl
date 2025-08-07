using MAT: matread

BASE_PATH = joinpath(@__DIR__, "..")
DATA_PATH = joinpath(BASE_PATH, "data")
MSSP_DATA_PATH = joinpath(DATA_PATH, "2015_MSSP_Barton")

# Data from the paper:
#     David A.W. Barton,
#     Control-based continuation: Bifurcation and stability analysis for physical experiments,
#     Mechanical Systems and Signal Processing,
#     Volume 84, Part B,
#     2017,
#     Pages 54-64,
#     ISSN 0888-3270,
#     https://doi.org/10.1016/j.ymssp.2015.12.039.
#
# Data available from https://doi.org/10.5523/bris.1esk6h8klq0bg1gdcr78a2ypyz
#
# Fields in the data structure:
#     forcing_freq: frequency of the forcing signal (Hz) (ideal - actual may vary due to 32bit floating point errors)
#     forcing_coeffs: Fourier coefficients of the forcing signal (V) (sine and cosine terms)
#     rand_amp: requested amplitude of the random perturbation signal (V)
#     x_coeffs_ave: average Fourier coefficients of the relative displacement of the moving mass (mm) (sine and cosine terms)
#     x_coeffs_var: variance of the Fourier coefficients of the relative displacement of the moving mass (mm) (sine and cosine terms)
#     x_target_coeffs: Fourier coefficients of the control target (control reference) (mm) (sine and cosine terms)
#     out_coeffs_ave: average Fourier coefficients of the signal sent to the actuator (V) (sine and cosine terms)
#     out_coeffs_var: variance of the Fourier coefficients of the signal sent to the actuator (V) (sine and cosine terms)
#     time_mod_2pi: point in the forcing cycle (0..2pi)
#     x: timeseries of the relative displacement of the moving mass (mm)
#     x_target: timeseries of the control target (control reference) (mm)
#     base: timeseries of the base displacement (mm)
#     mass: timeseries of the absolute displacement of the moving mass (mm)
#     force: timeseries of the force applied to the base (N) (unreliable?)
#     out: timeseries of the signal sent to the actuator (V) (forcing + control + random perturbation)
#     rand_out: timeseries of the random perturbation signal sent to the actuator (V)
#     rand_perturb: measurements taken with a random perturbation signal applied
#     timestamp: timestamp that the data was recorded (MATLAB format)
#     x_amp: estimate of the amplitude of the relative displacement of the moving mass (mm)
#     out_amp: estimate of the amplitude of the signal sent to the actuator (V)
#
# Sample frequency is 1kHz

const SAMPLE_FREQ = 1000.0  # Hz

struct RandomMeasurement
    base::Vector{Float64}
    rand_out::Vector{Float64}
    x_target::Vector{Float64}
    out::Vector{Float64}
    x::Vector{Float64}
    mass::Vector{Float64}
    force::Vector{Float64}
    time_mod_2pi::Vector{Float64}
    rand_amp::Float64
end

struct Measurement
    forcing_freq::Float64
    forcing_coeffs::Vector{Float64}
    rand_amp::Float64
    x_coeffs_ave::Vector{Float64}
    x_coeffs_var::Vector{Float64}
    x_target_coeffs::Vector{Float64}
    out_coeffs_ave::Vector{Float64}
    out_coeffs_var::Vector{Float64}
    time_mod_2pi::Vector{Float64}
    x::Vector{Float64}
    x_target::Vector{Float64}
    base::Vector{Float64}
    mass::Vector{Float64}
    force::Vector{Float64}
    out::Vector{Float64}
    rand_out::Vector{Float64}
    rand_perturb::Union{RandomMeasurement, Nothing}
    timestamp::Vector{Float64}
    x_amp::Float64
    out_amp::Float64
end

function Base.show(io::IO, pt::Union{Measurement, RandomMeasurement})
    if get(io, :compact, true)::Bool
        print(io, "$(typeof(pt))(…)")
    else
        invoke(Base.show, Tuple{IO, Any}, io, pt)
    end
end

function Base.show(io::IO, ::MIME"text/plain", pt::Union{Measurement, RandomMeasurement})
    print(io, "$(typeof(pt))(\n")
    for name in fieldnames(typeof(pt))
        print(io, "  $name: ")
        show(io, getfield(pt, name))
        print(io, "\n")
    end
    print(io, ")")
end

function load_from(rawdata)
    runs = Vector{Vector{Measurement}}()
    for run_data in rawdata
        run = Vector{Measurement}()
        for i in eachindex(run_data["timestamp"])
            if haskey(run_data, "rand_perturb")
                rand_perturb = RandomMeasurement(
                    vec(run_data["rand_perturb"][i]["base"]),
                    vec(run_data["rand_perturb"][i]["rand_out"]),
                    vec(run_data["rand_perturb"][i]["x_target"]),
                    vec(run_data["rand_perturb"][i]["out"]),
                    vec(run_data["rand_perturb"][i]["x"]),
                    vec(run_data["rand_perturb"][i]["mass"]),
                    vec(run_data["rand_perturb"][i]["force"]),
                    vec(run_data["rand_perturb"][i]["time_mod_2pi"]),
                    run_data["rand_perturb"][i]["rand_amp"]
                )
            else
                rand_perturb = nothing
            end
            measurement = Measurement(
                run_data["forcing_freq"][i],
                vec(run_data["forcing_coeffs"][i]),
                run_data["rand_amp"][i],
                vec(run_data["x_coeffs_ave"][i]),
                vec(run_data["x_coeffs_var"][i]),
                vec(run_data["x_target_coeffs"][i]),
                vec(run_data["out_coeffs_ave"][i]),
                vec(run_data["out_coeffs_var"][i]),
                vec(run_data["time_mod_2pi"][i]),
                vec(run_data["x"][i]),
                vec(run_data["x_target"][i]),
                vec(run_data["base"][i]),
                vec(run_data["mass"][i]),
                vec(run_data["force"][i]),
                vec(run_data["out"][i]),
                vec(run_data["rand_out"][i]),
                rand_perturb,
                vec(run_data["timestamp"][i]),
                run_data["x_amp"][i],
                run_data["out_amp"][i]
            )
            push!(run, measurement)
        end
        push!(runs, run)
    end
    return runs
end

function load_random_varyrand()
    rawdata = matread(joinpath(MSSP_DATA_PATH, "ex20150603_3_a.mat"))["ex"]["data"]
    return load_from(rawdata)
end

function load_random_replicates()
    rawdata = matread(joinpath(MSSP_DATA_PATH, "ex20150603_3_b.mat"))["ex"]["data"]
    return load_from(rawdata)
end

function load_surface()
    rawdata = matread(joinpath(MSSP_DATA_PATH, "ex20150603_4.mat"))["ex"]["data"]
    return load_from(rawdata)
end
