import pathlib
import scipy.io as sio

BASE_PATH = pathlib.Path(__file__).parent.parent.absolute()
DATA_PATH = BASE_PATH / 'data'
MSSP_DATA_PATH = DATA_PATH / '2015_MSSP_Barton'

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

def load_random_varyrand():
    return sio.loadmat(MSSP_DATA_PATH / 'ex20150603_3_a.mat', squeeze_me=True)['ex']['data']
    # Additional fixed parameters (e.g., sample frequency) are stored in ex.par
