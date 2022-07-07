

SAMPLE_RATE = 44100

# FFT Window Size
FFT_WINDOW_SIZE_SAMPS = 44100
FFT_WINDOW_SIZE_T = FFT_WINDOW_SIZE_SAMPS / SAMPLE_RATE

# FFT LIMITATIONS
FFT_MAX_FREQUENCY = FFT_WINDOW_SIZE_SAMPS // 2


# IMPULSE DETECTION

IMPULSE_KERNEL_W_T = 1 / 6         # To detect impulses w/ frequencies down to 20Hz. This (1/10)provides for 2 periods of said frequency
IMPULSE_KERNEL_STEP_SIZE = 1 / 6    # How far should we move the impulse kernel when checking for impulses.
IMPULSE_POWER_MULT_THRESHOLD = 70  # Required Power multiplier between kernels to be considered an impulse.
IMPULSE_POWER_DELTA_THRESHOLD = 0.05

# Audio Buffer Step Size for Recall
AUDIO_BUFFER_SAMPLES_PER_CALLBACK = 20000

