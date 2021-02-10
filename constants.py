

SAMPLE_RATE = 44100

# FFT Window Size
FFT_WINDOW_SIZE_SAMPS = 44100
FFT_WINDOW_SIZE_T = FFT_WINDOW_SIZE_SAMPS / SAMPLE_RATE

# FFT LIMITATIONS
FFT_MAX_FREQUENCY = FFT_WINDOW_SIZE_SAMPS // 2


# IMPULSE DETECTION

IMPULSE_KERNEL_W_T = 1 / 10         # To detect impulses w/ frequencies down to 20Hz. This provides for 2 periods of said frequency
IMPULSE_KERNEL_STEP_SIZE = 1 / 10    # How far should we move the impulse kernel when checking for impulses.
IMPULSE_POWER_MULT_THRESHOLD = 500  # Required Power multiplier between kernels to be considered an impulse.

# Audio Buffer Step Size for Recall
AUDIO_BUFFER_SAMPLES_PER_CALLBACK = 20000

