from scipy.io import wavfile
from scipy.signal import  resample

import numpy as np
import config


def mackey_glass(
    total_steps=1000,
    delay=17
):

    beta = 0.2
    gamma = 0.1
    exponent = 10
    time_step = 1

    data = np.zeros(total_steps)

    data[0] = 1.2

    for t in range(total_steps - 1):

        if t >= delay:
            delayed_value = data[t - delay]
        else:
            delayed_value = 0

        growth = (
            beta
            * delayed_value
            / (
                1
                + delayed_value ** exponent
            )
        )

        decay = gamma * data[t]

        data[t + 1] = (
            data[t]
            + (growth - decay) * time_step
        )

    return data

def load_audio(path):

    sample_rate, data = wavfile.read(path)

    # Convert stereo to mono
    if data.ndim > 1:
        data = data.mean(axis=1)

    # Convert to float
    data = data.astype(float)

    # Normalize
    max_value = np.max(np.abs(data))

    if max_value > 0:
        data /= max_value

    print("Original samples:", len(data))

    # RMS amplitude envelope
    num_points = config.MAX_SEQUENCE_LENGTH
    window_size = len(data) // num_points

    envelope = []

    for i in range(num_points):

        start = i * window_size
        end = (i + 1) * window_size

        window = data[start:end]

        envelope.append(
            np.sqrt(np.mean(window ** 2))
        )

    data = np.array(envelope)

    print("Min:", np.min(data))
    print("Max:", np.max(data))
    print("Mean:", np.mean(data))
    print("Std:", np.std(data))

    return data

def resize_time_series(data):

    if len(data) != config.MAX_SEQUENCE_LENGTH:

        data = resample(
            data,
            config.MAX_SEQUENCE_LENGTH
        )

    return data


def load_time_series():

    if config.TIME_SERIES == "mackey":

        data = mackey_glass(
            total_steps=config.MAX_SEQUENCE_LENGTH
        )

    elif config.TIME_SERIES == "audio_medieval":

        data = load_audio(
            "audioData/audio_medieval.wav"
        )

    else:

        raise ValueError(
            f"Unknown TIME_SERIES: "
            f"{config.TIME_SERIES}"
        )

    data = resize_time_series(data)

    print(
        "Final time series length:",
        len(data)
    )

    return data