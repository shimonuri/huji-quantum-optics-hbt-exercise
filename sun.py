import numpy as np
import random
import dataclasses
from matplotlib import pyplot as plt

# increase font size
plt.rcParams.update({"font.size": 22})


@dataclasses.dataclass
class Sun:
    vector: np.ndarray
    field: np.ndarray

    @property
    def intensity(self):
        return np.abs(self.field) ** 2


@dataclasses.dataclass
class Experiment:
    intensity_product: np.ndarray
    stationary_intensity: float


def initialize_sun_data():
    sun_data = [0] * 1024
    for i in range(480, 521):
        sun_data[i] = np.exp(random.random() * 2 * np.pi * 1j)

    return np.array(sun_data)


def get_field():
    sun_data = initialize_sun_data()
    fft_sun_data = np.fft.fft(sun_data)
    return Sun(vector=sun_data, field=fft_sun_data)


def make_experiment(stationary):
    field = get_field()
    return Experiment(
        intensity_product=field.intensity * field.intensity[stationary],
        stationary_intensity=field.intensity[stationary],
    )


def get_second_coherence_function():
    experiments = [make_experiment(500) for i in range(10**5)]
    return (
        np.mean([experiment.intensity_product for experiment in experiments], axis=0)
        / np.mean([experiment.stationary_intensity for experiment in experiments]) ** 2
    )


def main():
    plot_sun_fields()


def plot_sun_fields():
    suns = [get_field() for i in range(3)]
    for i, sun in enumerate(suns):
        plt.plot(sun.intensity, label=f"Sun {i+1}")

    plt.xlabel("Cell number")
    plt.ylabel("Intensity")
    plt.legend()
    plt.show()


def plot_average_sun_field():
    suns = [get_field() for i in range(10**5)]
    plt.plot(np.mean([sun.intensity for sun in suns], axis=0))
    plt.xlabel("Cell number")
    plt.ylabel("Intensity")
    plt.show()


def plot_numeric_second_coherence():
    second_coherence = get_second_coherence_function()
    plt.plot(np.array(range(1024)) - 500, np.abs(second_coherence))
    plt.axvline(x=-20, color="black", linestyle="--")
    plt.axvline(x=21, color="black", linestyle="--")
    plt.xlabel("Distance from the stationary point (sun center)")
    plt.ylabel("Second order coherence function")
    plt.show()


if __name__ == "__main__":
    main()
