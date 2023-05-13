import numpy as np
import random
import dataclasses
from matplotlib import pyplot as plt


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
    experiments = [make_experiment(500) for i in range(10 ** 5)]
    return (
        np.mean([experiment.intensity_product for experiment in experiments], axis=0)
        / np.mean([experiment.stationary_intensity for experiment in experiments]) ** 2
    )


def main():
    plt.plot(np.array(range(1024)) - 500, np.abs(get_second_coherence_function()))
    plt.axvline(x=-20, color="black", linestyle="--")
    plt.axvline(x=21, color="black", linestyle="--")
    plt.show()


if __name__ == "__main__":
    main()
