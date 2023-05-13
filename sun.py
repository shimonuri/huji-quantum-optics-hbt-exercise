import numpy as np
import random
import dataclasses
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit

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


def plot_sun_fields():
    suns = [get_field() for i in range(2)]
    for i, sun in enumerate(suns):
        plt.plot(sun.intensity, label=f"Sun {i+1}")

    plt.xlabel("Cell number")
    plt.ylabel("Intensity")
    plt.legend()
    plt.show()

    ax = plt.axes(projection="3d")
    # inprove spacing between text and numbers
    ax.xaxis.labelpad = 25
    ax.yaxis.labelpad = 25
    ax.zaxis.labelpad = 25
    for i, sun in enumerate(suns):
        # plot  complex vector
        ax.scatter(
            range(1024),
            np.real(sun.vector),
            np.imag(sun.vector),
            label=f"Sun {i + 1}",
        )
    # plot 3d unit circle
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x + 500, y, z, color="black", alpha=0.1)
    ax.set_xlabel("Cell Number")
    ax.set_ylabel("Real Part")
    ax.set_zlabel("Imaginary Part")
    ax.legend()
    plt.show()


def plot_average_sun_field():
    suns = [get_field() for _ in range(10000)]
    plt.plot(np.mean([sun.intensity for sun in suns], axis=0))
    plt.xlabel("Cell number")
    plt.ylabel("Intensity")
    plt.show()


def plot_numeric_second_coherence():
    second_coherence = get_second_coherence_function()
    x_points = np.array(range(1024)) - 500

    def fit(x, t_c):
        return 1 + np.exp(-(2 * np.abs(x)) / t_c)

    result = curve_fit(fit, x_points, np.abs(second_coherence))

    plt.plot(x_points, np.abs(second_coherence))
    plt.plot(
        x_points,
        [fit(x, result[0][0]) for x in x_points],
        label=fr"Fit, $\tau_c$ = {result[0][0]:.2f}",
    )
    # plt.axvline(x_points=-20, color="black", linestyle="--")
    # plt.axvline(x_points=21, color="black", linestyle="--")
    plt.xlabel("Distance from the stationary point (sun center)")
    plt.ylabel("Second order coherence function")
    plt.legend()
    plt.show()


def plot_intensity_histogram():
    suns = [get_field() for _ in range(10000)]
    plt.hist([sun.intensity for sun in suns])
    plt.xlabel("Intensity")
    plt.ylabel("Frequency")
    plt.show()


def main():
    plot_numeric_second_coherence()


if __name__ == "__main__":
    main()
