import random
import itertools
import enum

import numpy as np
import dataclasses
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
import pandas

# increase font size
plt.rcParams.update({"font.size": 22})


class ExperimentType(enum.Enum):
    CHAOTIC = enum.auto()
    DOPPLER = enum.auto()


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


def initialize_sun_data(
    total_size, sun_size, sun_center=500, experiment_type=ExperimentType.CHAOTIC
):
    sun_data = [0] * total_size
    for i in range(sun_center - sun_size // 2, sun_center + sun_size // 2 + 1):
        if experiment_type == ExperimentType.CHAOTIC:
            sun_data[i] = np.exp(random.random() * 2 * np.pi * 1j)
        else:
            sun_data[i] = np.exp(
                -(1 / 2) * (((i - sun_center) / (sun_size // 2)) ** 2)
            ) * np.exp(random.random() * 2 * np.pi * 1j)

    return np.array(sun_data)


def get_field(total_size, sun_size, experiment_type=ExperimentType.CHAOTIC):
    sun_data = initialize_sun_data(
        total_size, sun_size, experiment_type=experiment_type
    )
    fft_sun_data = np.fft.fft(sun_data)
    return Sun(vector=sun_data, field=fft_sun_data)


def make_experiment(
    stationary, total_size, sun_size, average=1, experiment_type=ExperimentType.CHAOTIC
):
    field_intensity = np.mean(
        [
            get_field(total_size, sun_size, experiment_type).intensity
            for _ in range(average)
        ],
        axis=0,
    )
    return Experiment(
        intensity_product=field_intensity * field_intensity[stationary],
        stationary_intensity=field_intensity[stationary],
    )


def get_second_coherence_function(
    total_size, sun_size, number_of_steps=10**5, is_smeared=False
):
    experiments = [
        make_experiment(
            500,
            total_size=total_size,
            sun_size=sun_size,
            average=1 if not is_smeared else 10,
        )
        for _ in range(number_of_steps)
    ]

    return (
        np.mean([experiment.intensity_product for experiment in experiments], axis=0)
        / np.mean([experiment.stationary_intensity for experiment in experiments]) ** 2
    )


def get_doppler_coherence_function(total_size, sun_size, number_of_steps=10**5):
    experiments = [
        make_experiment(
            500,
            total_size=total_size,
            sun_size=sun_size,
            average=1,
            experiment_type=ExperimentType.DOPPLER,
        )
        for _ in range(number_of_steps)
    ]

    return (
        np.mean([experiment.intensity_product for experiment in experiments], axis=0)
        / np.mean([experiment.stationary_intensity for experiment in experiments]) ** 2
    )


def plot_sun_fields(experiment_type=ExperimentType.CHAOTIC):
    suns = [get_field(1024, 40, experiment_type) for i in range(2)]
    for i, sun in enumerate(suns):
        plt.plot(sun.intensity, label=f"Sun {i+1}")

    plt.xlabel("Cell number")
    plt.ylabel("Intensity")
    plt.legend()
    plt.show()

    ax = plt.axes(projection="3d")
    # improve spacing between text and numbers
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
    suns = [get_field(1024, 40) for _ in range(10000)]
    plt.plot(np.mean([sun.intensity for sun in suns], axis=0))
    plt.xlabel("Cell number")
    plt.ylabel("Intensity")
    plt.show()


def plot_numeric_second_coherence(experiment_type=ExperimentType.CHAOTIC):
    if experiment_type == ExperimentType.CHAOTIC:
        second_coherence = get_second_coherence_function(total_size=1024, sun_size=40)

        def fit(x, t_c):
            return (np.max(abs(second_coherence)) - 1) + np.exp(-(2 * np.abs(x)) / t_c)

    elif experiment_type == ExperimentType.DOPPLER:
        second_coherence = get_doppler_coherence_function(
            total_size=1024, sun_size=40, number_of_steps=10**6
        )

        def fit(x, t_c):
            return (np.max(abs(second_coherence)) - 1) + np.exp(-(2 * (x / t_c) ** 2))

    else:
        raise ValueError("Invalid experiment type")

    x_points = np.array(range(1024)) - 500

    result = curve_fit(fit, x_points, np.abs(second_coherence))

    plt.plot(x_points, np.abs(second_coherence))
    plt.plot(
        x_points,
        [fit(x, result[0][0]) for x in x_points],
        label=rf"Fit, $\tau_c$ = {result[0][0]:.2f}",
    )
    # plt.axvline(x_points=-20, color="black", linestyle="--")
    # plt.axvline(x_points=21, color="black", linestyle="--")
    plt.xlabel("Distance from the stationary point (sun center)")
    plt.ylabel("Second order coherence function")
    plt.legend()
    plt.show()


def plot_smeared_second_coherence():
    second_coherence = get_second_coherence_function(
        total_size=1024, sun_size=40, is_smeared=True
    )
    x_points = np.array(range(1024)) - 500

    plt.plot(x_points, np.abs(second_coherence))
    plt.xlabel("Distance from the stationary point (sun center)")
    plt.ylabel("Second order coherence function")
    plt.show()


def plot_intensity_histogram(number_of_fields=10**4):
    suns = [get_field(1024, 40) for _ in range(number_of_fields)]
    histogram = plt.hist(
        list(itertools.chain.from_iterable([sun.intensity for sun in suns])),
        bins=20,
    )

    def fit(x, P_0, I_0):
        return P_0 * np.exp(-(x / I_0))

    result = curve_fit(fit, histogram[1][:-1], histogram[0])

    plt.plot(
        histogram[1][:-1],
        [fit(x, result[0][0], result[0][1]) for x in histogram[1][:-1]],
        label=rf"Fit $I_0$ = {result[0][1]:.2f}",
    )
    plt.legend()
    plt.xlabel("Intensity")
    plt.ylabel("Number of cells")
    plt.show()


def find_fwhm():
    sun_size_to_second_coherence = {}
    for sun_size in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
        sun_size_to_second_coherence[sun_size] = get_second_coherence_function(
            total_size=1024, sun_size=sun_size
        )

    for sun_size, second_coherence in sun_size_to_second_coherence.items():
        x_points = np.array(range(1024)) - 500
        plt.plot(
            x_points,
            second_coherence,
        )
        plt.axhline(
            y=0.5 * np.max(abs(second_coherence)), color="black", linestyle="--"
        )
        plt.title(f"Sun size = {sun_size}")
        plt.show()


def plot_fwhm_from_file():
    csv = pandas.read_csv("data/fwhm.csv")
    plt.plot(csv["size"], csv["FWHM"])
    plt.xlabel("Sun size")
    plt.ylabel("FWHM")
    plt.show()


def main():
    plot_numeric_second_coherence(experiment_type=ExperimentType.DOPPLER)


if __name__ == "__main__":
    main()
