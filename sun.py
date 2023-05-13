import numpy as np
import random
from matplotlib import pyplot as plt


def initialize_sun_data():
    sun_data = [0] * 1024
    for i in range(480, 521):
        sun_data[i] = np.exp(random.random() * 2 * np.pi * 1j)

    return sun_data


def main():
    sun_data = initialize_sun_data()
    plt.plot(sun_data)
    plt.show()


if __name__ == "__main__":
    main()
