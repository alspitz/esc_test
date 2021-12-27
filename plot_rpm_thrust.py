import argparse
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    rpm_thrust_coeffs = [
      ("kotleta2", [ 6.41527685e-06, -2.17608417e-02,  3.97188659e+01], "F60"),
      ("kotleta3", [ 6.40120200e-06 ,-2.07110888e-02 , 3.45876910e+01], "F60"),
    ]
    plt.figure("RPM-thrust curves")
    plt.title("RPM-thrust curves", fontsize=16)

    rpm_range = np.arange(0, 18001, 100)

    for coeffs in rpm_thrust_coeffs:
        label = coeffs[0]

        thrust_range = [coeffs[1][2] + coeffs[1][1] * rpm + coeffs[1][0] * rpm ** 2 for rpm in rpm_range]
        plot_style = "-"
        if coeffs[2] == "F80":
            plot_style = "x-"
        plt.plot(rpm_range, thrust_range, plot_style, markersize=3, label=label)

    plt.xlabel("RPM")
    plt.ylabel("Thrust")
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.legend()
    plt.show()
