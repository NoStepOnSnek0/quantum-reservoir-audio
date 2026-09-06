import Gate
import Spin
from config import SEEDS

import numpy as np
import matplotlib.pyplot as plt

from utils import print_results, print_evaluation


print("Running Gate...")

gate_results = []

for seed in SEEDS:
    gate_results.append(
        Gate.run(seed)
    )


print("\nRunning Spin...")

spin_results = []

for seed in SEEDS:
    spin_results.append(
        Spin.run(seed)
    )


# Individual seed metrics
gate_nrmse = [
    result[2]
    for result in gate_results
]

gate_r2 = [
    result[3]
    for result in gate_results
]

spin_nrmse = [  
    result[2]
    for result in spin_results
]

spin_r2 = [
    result[3]
    for result in spin_results
]


print("\n========== FINAL RESULTS ==========")

print(
    f"Gate:"
    f"\nNRMSE = {np.mean(gate_nrmse):.4f} ± {np.std(gate_nrmse, ddof=1):.4f}"
    f"\nR²    = {np.mean(gate_r2):.4f} ± {np.std(gate_r2, ddof=1):.4f}"
)

print(
    f"\nSpin:"
    f"\nNRMSE = {np.mean(spin_nrmse):.4f} ± {np.std(spin_nrmse, ddof=1):.4f}"
    f"\nR²    = {np.mean(spin_r2):.4f} ± {np.std(spin_r2, ddof=1):.4f}"
)


# Average predictions across seeds
gate_mean_predictions = np.mean(
    [result[1] for result in gate_results],
    axis=0
)

spin_mean_predictions = np.mean(
    [result[1] for result in spin_results],
    axis=0
)


# Calculate metrics from averaged predictions
gate_mean_nrmse, gate_mean_r2 = print_evaluation(
    gate_results[0][0],
    gate_mean_predictions,
    "Gate Average"
)

spin_mean_nrmse, spin_mean_r2 = print_evaluation(
    spin_results[0][0],
    spin_mean_predictions,
    "Spin Average"
)


# Gate plot
print_results(
    gate_results[0][0],
    gate_mean_predictions,
    gate_mean_nrmse,
    gate_mean_r2,
    "Gate"
)


# Spin plot
print_results(
    spin_results[0][0],
    spin_mean_predictions,
    spin_mean_nrmse,
    spin_mean_r2,
    "Spin"
)


plt.show()
