import numpy as np
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import config

def create_rng(seed):

    return np.random.default_rng(seed)


def get_number_of_input_qubits(
    input_qubits,
    num_qubits
):

    if input_qubits == "all":

        return num_qubits

    if not isinstance(input_qubits, int):

        raise ValueError(
            "INPUT_QUBITS must be 'all' or an integer."
        )

    if input_qubits < 1 or input_qubits > num_qubits:

        raise ValueError(
            "INPUT_QUBITS must be between "
            "1 and NUM_QUBITS."
        )

    return input_qubits


def calculate_nrmse(
    targets,
    predictions
):

    rmse = np.sqrt(
        np.mean(
            (targets - predictions) ** 2
        )
    )

    std = np.std(targets)

    if std == 0:

        raise ValueError(
            "Target values have zero standard deviation."
        )

    return rmse / std


def calculate_r2(
    targets,
    predictions
):

    return r2_score(
        targets,
        predictions
    )


def print_evaluation(
    targets,
    predictions,
    name
):
    print("Number of targets:", len(targets))
    print("Number of predictions:", len(predictions))
    print("Target std:", np.std(targets))

    nrmse = calculate_nrmse(
        targets,
        predictions
    )

    r2 = calculate_r2(
        targets,
        predictions
    )

    print(f"\n{name} Reservoir")
    print(f"NRMSE = {nrmse:.4f}")
    print(f"R²    = {r2:.4f}")

    print("\nStep   Target   Prediction")

    for i in range(
        min(20, len(targets))
    ):

        print(
            f"{i:4d}   "
            f"{targets[i]:7.4f}   "
            f"{predictions[i]:7.4f}"
        )

    return nrmse, r2

def print_results(
    test_targets,
    predictions,
    nrmse,
    r2,
    name
):

    plt.figure(
        figsize=(10, 5)
    )

    plt.plot(
        test_targets,
        label="Target",
        color="blue",
        linewidth=2
    )

    plt.plot(
        predictions,
        label="Prediction",
        color="red",
        linestyle="--",
        linewidth=2
    )

    if config.PREDICTION_OFFSET != 0:

        offset_text = (
            f", Offset={config.PREDICTION_OFFSET:+d} steps"
        )

    else:

        offset_text = ""

    plt.title(
        f"{name} Reservoir Predictions "
        f"(NRMSE={nrmse:.4f}, R²={r2:.4f}"
        f"{offset_text})"
    )

    plt.xlabel(
        "Test Step"
    )

    plt.ylabel(
        "Time Series Value"
    )

    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    return plt.gcf()

def apply_prediction_offset(targets, predictions, offset):
    if offset > 0:
        targets = targets[:-offset]
        predictions = predictions[offset:]

    elif offset < 0:
        offset = abs(offset)
        targets = targets[offset:]
        predictions = predictions[:-offset]

    return targets, predictions