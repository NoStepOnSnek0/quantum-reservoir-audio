import numpy as np

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

from sklearn.linear_model import Ridge



import config

from datasets import load_time_series

from utils import (
    create_rng,
    get_number_of_input_qubits,
    print_evaluation,
    apply_prediction_offset
)


# Setup

simulator = AerSimulator()

num_qubits = config.NUM_QUBITS
num_measurements = config.NUM_MEASUREMENTS

washout = config.WASHOUT
num_layers = config.NUM_LAYERS

topology = config.TOPOLOGY

input_qubits = config.INPUT_QUBITS
input_scale = config.INPUT_SCALE

leak_rate = config.LEAK_RATE

memory_strength = config.MEMORY_STRENGTH
readout = config.READOUT

ridge_alpha = config.RIDGE_ALPHA

train_size = config.TRAIN_SIZE
test_size = config.TEST_SIZE


# Measurement

def measure_circuit(circuit):

    measured_circuit = circuit.copy()

    measured_circuit.measure_all()

    result = simulator.run(
        transpile(
            measured_circuit,
            simulator
        ),
        shots=num_measurements
    ).result()

    counts = result.get_counts()

    z_expectations = np.zeros(
        num_qubits
    )

    zz_expectations = {}

    if readout == "z+zz":

        for i in range(num_qubits):

            for j in range(i + 1, num_qubits):

                zz_expectations[(i, j)] = 0

    for bitstring, count in counts.items():

        bitstring = bitstring[::-1]

        z_values = []

        for qubit in range(num_qubits):

            if bitstring[qubit] == "0":
                z = 1
            else:
                z = -1

            z_values.append(z)

            z_expectations[qubit] += (
                count * z
            )

        if readout == "z+zz":

            for i in range(num_qubits):

                for j in range(i + 1, num_qubits):

                    zz_expectations[(i, j)] += (
                        count
                        * z_values[i]
                        * z_values[j]
                    )

    z_expectations /= num_measurements

    if readout == "z":

        return z_expectations

    elif readout == "z+zz":

        zz_values = []

        for i in range(num_qubits):

            for j in range(i + 1, num_qubits):

                zz_values.append(
                    zz_expectations[(i, j)]
                    / num_measurements
                )

        return np.concatenate(
            [
                z_expectations,
                np.array(zz_values)
            ]
        )

    else:

        raise ValueError(
            f"Unknown READOUT: {readout}"
        )


# Reservoir step

def reservoir_step(
    input_value,
    previous_state,
    rotation_angles
):

    circuit = QuantumCircuit(
        num_qubits
    )

    # Memory
    for qubit in range(num_qubits):
        circuit.ry(
            memory_strength
            * previous_state[qubit],
            qubit
        )

    # Number of input qubits

    number_of_input_qubits = (
        get_number_of_input_qubits(
            input_qubits,
            num_qubits
        )
    )

    # Reservoir layers

    for layer in range(num_layers):

        # Input encoding

        for qubit in range(
            number_of_input_qubits
        ):

            angle = (
                (qubit + 1)
                * input_value
                * input_scale
                / num_qubits
            )

            circuit.ry(
                angle,
                qubit
            )

        # Random rotations

        for qubit in range(num_qubits):

            circuit.ry(
                rotation_angles[
                    qubit,
                    0
                ],
                qubit
            )

            circuit.rz(
                rotation_angles[
                    qubit,
                    1
                ],
                qubit
            )

        # Entanglement

        if topology == "linear":

            for qubit in range(
                num_qubits - 1
            ):

                circuit.cx(
                    qubit,
                    qubit + 1
                )

        elif topology == "ring":

            for qubit in range(
                num_qubits - 1
            ):

                circuit.cx(
                    qubit,
                    qubit + 1
                )

            circuit.cx(
                num_qubits - 1,
                0
            )

        elif topology == "all_to_all":

            for i in range(num_qubits):

                for j in range(
                    i + 1,
                    num_qubits
                ):

                    circuit.cx(
                        i,
                        j
                    )

        else:

            raise ValueError(
                f"Unknown topology: {topology}"
            )

    # Final random rotation

    for qubit in range(num_qubits):

        circuit.ry(
            rotation_angles[
                qubit,
                2
            ],
            qubit
        )

    return measure_circuit(
        circuit
    )


# Run reservoir

def run_reservoir(
    input_values,
    rotation_angles
):

    reservoir_states = []

    previous_state = np.zeros(
        num_qubits
    )

    for input_value in input_values:

        current_state = reservoir_step(
            input_value,
            previous_state,
            rotation_angles
        )

        # Z values are always
        # the first num_qubits features

        z_state = current_state[
            :num_qubits
        ]

        updated_state = (
            (1 - leak_rate)
            * previous_state
            + leak_rate
            * z_state
        )

        if readout == "z":

            reservoir_states.append(
                updated_state
            )

        elif readout == "z+zz":

            zz_state = current_state[
                num_qubits:
            ]

            reservoir_states.append(
                np.concatenate(
                    [
                        updated_state,
                        zz_state
                    ]
                )
            )

        previous_state = updated_state

    return np.array(
        reservoir_states
    )


# Train readout

def train_readout(
    train_states,
    train_targets
):

    model = Ridge(
        alpha=ridge_alpha
    )

    model.fit(
        train_states,
        train_targets
    )

    return model

# Main

def run(seed):

    print(
        f"Starting Gate Reservoir "
        f"(Seed {seed})"
    )

    # Random initialization

    rng = create_rng(seed)

    rotation_angles = rng.uniform(
        0,
        2 * np.pi,
        (num_qubits, 3)
    )

    time_series = load_time_series()

    input_values = time_series[:-1]
    target_values = time_series[1:]

    reservoir_states = run_reservoir(
        input_values,
        rotation_angles
    )

    # Washout

    reservoir_states = (
        reservoir_states[washout:]
    )

    target_values = (
        target_values[washout:]
    )

    # Training

    train_states = (
        reservoir_states[:train_size]
    )

    train_targets = (
        target_values[:train_size]
    )

    # Testing

    test_states = (
        reservoir_states[
            train_size:
            train_size + test_size
        ]
    )

    test_targets = (
        target_values[
            train_size:
            train_size + test_size
        ]
    )

    # Train readout

    readout_model = train_readout(
        train_states,
        train_targets
    )

    # Prediction

    predictions = (
        readout_model.predict(
            test_states
        )
    )

    print("\n--- Temporal alignment check ---")

    for i in range(10):
        print(
            i,
            "target =", test_targets[i],
            "prediction =", predictions[i],
            "prev =", test_targets[i - 1] if i > 0 else None,
            "next =", test_targets[i + 1] if i < len(test_targets) - 1 else None
        )

    # Apply optional prediction offset
    if config.PREDICTION_OFFSET != 0:
        test_targets, predictions = apply_prediction_offset(
            test_targets,
            predictions,
            config.PREDICTION_OFFSET
        )

    print(
        f"Prediction offset: "
        f"{config.PREDICTION_OFFSET:+d} steps"
    )

    # Evaluation

    nrmse, r2 = print_evaluation(
        test_targets,
        predictions,
        "Gate"
    )

    return (
        test_targets,
        predictions,
        nrmse,
        r2
    )
