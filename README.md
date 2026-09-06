# Comparison of Quantum Reservoir Architectures for Audio Dynamics and Prediction of Audio Time Series

This repository contains the implementation and experimental results for my bachelor's thesis.

The project investigates Quantum Reservoir Computing (QRC) for temporal data and compares two different quantum reservoir architectures:

- Gate-based quantum reservoir
- Spin-based quantum reservoir

The reservoirs are evaluated on a Mackey-Glass benchmark and audio time series.

## Project Structure

- `Gate.py` – Gate-based quantum reservoir implementation
- `Spin.py` – Spin-based quantum reservoir implementation
- `config.py` – Experimental configuration and parameters
- `datasets.py` – Time-series and audio data processing
- `utils.py` – Evaluation and helper functions
- `run.py` – Runs the experiments
- `audioData/` – Audio input data
- `results/` – Experimental results and plots

## Experiments

The experiments investigate the influence of different reservoir configurations, including:

- Number of qubits
- Reservoir topology
- Number of layers
- Input qubits
- Readout method
- Leak rate

The main evaluation metrics are:

- NRMSE
- R²

## Data

The project uses the Mackey-Glass time series as a benchmark and audio recordings represented as RMS amplitude envelopes.

## Requirements

- Python
- NumPy
- SciPy
- scikit-learn
- Qiskit
- Qiskit Aer
- Matplotlib

## Running the Experiments

Run:

```bash
python run.py