import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

file = Path(__file__).parent / "results" / "master_results_main.xlsx"

df = pd.read_excel(file, header=0)

# Remove accidental spaces from column names
df.columns = df.columns.str.strip()

df["Inputqubit"] = df["Inputqubit"].replace("ALL", 8)

print(df.columns.tolist())



section_column = "Experiment"
model_column = "Model"
nrmse_column = "Nrmse"
r2_column = "R2"

# Which column should be on the X-axis
x_columns = {
    "QUBIT": "Qubits",
    "LAYERS": "Layers",
    "LEAK": "Leak",
    "TOPOLOGY": "Topology",
    "INPUT": "Inputqubit",
    "FEATURE": "Feature"
}



# CREATE GRAPHS

for section, x_column in x_columns.items():

    data = df[
        df[section_column].astype(str).str.upper() == section
    ].copy()

    # Skip if there are no results
    if data.empty:
        continue

    # Convert metrics to numbers
    data[nrmse_column] = pd.to_numeric(
        data[nrmse_column],
        errors="coerce"
    )

    data[r2_column] = pd.to_numeric(
        data[r2_column],
        errors="coerce"
    )

    # Remove invalid rows
    data = data.dropna(
        subset=[nrmse_column, r2_column]
    )


    # NRMSE GRAPH


    plt.figure(figsize=(8, 5))

    for model in data[model_column].unique():

        model_data = data[
            data[model_column] == model
        ]

        plt.plot(
            model_data[x_column],
            model_data[nrmse_column],
            marker="o",
            label=model.capitalize()
        )

    plt.title(f"Effect of {x_column} on Prediction Error (NRMSE)")
    plt.xlabel(x_column)
    plt.ylabel("NRMSE")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()



    # R² GRAPH


    plt.figure(figsize=(8, 5))

    for model in data[model_column].unique():

        model_data = data[
            data[model_column] == model
        ]

        plt.plot(
            model_data[x_column],
            model_data[r2_column],
            marker="o",
            label=model.capitalize()
        )

    plt.title(f"Effect of {x_column} on Prediction Performance (R²)")
    plt.xlabel(x_column)
    plt.ylabel("R²")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()