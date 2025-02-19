"""TODO."""

import copy
import os

import pandas as pd

attributes = [
    "speed",
    "RPM",
    "Steering wheel angle",
    "Gas pedal",
    "Brake pedal",
    "Clutch pedal",
    "Gear",
    "Maneuver marker flag",
]
features = [
    "speed",
    "RPM",
    "Steering wheel angle",
    "Gas pedal",
    "Brake pedal",
    "Clutch pedal",
    "Gear",
]
target = "Maneuver marker flag"


def get_dfs_copy(dfs):
    return copy.deepcopy(dfs)


def load_data(data_path="data/"):
    drivers = list(os.listdir(data_path))

    maneuver_names = [
        f.split("_")[1].split(".")[0]
        for f in os.listdir(os.path.join(data_path, drivers[0]))
    ]

    dfs = {}
    for driver in drivers:
        maneuvers = {}
        for maneuver in maneuver_names:
            df_path = os.path.join(data_path, driver, f"STISIMData_{maneuver}.xlsx")
            df = pd.read_excel(df_path)
            maneuvers[maneuver] = df
        dfs[driver] = maneuvers

    dfs_copy = copy.deepcopy(dfs)

    for driver in drivers:
        for maneuver in maneuver_names:
            dfs_copy[driver][maneuver] = dfs_copy[driver][maneuver][attributes]

    return drivers, maneuver_names, dfs_copy

def get_loaders(
    test_driver_ind, maneuver_ind, window_size, drivers, maneuver_names, dfs ,  window_step=None, verbose=False
):
    if verbose:
        print("INFO: parsing driving dataset")
        print(f"INFO: drivers found {drivers}")
        print(f"INFO: maneuvers found {maneuver_names}")
        print("INFO:")
        print(f"INFO: summary:")
        print(f"INFO: using maneuver {maneuver_names[maneuver_ind]} data")
        print(f"INFO: using driver {drivers[test_driver_ind]} for test split")
        print(f"INFO: {'no-overlapping' if window_step else 'overlapping'} processed dataframe")
        print(f"INFO: window-size {window_size}")

    X_train_list, X_test_list = [], []
    y_train_list, y_test_list = [], []

    # Iterar sobre todos los conductores para una misma maniobra
    for i, driver in enumerate(drivers):
        df = copy.deepcopy(dfs[driver][maneuver_names[maneuver_ind]])

        # Crear nuevas características con ventana deslizante
        df['speed_mean'] = df['speed'].rolling(window=window_size, min_periods=1, step=window_step).mean()
        df['speed_std'] = df['speed'].rolling(window=window_size, min_periods=1, step=window_step).std()
        df['RPM_mean'] = df['RPM'].rolling(window=window_size, min_periods=1, step=window_step).mean()
        df['RPM_std'] = df['RPM'].rolling(window=window_size, min_periods=1, step=window_step).std()
        df['brake_mean'] = df['Brake pedal'].rolling(window=window_size, min_periods=1, step=window_step).mean()
        df['brake_std'] = df['Brake pedal'].rolling(window=window_size, min_periods=1, step=window_step).std()
        df['gas_mean'] = df['Gas pedal'].rolling(window=window_size, min_periods=1, step=window_step).mean()
        df['gas_std'] = df['Gas pedal'].rolling(window=window_size, min_periods=1, step=window_step).std()

        # Definir el target: si ocurre una maniobra en el próximo instante (t+1)
        df["target"] = df["Maneuver marker flag"].shift(-1)

        # Eliminar filas con valores NaN creados por shift y rolling
        df = df.dropna()

        # Seleccionar variables relevantes
        X_driver = df.drop(columns=["Maneuver marker flag", "target"])
        y_driver = df["target"]

        if i == test_driver_ind:
            X_test_list.append(X_driver)
            y_test_list.append(y_driver)
        else:
            X_train_list.append(X_driver)
            y_train_list.append(y_driver)

    X_train = pd.concat(X_train_list, ignore_index=True)
    X_test = pd.concat(X_test_list, ignore_index=True)
    y_train = pd.concat(y_train_list, ignore_index=True)
    y_test = pd.concat(y_test_list, ignore_index=True)

    if verbose:
        print(f"INFO: X_train {X_train.shape}, X_test {X_test.shape}")
        print(f"INFO: y_train {y_train.shape}, y_test {y_test.shape}")

    return X_train, X_test, y_train, y_test
