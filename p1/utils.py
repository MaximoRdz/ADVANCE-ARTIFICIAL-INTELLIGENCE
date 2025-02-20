"""TODO."""

import copy
import os

import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input


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
    test_driver_ind, maneuver_ind, window_size, drivers, maneuver_names, dfs, window_step=None, verbose=False
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


def evaluate_drivers(drivers, maneuver_id, maneuver_names, dfs, window_size, window_step, model_type="logreg", verbose=True):
    cross_val_accs = []
    test_accs = []
    test_f1_scores = []

    for i, driver in enumerate(drivers):
        if verbose:
            print(f"\nDriver: {driver}\n")
        
        X_train_driver_fold, X_test_driver_fold, y_train_driver_fold, y_test_driver_fold = get_loaders(
            test_driver_ind=i,
            maneuver_ind=maneuver_id,
            drivers=drivers,
            maneuver_names=maneuver_names,
            dfs=dfs,
            window_size=window_size,
            window_step=window_step,
        )
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train_driver_fold)
        X_test = scaler.transform(X_test_driver_fold)

        # TimeSeriesSplit with 5 splits
        tscv = TimeSeriesSplit(n_splits=5)
        model_scores = []

        # Cross-validation
        for train_index, test_index in tscv.split(X_train):  
            X_train_fold, X_val_fold = X_train[train_index], X_train[test_index]
            y_train_fold, y_val_fold = y_train_driver_fold.iloc[train_index], y_train_driver_fold.iloc[test_index]

            if model_type == "logreg":
                # Logistic Regression model
                logreg_model = LogisticRegression(penalty='l1', solver='liblinear')
                logreg_model.fit(X_train_fold, y_train_fold)
                
                logreg_y_pred = logreg_model.predict(X_val_fold)
                model_scores.append(accuracy_score(y_val_fold, logreg_y_pred))
            
            elif model_type == "lstm":
                # LSTM/GRU model
                X_train_fold = X_train_fold.reshape((X_train_fold.shape[0], 1, X_train_fold.shape[1]))  # Reshape for LSTM/GRU
                X_val_fold = X_val_fold.reshape((X_val_fold.shape[0], 1, X_val_fold.shape[1]))  # Reshape for LSTM/GRU
                
                model = Sequential()
                model.add(LSTM(50, activation='relu', return_sequences=True, input_shape=(1, X_train_fold.shape[2])))
                
                model.add(Dropout(0.2))
                model.add(Dense(1, activation='sigmoid'))
                model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
                
                model.fit(X_train_fold, y_train_fold, epochs=5, batch_size=32, verbose=0)
                
                lstm_y_pred = (model.predict(X_val_fold) > 0.5).astype(float)
                model_scores.append(accuracy_score(y_val_fold, lstm_y_pred))

        if verbose:
            print("=================== Cross-Validation ===================")
            print(f"Accuracy fold {i} {model_type}: {np.mean(model_scores):.4f}\n")
        
        cross_val_accs.append(np.mean(model_scores))
        
        # Train on full dataset
        if model_type == "logreg":
            logreg_model = LogisticRegression(penalty='l1', solver='liblinear')
            logreg_model.fit(X_train, y_train_driver_fold)
            model_test_pred = logreg_model.predict(X_test)
            
            test_score = accuracy_score(y_test_driver_fold, model_test_pred)
            test_f1 = f1_score(y_test_driver_fold, model_test_pred)
        
        elif model_type == "lstm":
            X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))  # Reshape for LSTM/GRU
            X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))  # Reshape for LSTM/GRU
            
            model = Sequential()
            model.add(LSTM(50, activation='relu', return_sequences=True, input_shape=(1, X_train.shape[2])))
                
            model.add(Dropout(0.2))
            model.add(Dense(1, activation='sigmoid'))
            model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
            
            model.fit(X_train, y_train_driver_fold, epochs=5, batch_size=32, verbose=0)
            lstm_y_pred = (model.predict(X_test) > 0.5).astype(float)
            
            test_score = accuracy_score(y_test_driver_fold, lstm_y_pred)
            test_f1 = f1_score(y_test_driver_fold, lstm_y_pred)

        if verbose:
            print("================ Test Evaluation =================")
            print(f"Accuracy final on test set {model_type}: {test_score:.4f}")
            print(f"F1 score on test set {model_type}: {test_f1:.4f}")
            print(classification_report(y_test_driver_fold, lstm_y_pred if model_type != "logreg" else model_test_pred, target_names=["no-maneuver", "maneuver"]))
            print(confusion_matrix(y_test_driver_fold, lstm_y_pred if model_type != "logreg" else model_test_pred))
        
        test_accs.append(test_score)
        test_f1_scores.append(test_f1)

    return cross_val_accs, test_accs, test_f1_scores



def create_lstm_model(input_shape, lstm_units_1=50, lstm_units_2=50, dropout_rate=0.2):
    model = Sequential()

    # Define input layer with shape
    model.add(Input(shape=input_shape))

    # Add LSTM layers with tanh activation
    model.add(LSTM(lstm_units_1, activation='relu', return_sequences=True))  # First LSTM layer
    model.add(LSTM(lstm_units_2, activation='relu', return_sequences=False))  # Second LSTM layer

    # Add Dropout layer for regularization
    model.add(Dropout(dropout_rate))

    # Add Dense output layer with sigmoid activation for binary classification
    model.add(Dense(1, activation='sigmoid'))

    # Compile the model with Adam optimizer and binary crossentropy loss
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

    return model



def evaluate_drivers(drivers, maneuver_id, maneuver_names, dfs, window_size, window_step, model_type="logreg", verbose=True):
    cross_val_accs = []
    test_accs = []
    test_f1_scores = []

    for i, driver in enumerate(drivers):
        if verbose:
            print(f"\nDriver: {driver}\n")
        
        X_train_driver_fold, X_test_driver_fold, y_train_driver_fold, y_test_driver_fold = get_loaders(
            test_driver_ind=i,
            maneuver_ind=maneuver_id,
            drivers=drivers,
            maneuver_names=maneuver_names,
            dfs=dfs,
            window_size=window_size,
            window_step=window_step,
        )
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train_driver_fold)
        X_test = scaler.transform(X_test_driver_fold)

        # TimeSeriesSplit with 5 splits
        tscv = TimeSeriesSplit(n_splits=5)
        model_scores = []

        # Cross-validation
        for train_index, test_index in tscv.split(X_train):  
            X_train_fold, X_val_fold = X_train[train_index], X_train[test_index]
            y_train_fold, y_val_fold = y_train_driver_fold.iloc[train_index], y_train_driver_fold.iloc[test_index]

            if model_type == "logreg":
                # Logistic Regression model
                logreg_model = LogisticRegression(penalty='l1', solver='liblinear')
                logreg_model.fit(X_train_fold, y_train_fold)
                
                logreg_y_pred = logreg_model.predict(X_val_fold)
                model_scores.append(accuracy_score(y_val_fold, logreg_y_pred))
            
            elif model_type == "lstm":
                # LSTM/GRU model
                X_train_fold = X_train_fold.reshape((X_train_fold.shape[0], 1, X_train_fold.shape[1]))  # Reshape for LSTM/GRU
                X_val_fold = X_val_fold.reshape((X_val_fold.shape[0], 1, X_val_fold.shape[1]))  # Reshape for LSTM/GRU
                
                model = create_lstm_model((1, X_train_fold.shape[2]))
                model.fit(X_train_fold, y_train_fold, epochs=10, batch_size=32, verbose=0)
                
                lstm_y_pred = (model.predict(X_val_fold) > 0.5).astype(int).squeeze()
                model_scores.append(accuracy_score(y_val_fold, lstm_y_pred))

        if verbose:
            print("=================== Cross-Validation ===================")
            print(f"Accuracy fold {i} {model_type}: {np.mean(model_scores):.4f}\n")
        
        cross_val_accs.append(np.mean(model_scores))
        
        # Train on full dataset
        if model_type == "logreg":
            logreg_model = LogisticRegression(penalty='l1', solver='liblinear')
            logreg_model.fit(X_train, y_train_driver_fold)
            model_test_pred = logreg_model.predict(X_test)
            
            test_score = accuracy_score(y_test_driver_fold, model_test_pred)
            test_f1 = f1_score(y_test_driver_fold, model_test_pred)
        
        elif model_type == "lstm":
            X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))  # Reshape for LSTM/GRU
            X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))  # Reshape for LSTM/GRU
            
            model = create_lstm_model((1, X_train_fold.shape[2]))
            model.fit(X_train, y_train_driver_fold, epochs=10, batch_size=32, verbose=0)
            lstm_y_pred = (model.predict(X_test) > 0.5).astype(int).squeeze()
            
            test_score = accuracy_score(y_test_driver_fold, lstm_y_pred)
            test_f1 = f1_score(y_test_driver_fold, lstm_y_pred)

        if verbose:
            print("================ Test Evaluation =================")
            print(f"Accuracy final on test set {model_type}: {test_score:.4f}")
            print(f"F1 score on test set {model_type}: {test_f1:.4f}")
            print(classification_report(y_test_driver_fold, lstm_y_pred if model_type != "logreg" else model_test_pred, target_names=["no-maneuver", "maneuver"]))
            print(confusion_matrix(y_test_driver_fold, lstm_y_pred if model_type != "logreg" else model_test_pred))
        
        test_accs.append(test_score)
        test_f1_scores.append(test_f1)

    return cross_val_accs, test_accs, test_f1_scores

