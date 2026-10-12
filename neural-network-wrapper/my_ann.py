"""
MyANN - A configurable Artificial Neural Network wrapper built with TensorFlow/Keras.

This module defines the MyANN class, which:
- Loads tabular data from a CSV file
- Automatically preprocesses features (one-hot encoding + scaling)
- Builds and trains a neural network for regression or classification
- Supports EarlyStopping during training
- Provides evaluation metrics and visualization tools

The class is designed to be used both programmatically and from a Jupyter notebook
for experimentation and analysis.
"""

# ===== my_ann.py - Step 1: skeleton + validations + csv load =====
import os
import pandas as pd
import numpy as np

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping

from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, classification_report

import matplotlib.pyplot as plt



class MyANN:
    def __init__(
            self,
            data_set: str,
            target: str,
            hidden_layer_sizes: tuple,
            activation: str = "relu",
            loss: str = "mse",
            optimizer: str = "adam",
            batch_size: int = None,
            epochs: int = 1,
            monitor: str = "val_loss",
            patience: int = None,
            mode: str = "auto",
            verbose: int = 1,
        ):

            # data_set: path to existing csv
            if not isinstance(data_set, str):
                raise ValueError("'data_set' must be a string path to a CSV file.")
            
            if not os.path.isabs(data_set):
                raise ValueError("'data_set' must be an absolute path to a CSV file.")
            
            if not os.path.isfile(data_set):
                raise ValueError(f"Could not find CSV file at: {os.path.abspath(data_set)}")
            try:
                self._df = pd.read_csv(data_set)  # read CSV directly
            except Exception as e:
                raise ValueError(f"Could not read CSV file at '{os.path.abspath(data_set)}': {e}")

            self.data_set = data_set

    

            # target: must be a column in the csv
            if not isinstance(target, str):
                raise ValueError("'target' must be a string column name.")
            if target not in self._df.columns:
                raise ValueError(f"'target' '{target}' not found in columns: {list(self._df.columns)}")
            self.target = target

            # hidden_layer_sizes: tuple of ints (neurons) or negative floats (dropout fractions)
            if not isinstance(hidden_layer_sizes, tuple):
                raise ValueError("'hidden_layer_sizes' must be a tuple.")
            for i, layer in enumerate(hidden_layer_sizes, start=1):
                if isinstance(layer, int):
                    if layer <= 0:
                        raise ValueError(f"Layer #{i} size must be a positive integer, got {layer}")
                elif isinstance(layer, float):
                    if not (-1 < layer < 0):
                        raise ValueError(f"Layer #{i} must be int (neurons) or negative float (dropout), got {type(layer)}.")
                else:
                    raise ValueError(f"Layer #{i} must be int (neurons) or negative float (dropout), got {type(layer)}.")
            self.hidden_layer_sizes = hidden_layer_sizes


            # activation / loss / optimizer
            if activation not in {"relu", "sigmoid", "softmax", "tanh"}:
                raise ValueError("'activation' must be one of {'relu','sigmoid','softmax','tanh'}.")
            self.activation = activation

            if loss not in {"mse", "binary_crossentropy", "categorical_crossentropy"}:
                raise ValueError("'loss' must be one of {'mse', 'binary_crossentropy', 'categorical_crossentropy'}.")
            self.loss = loss
    
            if optimizer not in {"adam", "rmsprop", "sgd"}:
                raise ValueError("'optimizer' must be one of {'adam', 'rmsprop', 'sgd'}.")
            self.optimizer = optimizer
    

            # batch_size / epochs
            if batch_size is None:
                batch_size = 32

            elif not (isinstance(batch_size, int) and batch_size > 0):
                raise ValueError("'batch_size' must be a positive integer or None.")
            self.batch_size = batch_size
    
            if not (isinstance(epochs, int) and epochs >= 1):
                raise ValueError("'epochs' must be an integer >= 1.")
            self.epochs = epochs
    

            # monitor / patience / mode / verbose
            if monitor not in {"val_loss", "accuracy"}:
                raise ValueError("'monitor' must be 'val_loss' or 'accuracy'.")
            self.monitor = monitor
    
            if patience is None:
                patience = epochs
            elif not (isinstance(patience, int) and patience >= 0):
                raise ValueError("'patience' must be a non-negative integer or None.")
            self.patience = patience
    
            if mode not in {"auto", "min", "max"}:
                raise ValueError("'mode' must be one of {'auto','min','max'}.")
            self.mode = mode
    
            if verbose not in {0, 1, 2}:
                raise ValueError("'verbose' must be 0, 1 or 2.")
            self.verbose = verbose
    

            # feature names now; will transform later
            self.features_ = [c for c in self._df.columns if c != self.target]
    
            # public attributes to fill after training
            self.classes_ = None
            self.n_outputs_ = None
            self.out_activation_ = None
            self.loss_ = None
            self.best_loss = None
            self.n_layers = None
    
            # placeholders for objects we will build later
            self._scaler = None
            self._model = None
            self._history = None
            self._X_train = self._X_test = None
            self._y_train = self._y_test = None


            # ===== Step 2: Data prep (one-hot, split, scale) =====
            data = self._df.copy()

            # Split X / y
            X = data.drop(columns=[self.target])
            y = data[self.target]

            # One-hot encode categorical columns in X
            cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
            if cat_cols:
                X = pd.get_dummies(X, columns=cat_cols, drop_first=True)


            # Store final feature names
            self.features_ = X.columns.tolist()

            self._train_columns = X.columns



            # To numpy
            X_arr = X.values
            y_arr = y.values


            # Train/test split
            X_train, X_test, y_train, y_test = train_test_split(
                        X_arr, y_arr, test_size=0.2, random_state=42
                    )


            # Scale features

            self._scaler = StandardScaler()
            X_train_scaled = self._scaler.fit_transform(X_train)
            X_test_scaled = self._scaler.transform(X_test)


            # Keep for later steps

            self._X_train, self._X_test = X_train_scaled, X_test_scaled
            self._y_train, self._y_test = y_train, y_test
            # Keep raw (unprocessed) labels for metrics/reports later
            self._y_train_raw, self._y_test_raw = y_train.copy(), y_test.copy()

            # ===== Step 3: Build + compile the Keras model =====

            input_dim = self._X_train.shape[1]

            # Decide how to handle y based on loss
            
            self._label_encoder = None
            self._task = "regression"  # or "binary" / "multiclass" set below
            metrics = []

            if self.loss == "mse":
                self._task = "regression"
                # Regression
                y_train_proc = self._y_train.astype(float)
                y_test_proc = self._y_test.astype(float)
                units = 1
                out_activation = "linear"
                self.classes_ = None
                self.n_outputs_ = 1

            elif self.loss == "binary_crossentropy":
                self._task = "binary"
                # Binary classification
                self._label_encoder = LabelEncoder().fit(self._y_train)
                y_train_proc = self._label_encoder.transform(self._y_train)
                y_test_proc = self._label_encoder.transform(self._y_test)
                self.classes_ = self._label_encoder.classes_.tolist()
                units = 1
                out_activation = "sigmoid"
                metrics = ["accuracy"]
                self.n_outputs_ = 1

            elif self.loss == "categorical_crossentropy":
                self._task = "multiclass"
                # Multi-class classification
                self._label_encoder = LabelEncoder().fit(self._y_train)
                y_train_int = self._label_encoder.transform(self._y_train)
                y_test_int = self._label_encoder.transform(self._y_test)
                n_classes = len(self._label_encoder.classes_)
                y_train_proc = to_categorical(y_train_int, num_classes=n_classes)
                y_test_proc = to_categorical(y_test_int, num_classes=n_classes)
                self.classes_ = self._label_encoder.classes_.tolist()
                units = n_classes
                out_activation = "softmax"
                metrics = ["accuracy"]
                self.n_outputs_ = n_classes

            else:
                raise ValueError(f"Unsupported loss: {self.loss}")
            
            self.out_activation_ = out_activation

            # Build the Sequential model from hidden_layer_sizes

            self._model = Sequential()
            first_dense_added = False

            for layer in self.hidden_layer_sizes:
                if isinstance(layer, int):
                    # Dense layer
                    if not first_dense_added:
                        # Prefer an explicit Input layer (avoids Keras warning)
                        self._model.add(Input(shape=(input_dim,)))
                        self._model.add(Dense(layer, activation=self.activation))
                        first_dense_added = True

                    else:
                        self._model.add(Dense(layer, activation=self.activation))
                    
                else:
                    # negative float -> dropout (e.g. -0.3 => Dropout(0.3))
                    self._model.add(Dropout(-layer))

            # Add output layer
            self._model.add(Dense(units, activation=out_activation))

            # Compile model
            self._model.compile(
                optimizer=self.optimizer,
                loss=self.loss,
                metrics=metrics
            )


            # ===== Step 4: Train with EarlyStopping =====
            early_stop = EarlyStopping(
                monitor=self.monitor,
                patience=self.patience,
                mode=self.mode,
                verbose=self.verbose,
                restore_best_weights=True
            )

            history = self._model.fit(
                self._X_train,
                y_train_proc,
                validation_data=(self._X_test, y_test_proc),
                epochs=self.epochs,
                batch_size=self.batch_size,
                callbacks=[early_stop],
                verbose=self.verbose
            )

            # Save history & summary attributes
            self._history = history.history
            self.loss_ = float(self._history["loss"][-1])
            self.best_loss = float(min(self._history.get("val_loss", self._history["loss"])))
            self.n_layers = len(self._model.layers)

            # Overwrite y with processed versions (for consistency later)
            self._y_train = y_train_proc
            self._y_test = y_test_proc
    
    def model_loss(self) -> pd.DataFrame:
        """Return training history as a DataFrame (loss/val_loss and, if present, accuracy/val_accuracy)."""
        if self._history is None:
            raise ValueError("No history found. Train the model first.")
        return pd.DataFrame(self._history)

    def plot_model_loss(self) -> None:
        """Plot model loss and val_loss against epochs."""
        df = self.model_loss()
        if "loss" not in df.columns:
            raise ValueError("History does not contain 'loss'.")

        plt.figure()
        plt.plot(df["loss"], label="loss")
        if "val_loss" in df.columns:
            plt.plot(df["val_loss"], label="val_loss")
        plt.title("Model loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()

    def plot_model_accuracy(self) -> None:
        """Plot accuracy and val_accuracy (classification only)."""
        if self._task == "regression":
            print("Accuracy plot is not applicable for regression (loss='mse').")
            return

        df = self.model_loss()
        if "accuracy" not in df.columns:
            print("No accuracy information found in history.")
            return

        plt.figure()
        plt.plot(df["accuracy"], label="accuracy")
        if "val_accuracy" in df.columns:
            plt.plot(df["val_accuracy"], label="val_accuracy")
        plt.title("Model accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.show()

    def model_predict(self, X_new):
        """Predict from a single row (dict/Series) or multiple rows (DataFrame/list). Handles one-hot + scaling."""
        if self._model is None or self._scaler is None:
            raise ValueError("Model has not been trained yet.")

        # 1) Convert input to DataFrame
        if isinstance(X_new, dict):
            X_new = pd.DataFrame([X_new])
        elif isinstance(X_new, pd.Series):
            X_new = pd.DataFrame([X_new.to_dict()])
        elif isinstance(X_new, list):
            X_new = pd.DataFrame(X_new, columns=self.features_)
        elif not isinstance(X_new, pd.DataFrame):
            raise ValueError("X_new must be a pandas DataFrame, Series, dict, or list.")

        # 2) One-hot encode like training
        cat_cols = X_new.select_dtypes(include=["object", "category"]).columns.tolist()
        if cat_cols:
            X_new = pd.get_dummies(X_new, columns=cat_cols, drop_first=True)

        # 3) Align to training columns
        X_new = X_new.reindex(columns=self._train_columns, fill_value=0)

        # 4) Scale
        X_scaled = self._scaler.transform(X_new.values)

        # 5) Predict
        preds = self._model.predict(X_scaled, verbose=0)

        # Regression: flatten
        if self.loss == "mse":
            return preds.reshape(-1)

        # Binary classification
        if self.loss == "binary_crossentropy":
            probs = preds.reshape(-1)
            labels = (probs >= 0.5).astype(int)
            if self._label_encoder is not None:
                return self._label_encoder.inverse_transform(labels)
            return labels

        # Multiclass classification
        if self.loss == "categorical_crossentropy":
            labels = np.argmax(preds, axis=1)
            if self._label_encoder is not None:
                return self._label_encoder.inverse_transform(labels)
            return labels

        return preds

    def save_original_keras_model(self, file_path: str) -> None:
        """Save the underlying Keras model (NOT the MyANN object) in .h5 format."""
        if self._model is None:
            raise ValueError("Model has not been trained yet.")
        if not isinstance(file_path, str) or not file_path.lower().endswith(".h5"):
            raise ValueError("file_path must be a string ending with '.h5'.")
        self._model.save(file_path)

    def load_original_keras_model(self, file_path: str) -> None:
        """Load a saved Keras model (.h5) into this object (NOT the MyANN object)."""
        if not isinstance(file_path, str) or not file_path.lower().endswith(".h5"):
            raise ValueError("file_path must be a string ending with '.h5'.")
        if not os.path.isfile(file_path):
            raise ValueError(f"Could not find .h5 file at: {os.path.abspath(file_path)}")
        self._model = load_model(file_path)

    def print_classification_report(self) -> None:
        """Print sklearn classification report (classification only)."""
        if self._task == "regression":
            print("Classification report is not applicable for regression (loss='mse').")
            return

        # Build predictions from already-prepared test features
        preds = self._model.predict(self._X_test, verbose=0)

        if self._task == "binary":
            y_pred_int = (preds.reshape(-1) >= 0.5).astype(int)
            y_true_int = self._label_encoder.transform(self._y_test_raw)
            print(classification_report(y_true_int, y_pred_int, target_names=self.classes_))
            return

        # multiclass
        y_pred_int = np.argmax(preds, axis=1)
        y_true_int = self._label_encoder.transform(self._y_test_raw)
        print(classification_report(y_true_int, y_pred_int, target_names=self.classes_))

    def rmse(self) -> float:
        """Return RMSE (regression only)."""
        if self._task != "regression":
            print("RMSE is not applicable for classification.")
            return float("nan")
        y_pred = self._model.predict(self._X_test, verbose=0).reshape(-1)
        y_true = self._y_test_raw.astype(float)
        return float(np.sqrt(mean_squared_error(y_true, y_pred)))

    def mae(self) -> float:
        """Return MAE (regression only)."""
        if self._task != "regression":
            print("MAE is not applicable for classification.")
            return float("nan")
        y_pred = self._model.predict(self._X_test, verbose=0).reshape(-1)
        y_true = self._y_test_raw.astype(float)
        return float(mean_absolute_error(y_true, y_pred))

    def plot_predictions_scatter(self) -> None:
        """Scatter plot of real vs predicted target values (regression only)."""
        if self._task != "regression":
            print("Prediction scatter is not applicable for classification.")
            return
        y_pred = self._model.predict(self._X_test, verbose=0).reshape(-1)
        y_true = self._y_test_raw.astype(float)

        plt.figure()
        plt.scatter(y_true, y_pred)
        plt.title("Predicted vs Real")
        plt.xlabel("Real")
        plt.ylabel("Predicted")
        plt.show()

    def plot_residual_error(self) -> None:
        """Plot histogram of residual errors (real - predicted) (regression only)."""
        if self._task != "regression":
            print("Residual error plot is not applicable for classification.")
            return
        y_pred = self._model.predict(self._X_test, verbose=0).reshape(-1)
        y_true = self._y_test_raw.astype(float)
        residuals = y_true - y_pred

        plt.figure()
        plt.hist(residuals, bins=20)
        plt.title("Residual errors")
        plt.xlabel("Residual (real - predicted)")
        plt.ylabel("Count")
        plt.show()

    # Backward-compatible names from earlier iterations
    def evaluate(self):
        return self.model_loss()

    def predict(self, X_new):
        return self.model_predict(X_new)

    def plot_history(self):
        return self.plot_model_loss()