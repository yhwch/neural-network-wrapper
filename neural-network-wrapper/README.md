# MyANN: a configurable neural network wrapper

A Python class that turns a CSV file into a trained neural network with one call. It handles preprocessing, builds the network from a simple layer description, trains it, and gives you metrics and plots.

Built as a school assignment in the course *Maskininlärning och djupinlärning* (Machine Learning and Deep Learning).

## What it does

- Loads a CSV and validates every argument with clear error messages
- Preprocesses automatically: one-hot encodes categorical columns and scales numeric ones
- Detects the task type: regression, binary or multi-class classification
- Builds the network from a tuple: positive ints are Dense layers, negative floats are Dropout (e.g. `(32, -0.3, 16)`)
- Supports EarlyStopping
- Evaluation: RMSE, MAE, classification report, loss/accuracy curves, prediction scatter and residual plots
- Save and load the underlying Keras model

## Example

```python
import os
from my_ann import MyANN

m = MyANN(
    data_set=os.path.abspath("tips.csv"),
    target="tip",
    hidden_layer_sizes=(32, -0.3, 16),
    activation="relu",
    loss="mse",
    optimizer="adam",
    epochs=20,
)

print(m.rmse(), m.mae())
m.plot_model_loss()
m.plot_residual_error()
```

See [`my_ann_demo.ipynb`](my_ann_demo.ipynb) for a full run on the restaurant tips dataset.

## Run it

```bash
pip install tensorflow scikit-learn pandas numpy matplotlib jupyter
jupyter notebook my_ann_demo.ipynb
```

## Tech

Python, TensorFlow/Keras, scikit-learn, pandas, matplotlib
