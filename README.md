# ML Playground

An interactive web application for exploring and comparing classification algorithms. The goal is to make model behavior visible: adjust the data and hyperparameters, train, and immediately see the results as decision boundaries, metrics, learning curves, and more.

Built with Streamlit, scikit-learn, PyTorch, and Plotly.

## Features

### Decision Boundary

Displays the classifier's decision region over the feature space, with training and test points overlaid. Incorrectly classified test points are highlighted separately.

A toggle replaces the binary boundary with a confidence map: the background shows the maximum predicted probability at each point in the space, making it visible where the model is certain and where it hesitates.

### Metrics

- Confusion matrix normalized by row
- Table with precision, recall, F1, and support per class
- ROC curve with AUC for binary classification; one-vs-rest for multiclass
- Learning curves (training and cross-validation accuracy as a function of training set size), computed on demand with 5-fold CV

### Neural Network

A dedicated tab for neural network models (MLP and PyTorch):

- Interactive architecture diagram, updated in real time as you adjust the layer and neuron sliders, even before training
- Loss curve per iteration/epoch after training, with a convergence indicator
- Validation loss curve (PyTorch only) — 15% of data automatically held out to monitor overfitting epoch by epoch
- Weight distribution per layer after training (PyTorch only) — interactive histogram to inspect initialization and optimizer behavior
- Comparison chart of the most common activation functions (ReLU, Tanh, Sigmoid, Leaky ReLU)

### Model Comparison

Trains all 7 algorithms with default hyperparameters on the same dataset and displays a bar chart comparing training and test accuracy.

## Available Algorithms

| Algorithm | Configurable Hyperparameters |
|---|---|
| K-Nearest Neighbors | k (number of neighbors) |
| SVM with RBF kernel | C, gamma |
| SVM with Linear kernel | C |
| Decision Tree | max depth |
| Random Forest | number of trees, max depth |
| Gradient Boosting | number of estimators, learning rate |
| Neural Network (MLP) | hidden layers, neurons per layer, learning rate, activation function, solver |
| Neural Network (PyTorch) | hidden layers, neurons per layer, learning rate, activation function, epochs, batch size, dropout, batch normalization |

## Available Datasets

| Dataset | Description |
|---|---|
| Moons | Two crescent-shaped classes |
| Circles | Concentric circles |
| XOR | Classic XOR pattern |
| Spiral | Two interleaved spirals |
| Spiral (3) | Three interleaved spirals, hard multiclass problem |
| Checkerboard | Checkerboard pattern, requires periodic and non-linear boundaries |
| Blobs (2) | Two Gaussian clusters |
| Blobs (3) | Three Gaussian clusters |

The number of samples, noise level, train/test split ratio, and random seed are all controllable from the sidebar.

## Project Structure

```
ml_playground/
├── app.py                  # Main application (Streamlit)
├── requirements.txt
├── utils/
│   ├── datasets.py         # Dataset generation
│   ├── models.py           # Classifier instantiation
│   ├── plots.py            # All charts (Plotly)
│   └── torch_model.py      # PyTorchMLP — scikit-learn wrapper over a PyTorch network
└── .streamlit/
    └── config.toml         # Theme and server configuration
```

## Installation and Usage

**Requirements:** Python 3.10 or higher.

```bash
# Clone the repository
git clone https://github.com/ric-rky/interactive_ml_playground
cd ml_playground

# Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Linux / macOS

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
```

The application opens automatically in the browser at `http://localhost:8501`.

## Dependencies

```
streamlit>=1.35.0
scikit-learn>=1.4.0
numpy>=1.26.0
plotly>=6.0.0
pandas>=2.2.0
torch>=2.0.0
```
