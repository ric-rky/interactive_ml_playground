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

### K-Nearest Neighbors

Classifies a point by majority vote among its `k` closest neighbors under Euclidean distance:

$$\hat{y} = \arg\max_{c} \sum_{i \in N_k(x)} \mathbf{1}[y_i = c]$$

No training phase — the entire dataset is the model. Sensitive to feature scale and noisy labels.

**Hyperparameters:** `k` (number of neighbors)

---

### SVM — RBF Kernel

Finds the maximum-margin hyperplane in a high-dimensional space induced by the RBF kernel:

$$K(x, x') = \exp\left(-\gamma \|x - x'\|^2\right)$$

`C` controls the trade-off between margin width and training errors; `gamma` controls the reach of each support vector (high gamma → tight fit, low gamma → smoother boundary).

**Hyperparameters:** `C`, `gamma`

---

### SVM — Linear Kernel

Same objective as RBF-SVM, but the decision boundary is a hyperplane in the original feature space:

$$f(x) = w^\top x + b, \quad \text{with } \|w\| \text{ minimized subject to } y_i f(x_i) \geq 1 - \xi_i$$

Efficient on linearly separable data; `C` penalizes margin violations.

**Hyperparameters:** `C`

---

### Decision Tree

Recursively partitions the feature space by choosing the split that maximizes information gain (reduction in entropy):

$$\text{Gain}(S, f) = H(S) - \sum_{v} \frac{|S_v|}{|S|} H(S_v), \quad H(S) = -\sum_{c} p_c \log_2 p_c$$

Fully interpretable; prone to overfitting without depth constraints.

**Hyperparameters:** max depth

---

### Random Forest

An ensemble of `T` decision trees, each trained on a bootstrap sample with a random subset of features at each split. Predictions are aggregated by majority vote:

$$\hat{y} = \text{majority}\{h_1(x), h_2(x), \ldots, h_T(x)\}$$

Variance reduction through averaging makes it robust to overfitting.

**Hyperparameters:** number of trees, max depth

---

### Gradient Boosting

Builds an additive model by fitting each new tree to the residuals (negative gradient of the loss) of the current ensemble:

$$F_m(x) = F_{m-1}(x) + \eta \cdot h_m(x)$$

where `η` is the learning rate and `h_m` minimizes the loss at step `m`. Highly accurate but sensitive to overfitting when the learning rate is too high.

**Hyperparameters:** number of estimators, learning rate

---

### Neural Network (MLP)

A feedforward network where each layer applies a linear transformation followed by a non-linear activation:

$$a^{(l)} = \sigma\left(W^{(l)} a^{(l-1)} + b^{(l)}\right)$$

Trained by backpropagation via gradient descent. The final layer uses softmax for multiclass output.

**Hyperparameters:** hidden layers, neurons per layer, learning rate, activation function, solver

---

### Neural Network (PyTorch)

Same MLP architecture as above, with additional regularization techniques:

- **Dropout:** randomly zeros activations during training with probability `p`, preventing co-adaptation of neurons
- **Batch Normalization:** normalizes layer inputs to zero mean and unit variance, stabilizing and accelerating training:

$$\hat{x} = \frac{x - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}, \quad y = \gamma \hat{x} + \beta$$

Trained with Adam optimizer and a held-out validation set for monitoring overfitting.

**Hyperparameters:** hidden layers, neurons per layer, learning rate, activation function, epochs, batch size, dropout, batch normalization

---

### Summary Table

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
