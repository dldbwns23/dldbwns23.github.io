---
layout: post
title: "CCAI Forecasting ENSO Review"
date: 2025-11-26 20:54:00 +0900
tags: [CCAI, Visualization, Python, ML/DL]
---

# Review: Forecasting El Niño / Southern Oscillation with Machine Learning

## 0. Overview
This post reviews and shares key insights from the tutorial **"Forecasting the El Niño / Southern Oscillation (ENSO) with Machine Learning"** provided by Climate Change AI. This tutorial serves as an excellent introduction to applying both classical machine learning and deep learning techniques to climate time-series data.

For the full source code and implementation details, please refer to the [GitHub repository](https://github.com/climatechange-ai-tutorials/seasonal-forecasting?tab=readme-ov-file) or run the accompanying Colab Notebook.

---

## 1. Reference
Mahesh, A., & Hanna, M. (2024). *Forecasting the El Nino/ Southern Oscillation with Machine Learning*. Climate Change AI Tutorials.  
Available at: [https://climatechange.ai/tutorials](https://climatechange.ai/tutorials?search=id:forecasting-the-el-nino-southern-oscillation-with-machine-learning)

---

## 2. Dataset Description
The model utilizes historical climate data to predict ENSO events, specifically looking for lead times (predicting the future state based on current conditions).

* **COBE Sea-Surface Temperature (SST) Dataset:** A historical dataset of global sea surface temperatures spanning from 1880 to 2018.
* **Niño3.4 Index:** The target variable, representing the 3-month rolling average of equatorial Pacific Ocean temperature anomalies.

**Variables**
* **Input (Predictors):** Sea Surface Temperature (K).
* **Output (Target):** Niño3.4 Index (indicates the state of ENSO).

**Data Partitioning**
* **Training Set (Baseline):** 1980–1995 SST and corresponding Niño3.4 Index at lead times of 1 to 5 months.
* **Validation Set:** 1997–2006 SST and corresponding Niño3.4 Index.
* **Test Set:** 2007–2017 SST and corresponding Niño3.4 Index.

---

## 3. Machine Learning for Basic ENSO Forecasting

The tutorial first introduces basic linear regression. This section provides clear examples of overfitting and demonstrates the necessity of regularization techniques like Ridge Regression.

**Key Observation: Temporal Correlation**
A critical takeaway is that standard random sampling is insufficient for time series data. Even if the training, validation, and test sets are randomly selected, the model may overfit due to the "memory" inherent in time series data (autocorrelation).
* **Implication:** To ensure robust evaluation, data splits must be separated by sufficient time gaps to prevent data leakage.
* **Critique:** The tutorial does not explicitly define a standard metric for determining the optimal time gap, likely because this parameter is highly problem-dependent.

### Regularization and Explainability
The following code snippets demonstrate the impact of regularization. By penalizing large weights, Ridge Regression reduces model variance and prevents the model from relying too heavily on noise.

**Visualizing Weight Magnitude**
```python
"""In this cell, we will visualize how the weights of the linear
regression model are bigger than those of the ridge regression model"""

#coef_ loads in the coefficients from the trained model
regr_1_month_lr_weights = regr_1_month_lr.coef_
regr_1_month_ridge_weights = regr_1_month_ridge.coef_

plt.bar(range(regr_1_month_lr_weights.shape[0]), regr_1_month_lr_weights,
        label='Linear Regression Weights')
plt.bar(range(regr_1_month_ridge_weights.shape[0]), regr_1_month_ridge_weights,
        label='Ridge Regression Weights')
plt.legend(loc='best')
plt.ylabel('Value of Learned Weight')
plt.title('Comparison of the Size of Weights of Linear and Ridge Regression')
plt.show()
```

**Visualizing Variance with Perturbed Data**
```python
"""In this cell, we will visualize the variance of linear regression and ridge regression"""
for experiment_num in range(1000):
    perturbed_X_test = X_test * np.random.uniform(-1.05,1.05)

    perturbed_predictions_linear = regr_1_month_lr.predict(perturbed_X_test)
    perturbed_predictions_linear = pd.Series(perturbed_predictions_linear,
                                             index=y_test.index)
    plt.plot(perturbed_predictions_linear, '--', label='Linear', color='blue',
             alpha=0.6)

    perturbed_predictions_ridge = regr_1_month_ridge.predict(perturbed_X_test)
    perturbed_predictions_ridge = pd.Series(perturbed_predictions_ridge,
                                            index=y_test.index)
    plt.plot(perturbed_predictions_ridge, '--', label='Ridge', color='orange',
             alpha=0.6)

    if experiment_num == 0: plt.legend(loc='best')

plt.title("Spread of ML Predictions With Perturbed Predictors")
plt.ylabel("Nino 3.4 Index")
plt.show()
```

**Note on Physics-Informed Neural Networks (PINNs):**
The tutorial briefly touches upon PINNs, which incorporate physical laws (e.g., conservation of mass and momentum) directly into the loss function. This approach aims to ensure physical consistency and improve cost-efficiency in training, bridging the gap between pure data-driven models and physical simulations.

---

## 4. CNN for ENSO Forecasting

Convolutional Neural Networks (CNNs) are advantageous here as they can leverage the spatial structure of the 2D input data. Consequently, the data preprocessing steps differ slightly from those used in linear regression; the data is not flattened immediately, but kept in `(Lat, Lon)` format.


**Model Architecture**
The implemented CNN follows a standard architecture. Given the input dimensions (Lat: 180, Lon: 360), the dimensionality of the subsequent linear layers is derived using standard calculations for kernel size, stride, padding, and pooling.

```python
class CNN(nn.Module):
    def __init__(self, num_input_time_steps=1, print_feature_dimension=False):
        """
        inputs
        -------
            num_input_time_steps        (int) : the number of input time
                                                steps in the predictor
            print_feature_dimension    (bool) : whether or not to print
                                                out the dimension of the features
                                                extracted from the conv layers
        """
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(num_input_time_steps, 6, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.print_layer = Print()

        #TIP: print out the dimension of the extracted features from
        #the conv layers for setting the dimension of the linear layer!
        #Using the print_layer, we find that the dimensions are
        #(batch_size, 16, 42, 87)
        self.fc1 = nn.Linear(16 * 42 * 87, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)
        self.print_feature_dimension = print_feature_dimension

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        if self.print_feature_dimension:
          x = self.print_layer(x)
        x = x.view(-1, 16 * 42 * 87)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
```

**Performance Observations**
Even with a reduced parameter count, the model exhibited decay in training loss alongside fluctuations in test loss, indicating potential overfitting. However, despite these signs, the correlation between predictions and ground truth remained relatively high.

---

## 5. Physical Interpretability and Future Goals

The tutorial explores interpretability by visualizing the learned weights. The following code maps the flattened weights back to their spatial dimensions to observe their geographical distribution.

```python
#The weights have shape (64800,) because the 2D (lat x lon) field
#has been flattened into a 1D vector in order to perform linear regression
#To visualize the weights, we set them back into (lat x lon) field
#to see the spatial distribution of the weights
regr_1_month_lr_weights = regr_1_month_lr.coef_.reshape((180,360))

plt.imshow(regr_1_month_lr_weights, cmap='bwr', vmin=-0.004, vmax=0.004)
cb = plt.colorbar()
cb.set_label("Learned Linear Regression Weights")
```

### Advanced Interpretability for CNNs
While visualizing weights works well for Linear Regression, it is often insufficient for CNNs due to non-linear activations. To overcome the "black box" nature of Deep Neural Networks, we can utilize advanced attribution methods:

1.  **Integrated Gradients (IG):** IG calculates the average gradient of the model's output as the input varies from a baseline (e.g., zero anomalies) to the actual input. This is particularly effective for climate data as it avoids saturation issues and highlights specific regions (e.g., the Eastern Pacific warm pool) that drove the prediction.
2.  **Gradient SHAP:** This method assigns a "contribution score" to every pixel, indicating not just sensitivity, but the magnitude and direction of the impact (e.g., "This pixel increased the Niño prediction by 0.2").
3.  **Occlusion Sensitivity:** Systematically masking parts of the map to observe prediction drops. This helps verify if the model is focusing on physically relevant regions or spurious correlations.

### Implementation with Captum
To implement Integrated Gradients on the CNN architecture described above, we can use the `Captum` library. Below is an example of how this can be implemented to visualize which ocean regions contributed most to a specific prediction.

```python
import torch
import matplotlib.pyplot as plt
from captum.attr import IntegratedGradients

# 1. Initialize the model and Integrated Gradients
model = CNN(num_input_time_steps=1)
model.eval() # Set to evaluation mode
ig = IntegratedGradients(model)

# 2. Prepare Input
# Assume 'input_tensor' is a single sample from our test set
# Shape: (1, 1, 180, 360) -> (Batch, Time, Lat, Lon)
input_tensor = torch.tensor(X_test[0:1], dtype=torch.float32, requires_grad=True)

# 3. Compute Attributes
# We use a baseline of all zeros (neutral temperature anomaly)
baseline = torch.zeros_like(input_tensor)
attributions, delta = ig.attribute(input_tensor, baseline, return_convergence_delta=True)

# 4. Visualize
# Reshape to 2D for plotting (Lat, Lon)
attr_map = attributions.squeeze().detach().numpy()

plt.figure(figsize=(10, 5))
plt.imshow(attr_map, cmap='RdBu_r', vmin=-np.max(np.abs(attr_map)), vmax=np.max(np.abs(attr_map)))
plt.colorbar(label="Attribution Score (Contribution to ENSO Prediction)")
plt.title("Integrated Gradients: Which Regions Drove the Forecast?")
plt.show()
```

By applying these techniques, we can move beyond simple correlation scores and verify if the model is learning valid physical dynamics, such as Kelvin waves or Bjerknes feedback.
