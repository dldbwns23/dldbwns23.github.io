---
layout: post
title: "CCAI Forecasting ENSO Review"
date: 2025-11-26 20:54:00 +0900
tags: [CCAI, Visualization, Python, ML/DL]
---


# 0. Note
This post is a review and sharing of some insights from the "Forecasting the El Nino / Southern Oscillation with Machine Learning" in Climate Change AI Tutorials. Please visit the Github or run in Colab Notebook to look at the source code.

# 1. Reference

@misc{mahesh2024forecasting,
  title={Forecasting the El Nino/ Southern Oscillation with Machine Learning},
  author={Mahesh, Ankur and Hanna, Mel},
  year={2024},
  howpublished={\url{https://climatechange.ai/tutorials?search=id:forecasting-the-el-nino-southern-oscillation-with-machine-learning}}
}

Github Repository:
https://github.com/climatechange-ai-tutorials/seasonal-forecasting?tab=readme-ov-file

# 2. Data
Cobe Sea-Surface Temperature Dataset:: this is a dataset of historical sea surface temperatures form 1880 to 2018
Nino3.4 Indices: The Nino3.4 index measures the 3-month rolling average of equatorial Pacific Ocean temperature anomalies.
Input Variables
Sea surface temperature (K)

Output Variables
Nino3.4 index (K): this metric indicates the state of ENSO.

Training Set
Baseline training set: Training on 1980-1995 sea Surface temperatures and their corresponding Nino3.4 Index at lead times (1 month to 5 months)

Val Set
1997-2006 sea surface temperatures and their corresponding Nino3.4 Index at lead times.

Test Set:
Test on 2007-2017 sea surface temperatures and their corresponding Nino3.4 Index at lead times (1 month to 5 month).


# 3. Machine Learning in Basic ENSO Forecast

Code for basic linear regression and examples about overfitting and regularization can be easily understood. One thing to note, is that when handling time series data, random sampling is not enough. Even though the train set, validation set, and test set is randomly selected, we may encounter the overfitted result becausse time series data shares "memory" of previous data. That is, Each set should be selected far away from each time. However, there was no specific standard mentioned to choose such time gap between train set and test set. Maybe it's becasue those gap is problem-dependent.


```
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

```
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

Those two code cells above visualize the advantage of regulation, which are useful for both regulation and explainable AI.

There is a mention of Physics-Informed Neural Network (PINN) which incorporate physics into the model to aim consistency with  physical properties and cost-efficiency. Conservation of mass and conservation of momentum are some examples that can be applied to models.


# 4. CNN for ENSO Forecast

Since CNN takes advantage from being able to capture 2D input, data preprocessing step was slightly changed from linear regression. 

```
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

It is a basic construction for CNN. Considering shape of (lat, lon) = (180,360), the dimension of linear layer is easily understood using the kernel size, stride, padding and pooling.

Defining function for train_network can be useful resource for building other projects.

Even after reducing the number of parameters in CNN, there were still decay in train loss and fluctuation in the test loss, which implies overfitting. Even after those tendencies of overfitting, correlation was quite high.

Physical interpretability was another topic in this notebook. To visualized this, the code below was written.
```
#The weights have shape (64800,) because the 2D (lat x lon) field
#has been flattened into a 1D vector in order to perform linear regression
#To visualize the weights, we set them back into (lat x lon) field
#to see the spatial distribution of the weights
regr_1_month_lr_weights = regr_1_month_lr.coef_.reshape((180,360))

plt.imshow(regr_1_month_lr_weights, cmap='bwr', vmin=-0.004, vmax=0.004)
cb = plt.colorbar()
cb.set_label("Learned Linear Regression Weights")
```

Maybe occlusion sensitivity, gradient Shapley Additive Explanation and integrated gradient can be utilized for visulaizing physical interpretability.

