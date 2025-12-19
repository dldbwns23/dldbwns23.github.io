---
layout: post
title: "CCAI Quantus XAI Review"
date: 2025-12-19 14:13:00 +0900
tags: [CCAI, Visualization, Python, ML/DL]
---

# Review: Quantus x Climate: Applying Explainable AI Evaluation in Climate Science

## 0. Overview
This post reviews and shares key insights from the tutorial **"Quantus x Climate: Applying Explainable AI Evaluation in Climate Science"** provided by Climate Change AI. This tutorial serves as an introduction to applying Quantus to measure scores of XAI methods for climate related classification problem.

For the full source code and implementation details, please refer to the [GitHub repository](https://github.com/philine-bommer/Climate_X_Quantus) or run the accompanying Colab Notebook.

---

## 1. Reference
Bommer, Philine and Hedstroem, Anna and Kretschmer, Marlene and Hoehne, Marina. *Quantus x Climate: Applying Explainable AI Evaluation in Climate Science*
Available at: [https://climatechange.ai/tutorials](https://climatechange.ai/tutorials?search=id:quantus-x-climate)

## 2. Dataset and Network Description

Using data simulated by a fully-coupled general climate model, Community Earth System Model version 1 (Hurrell et al., 2013). In particular, we focus on the  40  member larger ensemble climate data simulation, with equal external forcings across all ensemble members but differing atmospheric initial conditions (Kay et al., 2015). For the data from  1920  to  2080 , historical forcings are imposed for  1920−2005  and Representative Concentration Pathways 8.5 for the following years. We employ 2-m air temperature (T2m) temperature maps and take the annual average of monthly data

### Dataset
You can load the already pre-processed and prepared batch of the data containing  N=155  images (1 sample per correctly predicted year) from an .npz-file into the notebook, as uploaded on Kaggle (https://www.kaggle.com/datasets/philinelou/climatexquantusiclr2023?select=Batch_data.npz#network). Moreover, we provide a trained CNN (.tf-files) and according input temperature maps as images on a  h=144  by  v=95  longitude-latitude-grid with  1.9∘  sampling in latitude and  2.5∘  sampling in longitude. For the dataset, we include the associated class vectors (probability entry for each class) and the years of each temperature map in the input data.

### Network
This network consists of a 2D-convolutional layer (2dConv) with 6 x 6 window size and a stride of 2. The second layer includes a max-pooling layer with a 2 x 2 window size, followed by a dense layer with $L^2$-regularization and a softmax output layer.
The CNN maintains the longitude-latitude grid and assigns the annual temperature maps to classes based on their decade. Similar to Labe and Barnes, 2021 we define classes for the
full two centuries ($1900-2100$), resulting in a class vector of $20$ classes. Each class vector entry includes the probability of the input sample belonging to the decade of the class [^1].


In the following cells you can load and inspect the data. Alongside input images, labels and years of each image we also included a segmentation map of the NA region ( 10−80∘ W,  20−60∘ N), which we use for explanation evaluation in Section 5.).
```python

# Load the full data object.
data = np.load('dataset.npz', allow_pickle=True)

# Input images.
x_batch = data['x_batch'].swapaxes(1,3).swapaxes(1,2)

# Classification labels.
y_batch = data['y_batch']

# Segmentation masks for explanations.
s_batch = data['s_batch'].reshape(len(x_batch), 1, 95, 144)

# Years of the input images.
years_batch = data['years_batch']

# Longitude and latitudes.
latitude = data['wh'][0]
longitude = data['wh'][1]
```

Sample 1940, 1990, 2040, and 2074
```python
# Prepare random year samples.
year_s = np.array([1940, 1990, 2040, 2074])
samples = [np.where(years_batch == ys)[0][0] for ys in year_s]

y_batch_samples = y_batch[samples]
s_batch_samples = s_batch[samples, :]
year_samples = years_batch[samples]
x_batch_samples = x_batch[samples, :]


# Plot!
plt_kwrgs = {'indx_order': [len(samples)],
             'figsize': (20,15)}

plot_multiple_temperature_maps(samples, x_batch_samples, year_samples, y_batch_samples, [], latitude, longitude,**plt_kwrgs)
```

[image1]

For each decades from 1900 to 2100, each temperature map is labelled with the one of 20 decade classes.

After all predictions, import Quantus

``` python
import quantus

# We load the available XAI methods with tensorflow.
quantus.AVAILABLE_XAI_METHODS_TF

# View the XAI methods available for PyTorch users.
quantus.AVAILABLE_XAI_METHODS_CAPTUM
```

In this tutorial VanillaGradients, IntegratedGradients, SmoothGrad, GradientsInput, OcclusionSensitivity, and GradCAM are assessed. Short descriptions for each methods are followed.

* VanillaGradeints
* Inte

