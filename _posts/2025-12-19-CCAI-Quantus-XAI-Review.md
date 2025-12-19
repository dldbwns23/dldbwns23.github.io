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

## 3. Generate Explanations

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
* 

```python
# Generate several explanation methods with Quantus.
xai_methods ={"VanillaGradients": {},
              "IntegratedGradients": {},
              "SmoothGrad": {},
              "GradientsInput": {},
              "OcclusionSensitivity": {"window": (1, 5, 6)}, # window size for the occlusion procedure
              "GradCAM": {"gc_layer": "conv2d", "shape": (1, 95, 144)} # gc_layer - layer of explanation, shape - output shape (1, img. dim., img. dim.)
              }

explanations = {}
for method, kwargs in xai_methods.items():
    a_batch = quantus.explain(model=model,
                            inputs=x_batch[samples,:,:,:],
                            targets=y_batch[samples],
                            **{**{"method": method}, **kwargs})

    # Normalise for GradCAM.
    if a_batch.min() == 0:
        explanations[method] = a_batch/a_batch.max(axis=(1,2), keepdims=True)
    else:
        # If not normalized, normalize by hand to comparable values [0,1].
        explanations[method] = np.abs((a_batch - a_batch.min(axis=(1,2), keepdims=True))/(a_batch.max(axis=(1,2), keepdims=True) -a_batch.min(axis=(1,2), keepdims=True)))

    print(f"{method} - {a_batch.shape}")
```

```python
# Visualise explanations, demonstrating the unintelligibility of visual comparision.
y_pred_samples = pred_class[samples]

plt_kwrgs = {'nrows': len(explanations)+1,
             'ncols':len(samples),
             'font': 16,
             'xtext': -0.3,
             'figsize': (15,20),
             'cb_pos': [0.35, 0.05, 0.35, 0.05],
             'keys': list(xai_methods.keys()),
             'explanation': explanations}

plot_multiple_temperature_maps(samples, x_batch_samples, year_samples, y_batch_samples, y_pred_samples, latitude, longitude, **plt_kwrgs)
```
[Image2]

Visualization description The plotted maps of the first row are the inputs which we explain in the following rows. We use 6 methods and all explanations are normalized between  0  and  1  (white to dark red), whereas the input T2m temperature maps are standardized and plotted in the same range as the previous visulization (see colorbar of first plot visulizing the inputs with the prediction). Here, we see especially strong visual differences between the first  4  explanation methods and the last  2  which can be related to the difference between the patch-wise and the pixel-wise calculation used for GradCAM/OcclusionSensitivity and gradient-based methods respectively (see description above). But also, the first  4  rows display incongruences regarding the areas assigned highest importance(dark red patches). This figure already highlights how complicated a visual comparison and visual-based choice is and how different explanations might lead to different interpretation of the model decision.

Since in climate science, primary aims of XAI are to either validate that the network learned features which relate to previously established climate drivers and physical relationships, or to uncover new insights about the research question at hand, we want to provide you with an example.

The NA region includes contributing regional temperature signal, refered to as warming hole or cooling patch. The signal evolves from a warming region in the 20th century to a cooling patch throughout in 21st century in the course of climate change. The feature was established to contribute to the network prediction (see Labe and Barnes, 2021). To assess wether the network has learned physically relevant features, by which we could validate the network skill, we want to find out if the patterns of NA cooling patch will recieve high importance (increased values in the explanation).

``` python
# When zooming in at the North Atlantic region, we can establish that there is a disagreement in the explainable evidence/ feature importance between the different explanation methods. This may be confusing from a Climate AI practitioner's perspective.

# Transform temperature maps similar to cartopy PlateCarre:
long_tf = (longitude + 180) % 360 - 180

north_atl = [62, 83, 117, 138]
print('Plotting at latitude %s - %s, longitude %s - %s' % (latitude[north_atl[0]],latitude[north_atl[1]], long_tf[north_atl[2]], long_tf[north_atl[3]]))

# Define North Atlantic region data.
x_n_a_samples = x_batch_samples[:,north_atl[0]:north_atl[1],north_atl[2]:north_atl[3],:]
lon_n_a = long_tf[north_atl[2]:north_atl[3]]
lat_n_a = latitude[north_atl[0]:north_atl[1]]

explanations_n_a ={}
for method, values in explanations.items():
  explanations_n_a[method] = values[:,north_atl[0]:north_atl[1],north_atl[2]:north_atl[3]]

# Re-define plot kwargs.
plt_kwrgs['explanation'] = explanations_n_a
plt_kwrgs['figsize'] = (13,18)
plt_kwrgs['font'] = 14
plt_kwrgs['xtext'] = -0.4
plt_kwrgs['globe'] = False


# Plot.
plot_multiple_temperature_maps(samples, x_n_a_samples, year_samples, y_batch_samples, y_pred_samples, lat_n_a, lon_n_a , **plt_kwrgs)
```
[Image3]

Visual Comparison. The first row in the plot again shows the NA region and visuaizes the temperature pattern. In the previous cell we discussed that the patter changes from a warming region to a cooling region, here we can see that as the pattern evolves from the warmer colored central pattern indicating higher relative temperature to the cooler colored central pattern indicating lower relative temperatures. The different area plots in the rows below already show discreptencies between the assigned evidence in the NA region. We can identify especially strong differences, when comparing GradCAM and Integrated Gradients. Although, theoretically similar XAI methods like SmoothGrad and Vanilla Gradients overlap, some methods assign overall higher relvances in this region, making it hard to deduce a conclusive assessment of the importance of the region for the network prediction.

## 4. XAI Evaluation

In the following section, we showcase how to use Quantus to evaluate the different explanation methods under various explanation qualities — and their underlying metrics. In the following, we describe each of the categories briefly. A more in-depth description of each category, including an account of the underlying metrics, is documented in the repository. The direction of the arrow indicates whether higher or lower values are considered better (exceptions within each category exist, so please carefully read the docstrings of each individual metric prior to usage and/or interpretation).

Faithfulness (↑) quantifies to what extent explanations follow the predictive behaviour of the model, asserting that more important features affect model decisions more strongly e.g., (Bach et al., 2015), (Rong, Leemann, et al., 2022) and (Dasgupta et al., 2022).

Robustness (↓) measures to what extent explanations are stable when subject to slight perturbations in the input, assuming that the model output approximately stayed the same e.g., (Alvarez-Melis et al., 2018), (Yeh et al., 2019) and (Agarwal, et. al., 2022).

Randomisation (↑, ↓) tests to what extent explanations deteriorate as the data labels or the model, e.g., its parameters are increasingly randomised (Adebayo et. al., 2018) and (Sixt et al., 2020).

Localisation (↑) tests if the explainable evidence is centred around a region of interest, which may be defined around an object by a bounding box, a segmentation mask or a cell within a grid e.g., (Zhang et al., 2018), (Arras et al., 2021) and (Arias et al., 2022).

Complexity (↓) captures to what extent explanations are concise, i.e., that few features are used to explain a model prediction e.g., (Chalasani et al., 2020) and (Bhatt et al., 2020).

```python
# Initialise the Quantus evaluation metrics.
metrics = {
    "Robustness": quantus.AvgSensitivity(
        nr_samples=2,
        lower_bound=0.2,
        norm_numerator=quantus.norm_func.fro_norm,
        norm_denominator=quantus.norm_func.fro_norm,
        perturb_func=quantus.perturb_func.uniform_noise,
        similarity_func=quantus.similarity_func.difference,
        abs=True,
        normalise=False,
        aggregate_func=np.mean,
        return_aggregate=True,
        disable_warnings=True,
    ),
    "Faithfulness": quantus.FaithfulnessCorrelation(
        nr_runs=10,
        subset_size=224,
        perturb_baseline="black",
        perturb_func=quantus.baseline_replacement_by_indices,
        similarity_func=quantus.similarity_func.correlation_pearson,
        abs=True,
        normalise=False,
        aggregate_func=np.mean,
        return_aggregate=True,
        disable_warnings=True,
    ),
    "Localisation": quantus.RelevanceRankAccuracy(
        abs=True,
        normalise=False,
        aggregate_func=np.mean,
        return_aggregate=True,
        disable_warnings=True,
    ),
    "Complexity": quantus.Sparseness(
        abs=True,
        normalise=False,
        aggregate_func=np.mean,
        return_aggregate=True,
        disable_warnings=True,
    ),
    "Randomisation": quantus.ModelParameterRandomisation(
        layer_order="independent",
        similarity_func=quantus.ssim,
        return_sample_correlation=True,
        abs=True,
        normalise=False,
        aggregate_func=np.mean,
        return_aggregate=True,
        disable_warnings=True,
    ),
}
```

```python
# Load evaluation indicies.
idx = np.load('index.npz', allow_pickle=True)['idx']
nr_test_samples = 30

# Run full quantification analysis!
results = {method : {} for method in xai_methods}

for method, kwargs in xai_methods.items():
    for metric, metric_func in metrics.items():

        print(f"Evaluating the {metric} of {method} method...")

        # Get scores and append results.
        scores = metric_func(
            model=model,
            x_batch=x_batch[idx[:nr_test_samples]],
            y_batch=y_batch[idx[:nr_test_samples]],
            a_batch=None,
            s_batch=s_batch[idx[:nr_test_samples]],
            explain_func=quantus.explain,
            explain_func_kwargs={
                **{"method": method,},
                **kwargs
            },
        )
        results[method][metric] = scores
```

``` python
# Postprocessing of scores: to get how the different explanation methods rank across criteria.
results_agg = {}
for method in xai_methods:
    results_agg[method] = {}
    for metric, metric_func in metrics.items():
        results_agg[method][metric] = np.mean(results[method][metric])

df = pd.DataFrame.from_dict(results_agg)
df = df.T.abs()
df
```
[Image4]


If we plot the result, we get
[Image5]

Given our intuition above, for our tutorial network task, we aim for an explanation that is robust towards internal variability in our data, displays significant features (complexity) without sacrificing faithful evidence, and captures the network parameter behavior (randomization).

Although the spyder plot, for completeness, still shows all 5 criteria, we consider only the area cover in faithfulness, robustness, complexity and randomization. That means we neglect the spread on the localisation axis.

As we can derive from the spyder plot as well as the Table of the ranks across all explanation methods oftentimes in practice explanation methods only rank highly in some criteria. Although the ideal case would be highest ranks across all our important criteria (full coverage in the spyder plot except for localisation), in this case both SmoothGrad and GradientsInput achieve higher ranks in most criteria we want to be fulfilled for our network task. However, GradientsInput presents more faithful and more robust explanations, which is important to analyze if a region of high importance is stable and faithfully displayed in our explanation. Thus, we choose this explanation for our task.



## 4. Limitations
XAI evaluation faces certain limitations due to the absence of a reliable ground-truth, which means the evaluation metrics provided can only assess crucial properties that a valid explanation must possess, and cannot provide a complete validation. While the evaluation of XAI methods is a rapidly evolving field, the metrics offered by the Quantus library have certain limitations, such as relying on perturbing the input which may lead to the creation of out-of-distribution inputs. It should be noted that evaluating explanation methods using quantification analysis does not guarantee the theoretical soundness or statistical validity of the methods. Therefore, when using the Quantus library for XAI method selection, it is essential to supplement the results with theoretical considerations.
