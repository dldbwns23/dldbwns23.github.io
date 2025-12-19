---
layout: post
title: "CCAI Quantus XAI Review"
date: 2025-12-19 14:13:00 +0900
tags: [CCAI, Visualization, Python, ML/DL]
---

# Review: Quantus x Climate – Applying Explainable AI Evaluation in Climate Science

## 0. Overview
This post reviews key insights from the Climate Change AI tutorial **"Quantus x Climate: Applying Explainable AI Evaluation in Climate Science."**

While neural networks are increasingly used in climate science for tasks like prediction and downscaling, their "black box" nature remains a barrier to trust. This tutorial demonstrates how to apply **Quantus**, an evaluation toolkit, to quantitatively measure the quality of Explainable AI (XAI) methods in the context of climate classification problems.

For full implementation details, refer to the [GitHub repository](https://github.com/philine-bommer/Climate_X_Quantus) or the accompanying Colab Notebook.

---

## 1. Reference
**Bommer, Philine et al.** *Quantus x Climate: Applying Explainable AI Evaluation in Climate Science*
Available at: [https://climatechange.ai/tutorials](https://climatechange.ai/tutorials?search=id:quantus-x-climate)

---

## 2. Dataset and Network Description

### Dataset: CESM1 Large Ensemble
The tutorial utilizes data simulated by the **Community Earth System Model version 1 (CESM1)**.
* **Source:** A 40-member large ensemble simulation (Hurrell et al., 2013; Kay et al., 2015).
* **Configuration:** Equal external forcings across members but differing atmospheric initial conditions.
* **Timeframe:** 1920–2080. Historical forcings are used for 1920–2005, followed by RCP 8.5 for subsequent years.
* **Variable:** Annual averages of monthly 2-meter air temperature (T2m) maps.

**Preprocessing:**
The data is pre-processed into batches of $N=155$ images. The input temperature maps are on a longitude-latitude grid ($144 \times 95$) with $1.9^\circ$ latitude and $2.5^\circ$ longitude sampling.

### Network Architecture
The model presented is a Convolutional Neural Network (CNN) designed to predict the decade of a given temperature map.
* **Layer 1:** 2D-convolution (6x6 window, stride 2).
* **Layer 2:** Max-pooling (2x2 window).
* **Layer 3:** Dense layer with $L^2$-regularization.
* **Output:** Softmax layer producing a probability vector for 20 classes (decades from 1900–2100).

The following code loads the data, including segmentation maps for the North Atlantic (NA) region ($10-80^\circ$ W, $20-60^\circ$ N), which are used later for regional evaluation.

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

Samples from the years 1940, 1990, 2040, and 2074 are visualized to inspect the temperature anomaly patterns.

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



---

## 3. Generate Explanations

To interpret the model's predictions, the tutorial employs **Quantus** to generate heatmaps using several prominent XAI methods. In the context of climate science, these explanations serve to validate whether the network is learning physical drivers (e.g., ENSO, North Atlantic Oscillation) or relying on spurious correlations.

The tutorial assesses the following methods:

* **Vanilla Gradients:** Calculates the gradient of the output class score with respect to the input image. It highlights pixels that, if changed, would most affect the classification.
* **Integrated Gradients:** Addresses the problem of gradient saturation by accumulating gradients along a path from a baseline (usually a black image) to the input image.
* **SmoothGrad:** Reduces noise in saliency maps by adding Gaussian noise to the input image multiple times, calculating gradients for each, and averaging the results.
* **GradientsInput (Input $\times$ Gradient):** Computes the element-wise product of the input image and its gradient. This accounts for the magnitude of the input feature itself.
* **Occlusion Sensitivity:** A perturbation-based method that systematically masks (occludes) different parts of the image and measures the drop in the model's confidence.
* **GradCAM:** Uses the gradients of the target class flowing into the final convolutional layer to produce a coarse localization map highlighting important regions.

```python
import quantus

# We load the available XAI methods with tensorflow.
quantus.AVAILABLE_XAI_METHODS_TF

# View the XAI methods available for PyTorch users.
quantus.AVAILABLE_XAI_METHODS_CAPTUM
```

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
                            **{"method": method, **kwargs})

    # Normalise for GradCAM.
    if a_batch.min() == 0:
        explanations[method] = a_batch/a_batch.max(axis=(1,2), keepdims=True)
    else:
        # If not normalized, normalize by hand to comparable values [0,1].
        explanations[method] = np.abs((a_batch - a_batch.min(axis=(1,2), keepdims=True))/(a_batch.max(axis=(1,2), keepdims=True) -a_batch.min(axis=(1,2), keepdims=True)))

    print(f"{method} - {a_batch.shape}")
```

### Visual Comparison of Explanations
The tutorial visualizes normalized explanation maps (scaled 0 to 1). This demonstrates the "Rashomon Effect" in XAI, where different methods provide vastly different interpretations of feature importance for the same prediction.

* **Pixel-wise methods** (e.g., GradientsInput) show fine-grained details.
* **Patch-wise methods** (e.g., Occlusion, GradCAM) show broader regional importance.
* **Disagreement:** Significant incongruences exist regarding which regions are "most important" (dark red patches), complicating visual selection of the "best" method.

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


### Case Study: North Atlantic Warming Hole
To better assess the methods, the authors zoom in on the North Atlantic (NA) region.
* **Physical Context:** This region exhibits a known signal—the "warming hole" or cooling patch—which evolves from a warming trend in the 20th century to a cooling trend in the 21st century under climate change (Labe and Barnes, 2021).
* **Validation Goal:** The aim is to verify if the network assigns high importance to this physically relevant feature.

Visual inspection reveals strong discrepancies. For instance, GradCAM and Integrated Gradients show different centers of mass for importance. While SmoothGrad and Vanilla Gradients overlap, the variation in relevance assignment makes it difficult to deduce a conclusive assessment purely through visualization.

```python
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


---

## 4. XAI Evaluation with Quantus

Since visual inspection is subjective, the tutorial uses Quantus to quantify explanation quality across five categories. Note that the arrows indicate the desired direction (↑ higher is better, ↓ lower is better).

* **Faithfulness (↑):** Measures how truly the explanation reflects the model's predictive behavior. If features marked as "important" are perturbed, the model's prediction should drop significantly.
* **Robustness (↓):** Assesses stability. If slight noise is applied to the input (without changing the prediction), the explanation should not change drastically.
* **Randomisation (↑ / ↓):** Tests the dependence on model parameters. If the model weights are randomized, the explanation should degrade (high correlation with the original explanation implies the method does not rely on learned weights).
* **Localisation (↑):** Checks if the evidence is centered around a known region of interest (e.g., using a bounding box or segmentation mask).
* **Complexity (↓):** Captures conciseness. Good explanations should ideally use a sparse set of features to explain the prediction (Occam's razor).

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

### Results Analysis
The scores are aggregated to rank the methods.

```python
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


Plotting these results on a spider plot allows for visualization of the trade-offs:


**Conclusion:**
For this specific network and task, the authors prioritize an explanation that is **robust** to internal variability (given the chaotic nature of climate data), **complex** (capturing significant features), and **faithful**.
* While SmoothGrad and GradientsInput both perform well across criteria, **GradientsInput** demonstrates superior faithfulness and robustness.
* Consequently, GradientsInput is selected as the most reliable method for analyzing regional importance in this study.

---

## 5. Limitations

While Quantus provides a rigorous framework, XAI evaluation in climate science faces inherent challenges:

* **Lack of Ground Truth:** Unlike object detection (where a cat is objectively a cat), climate drivers are complex. Metrics assess *properties* of explanations (like stability), not necessarily their physical correctness.
* **Out-of-Distribution (OOD) Issues:** Many metrics (e.g., faithfulness) rely on perturbing inputs (masking pixels). This can create unrealistic data samples that the model never saw during training, potentially skewing results.
* **Statistical Validity:** High scores in Quantus do not automatically guarantee theoretical soundness.

**Takeaway:** Quantitative evaluation should always be complemented by domain knowledge and theoretical verification of the chosen XAI method.
