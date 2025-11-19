<img width="550" height="157" alt="image" src="https://github.com/user-attachments/assets/ba91c4e1-1522-487a-a2c8-861e01f4a88d" />---
layout: post
title: "Interpretable ML for Extreme Events Detection"
date: 2025-11-19 13:33:00 +0900
tags: [Article, D&A, ML/DL]
---

# 1. Introduction

Traditional standardized drought indices, such as SPI and SPEI, frequently fail to accurately detect drought impacts because they focus on specific drivers (e.g., precipitation and evapotranspiration) without accounting for the complex interactions that eventually generate these impacts. Alternatively, indices derived from satellite data, such as the Vegetation Health Index (VHI), allow for the direct observation of vegetation status, making them suitable proxies for drought conditions. However, these indices are limited to real-time observation, preventing the prediction of events weeks or months in advance required for timely anticipatory operations.

In this study, a data-driven workflow is applied to the Po River basin in Italy. The authors utilize the remotely sensed VHI as a target proxy for the state of vegetation impacted by drought. A primary objective of this study is ensuring the interpretability of each step in the machine learning (ML) workflow. This work represents the first empirical validation of recent dimensionality reduction algorithms designed to generate interpretable, spatially aggregated features. Filter feature selection via Conditional Mutual Information (CMI) is employed to identify relevant and non-redundant features as candidate drivers. Finally, linear supervised learning models are trained, including a multi-task setting, to define a new data-driven, impact-based drought index. The preservation of index interpretability and the physical meaning of the results constitute the main contribution to the field, bridging the gap between advanced black-box algorithms and physical applicability.

# 2. Data

The VHI is a satellite index available in eight-day aggregations with a 231m spatial resolution. The target for each sub-basin is the eight-day average VHI value of cultivable areas. The initial set of candidate drivers consists of meteorological features:

* **Daily temperature and cumulative precipitation:** These are the core variables of traditional drought indices.
* **Daily snow depth:** Included for sub-basins with mountainous areas, as snowmelt can mitigate drought.
* **Lake-related features:** Water level, inflow, and release (specifically for Lakes Como, Iseo, Maggiore, and Lugano).

These features were aggregated into 8-day averages to match the temporal resolution of the VHI target. To account for long-term impacts and delayed effects, variables were averaged over preceding blocks of 0, 1, 4, 8, 12, 16, and 24 weeks. Furthermore, to address cyclo-stationarity and remove seasonality, the average value (based on the training set) for each feature associated with a specific week was subtracted from the weekly measurement.

The study considers original grid points at different locations as individual features, resulting in 991 distinct latitude-longitude grid points across the 10 sub-basins. With 14 temporal aggregations applied to these points, the total initial feature set comprises $14 \times 991$ features. To validate results, a benchmark experiment was conducted that replaces the dimensionality reduction step with mean feature values calculated directly for each sub-basin. This results in $14 \times 10$ candidate drivers for the benchmark.

The study period spans from January 2001 to December 2019, totaling 867 samples, divided as follows:
* **Training:** 2001–2009 (411 samples)
* **Validation:** 2010–2014 (228 samples)
* **Test:** 2015–2019 (228 samples)

# 3. Extraction of Interpretable Relevant Features

### Dimensionality Reduction
The study employs LinCFA, NonLinCFA, and GenLinCFA, which are algorithms designed to identify groups of features that are convenient, in terms of Mean Squared Error (MSE), to aggregate via their mean. These algorithms iteratively compare pairs of features and aggregate them based on a correlation threshold.

* **Linear Correlated Features Aggregation (LinCFA):** Assumes linear relationships between the features and the target.
* **Non-Linear Correlated Features Aggregation (NonLinCFA):** An extension of LinCFA designed assuming the target is a nonlinear function of the features with additive Gaussian noise.
* **Generalized Linear Correlated Features Aggregation (GenLinCFA):** Assumes the distribution of the target belongs to the canonical exponential family, making it applicable in classification settings.

The resulting reduced features are averages across neighboring grid points within the same region for the same variable. For example, applying NonLinCFA to 1-week aggregated cumulative precipitation revealed distinct sub-regions, extracting 17 features from 172 original ones.

### Feature Selection
Feature selection based on Conditional Mutual Information (CMI) was subsequently applied to identify relevant and non-redundant drivers. A forward CMI selection was applied to each sub-basin using the features obtained from dimensionality reduction. In both the main method and the benchmark experiment, the five most relevant features were identified for each sub-basin based on validation set performance.

The CMI-based approach performs comparably to or better than wrapper methods, with the distinct advantage of being model-independent. This independence preserves the interpretability of the selected candidate drivers.
* **Model-Independent (Filter/CMI):** Filters features based on pure statistical relevance (e.g., "does variable X share information with variable Y?"). If a feature is selected, it is because it actually contains unique information about the drought. If a complex variable is redundant (shares information with simple variables), it is avoided in favor of the simple, meaningful ones. This ensures drivers are physically meaningful and universal.
* **Model-Dependent (Wrapper):** Relies on a specific model (e.g., Random Forest) to pick features. This biases selection toward the model's internal logic, potentially selecting features that maximize a score through "hacking" rather than physical causality, and may fail if the model type changes.

# 4. Identification of Data-Driven, Impact-Based Drought Indices

The features selected via the approach described above are used as inputs for linear supervised learning models to reconstruct the target VHI. A multi-task variation was also explored, grouping the ten sub-basins into three clusters based on VHI correlations and hydrological similarity. In the multi-task setting, a single linear model is trained for each cluster, utilizing input features from all basins within the cluster and a bias term (via one-hot encoding) to associate samples with their specific sub-basin. This approach increases sample size and reduces the number of models required.

Results indicate improvements with the proposed workflow, particularly in the multi-task setting, with $R^2$ scores generally above 0.2. While the addition of snow depth and lake-related features provided useful information, it did not significantly enhance regression accuracy, suggesting their contribution is not significantly complementary to precipitation and temperature data.

# 5. Conclusions

The primary contribution of this work extends beyond accurate prediction to the preservation of interpretability. The resulting indices are linear combinations of drivers selected from features extracted as averages of meteorological variables at neighboring locations. Consequently, these data-driven indices are physically interpretable and suitable for analysis by domain experts. Future work aims to test this pipeline on a European scale.

{% raw %}
<img src="/images/Interpretable_ML_for_Extreme_Events_Detection/Figure_1.png" alt="Figure 1">
{% endraw %}


{% raw %}
<img src="/images/Interpretable_ML_for_Extreme_Events_Detection/Table_1.png" alt="Table 1">
{% endraw %}

# 6. Overview

Although the study characterizes a coefficient of determination ($R^2$) of approximately 0.3 as satisfactory, traditionally, this is often viewed as a weak result in predictive modeling. As such, there may be some reservation in fully accepting the efficacy of the model based solely on this metric. However, considering the inherent difficulty in predicting biological targets like VHI weeks in advance using only temperature and precipitation, the result is arguably fair. There is likely a significant opportunity for improvement by incorporating additional meteorological variables or higher-resolution data.

A particularly interesting aspect of this research is the methodology of segmenting sub-basins using dimensionality reduction and feature selection. This novel application of NonLinCFA to define spatial aggregations offers a fresh perspective on handling geospatial data. This method holds promise for application in other environmental projects where spatial heterogeneity is a challenge. Furthermore, the use of CMI filtering to ensure features remain interpretable and physically meaningful is a valuable takeaway for the broader field of interpretable machine learning.


# 7. Reference
Interpretable Machine Learning for Extreme Events detection: An application to droughts in the Po River Basin (Papers Track), Paolo Bonetti (Politecnico di Milano); Matteo Giuliani (Politecnico di Milano); Veronica Cardigliano (Politecnico di Milano); Alberto Maria Metelli (Politecnico di Milano); Marcello Restelli (Politecnico di Milano); Andrea Castelletti (Politecnico di Milano)
