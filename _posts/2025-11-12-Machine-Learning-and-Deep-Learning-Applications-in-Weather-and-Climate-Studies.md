---
layout: post
title: "Machine Learning and Deep Learning Applications in Weather and Climate Studies"
date: 2025-11-12 14:48:00 +0900
tags: [Article]
---

## 1. Introduction

Traditional Numerical Weather Prediction (NWP) models face challenges in weather forecasting because the complex, non-linear equations involved require significant computational resources, and their accuracy decreases due to the numerous inherent assumptions made.

In contrast, **Artificial Neural Networks (ANN)**, particularly **Machine Learning (ML)** and **Deep Learning (DL)**, have emerged as potent tools.

* **ANNs** are capable of approximating any non-linear function.
* **ML** and **DL** applications have found importance in time series problems, especially those characterized by intricate correlations, such as weather and climate prediction.
* When the behavior of a system is predominantly influenced by **spatial or temporal context**, traditional ML approaches may not be optimal. In these cases, **DL architectures** are more proficient at automatically extracting spatio-temporal features and are more effective in gaining comprehensive insights into such systems.

---

## 2. Weather and Climate Prediction

Historically, weather and climate prediction was approached as a physical problem, with meteorologists dedicating efforts to enhance forecast accuracy through understanding associated processes. However, the scenario has shifted due to the explosion of multi-scale and multi-dimensional meteorological data, transforming the entire landscape into spatio-temporal challenges.

NWP models simulate atmospheric and ocean conditions by solving governing physical equations derived from current atmospheric states on a **discretized grid system**.

* A significant constraint is the **horizontal resolution** of global NWP models, which can be at most 10 kilometers while simulating the states of atmosphere and ocean, hindering the accurate representation of critical processes like cloud microphysics.
* These models compensate by using **parameterizations** to approximate the effects of these phenomena.
* **Data Assimilation (DA)** techniques are also adopted for accurate representation of the current state of the atmosphere, which is necessary for initializing the model.

In recent years, data-driven models have shown comparable or superior performance. Examples of these alternatives include:

* **Graph Neural Networks (GNN)**.
* **FourCastNet**, which uses a modified vision transformer.
* **Pangu-weather**, which incorporates a variation of the vision transformer architecture and has achieved superior performance compared to a high-resolution Integrated Forecast System (IFS).

A major challenge with these data-driven models is their **interpretability**. Unlike traditional physical models with well-defined equations, data-driven models often operate as **"black boxes,"** making it difficult to understand how they arrive at predictions. This lack of transparency has opened up a new area of research, focused on enhancing the interpretability of data-driven models, often referred to as explainable AI (XAI).

---

### 2.1 How NWP Works

NWP models are tools that help to simulate atmospheric and ocean conditions and predict the weather based on current conditions.

* **Data Preparation:** Observed datasets are utilized to feed the models. To address the unevenly spaced observations, DA techniques and objective analysis approaches are utilized, providing quality control and getting values at areas that the model's mathematical algorithms can use (often an equally spaced grid). This data is then utilized as an initiation point for the model's prediction.
* **Calculation:** **Primitive equations** are a set of nonlinear equations used to anticipate atmospheric physics and behavior. After these equations have been initialized using the gridded data, the rates of change are determined, which are used to anticipate the future state of the atmosphere.
* **Time Stepping:** The equations are then applied to this newly formulated atmospheric condition to calculate new rates of change, which are subsequently used to forecast the atmosphere at a later date and/or time. This time stepping procedure is performed until the response achieves the desired predicted time. The distance between the points on the computational grid determines the duration of the time step in the model, which is selected to offer numerical stability.

The NWP is an intricate non-linear system, where physical forcing dominates the dynamics within the sub-grid scale, hence necessitating the physical **parameterization schemes**. Since the NWP model uses the primitive non-linear equations, this could lead to problems and uncertainties, such as:
* The approximation of initial conditions incorporated while compiling the model.
* The process of data assimilation.
* An incomplete understanding of the complex atmospheric processes, which could unavoidably introduce errors.
* It is **computationally expensive**, producing terabytes of simulation results.

---

### 2.2 How ML/DL Works

Deep Learning (DL) has seen a massive rise in popularity in almost all widely researched subjects in science, including weather forecasting, climate studies, air quality research and prediction, land use and land cover (LULC) related studies, and remote sensing applications.

* **Feature Extraction:** Such models can automatically extract relevant features from large and complex weather and climate datasets, which helps in identifying important patterns and relationships between atmospheric variables, like temperature, humidity, pressure, and wind speed.
* **Handling Complexity:** For instance, a **Deep Convolutional Neural Network (DCNN)** framework coupled with a Bayesian-based hyper-parameter optimization scheme was developed to detect patterns and increase the accuracy of forecasts.
* **Spatio-Temporal Data:** DL models, in particular, excel in capturing spatial dependencies and temporal patterns (e.g., **ConvLSTM**), making them suitable for such tasks.
* **Real-Time Capability:** These algorithms can effectively model non-linear relationships between different meteorological variables and can process large volumes of data quickly, making them suitable for real-time weather forecasting applications, leading to more accurate and timely predictions.

---

## 3.1 Applications Pertaining to Urban Studies

The urban-atmosphere interaction has its effect across various scales, from smaller scale (such as urban environment and individual buildings) to regional-scale phenomena (such as regional climate change). Urban features influence the atmospheric flow and microclimates, thereby altering the transfer, distribution, and deposition of atmospheric pollutants within the urban areas.

For mapping and analyzing urban features:
* **Convolutional Neural Network (CNN)** is the most popularly used algorithm in LULC classification due to its inherent capability in handling spatiotemporal imageries.
* **Deep Belief Network (DBN)** can be used to estimate Land Surface Temperature (LST) from satellite data.
* **Temporal CNN (Temp-CNN)** algorithm can classify time series satellite imageries and is usually more precise than Random Forest (RF) or other Recurrent Neural Networks (RNNs) for this task.
* DL-based algorithms (e.g., **LSTM and U-Net**) have been implemented successfully in studies relating to LULC classification, image segmentation, object-based image analysis, and object detection.

---

## 3.2 Air Pollution Studies

* **Prediction Models:** A **Legendre Neural Network (NN)** for prediction of air pollution parameters was devised.
* **Time Series Forecasting:** One-dimensional convnets and **Bidirectional Gated Recurrent Units (Bi-GRUs)** were used for time series forecasting of $\text{PM}_{2.5}$ concentration. Bi-GRUs process time series both chronologically and anti-chronologically, capturing patterns that one-direction GRUs may miss, and thereby increasing the time series feature learning capabilities.
* **Sophisticated Data:** Generative Adversarial Networks (GANs) were found to be particularly effective at creating content, producing better results when the model is trained properly in the presence of sophisticated data synthesizers.
* **DL vs. Baseline Models:** A comparison of the performance of various DL models with ARIMA suggested that DL models can be utilized to predict air pollution well. An architecture based on **LSTM networks and CNNs** performed admirably in accurately forecasting $\text{PM}_{10}$ concentrations in the short run and was better than the ARIMA baseline model.

---

## 3.3 Extreme Weather Events

Among the extreme weather events, ML finds its application in cases of **tropical cyclones (TCs)**, **thunderstorms**, and **heavy rainfall events**. TC forecasts are mainly focused on predicting the track, intensity, and landfall characteristics.

### Tropical Cyclones (TCs)

* **Genesis Prediction:** For predicting genesis, ML uses meteorological observations to envisage whether a depression will evolve into a TC and its frequency, making it a classification and regression task. Mainly used algorithms include **DTs, Logistic Regression, SVM, RF, MLP, and CNNs**. In recent studies, genesis forecasting is denoted as a classification task for envisaging whether TC symptoms will evolve into TCs. The Peter-Clark algorithm, SVR, and MLP can be utilized.
* **Track Forecasting:** Both ML and DL based models can be used to forecast the position of TCs using **DTs, MLP, RNNs, and CNNs**. **CNN-LSTM frameworks** can also predict spatiotemporal changes/formation of TCs. Path prediction was attempted using a **matrix neural network (MNN)-based** and **RNN-based** models to generate effective results compared to other methods. A fusion neural network model combined past trajectory data and re-analyzed images (3D wind and pressure) while adopting a moving frame of reference to track the storm for 24 hours. The **GRU\_CNN** framework, which includes three additional environmental factors (steering flow, SST, and geopotential height) along with historical trajectory data, was developed and outperformed conventional forecast results for short-term track forecasting.
* **Intensity/RI:** CNNs are mainly used to estimate intensity using satellite images. Hybrid networks of **CNN and LSTM** further improve results. The problem of rapid intensification (RI), which hinders intensity prediction, is treated as a classification or regression problem and can be predicted using **DTs, SVM, and RNN**.
* **NWP Integration:** ML can also be used for advancing the parameterization schemes in NWP models.

### Thunderstorms and Rainfall

* **Thunderstorm Prediction:** Recurrent-convolutional DL models, U-Nets and transformers have been used for nowcasting thunderstorm hazards. As ML continues to make strides in thunderstorm prediction, ongoing research efforts are essential to address challenges such as the need for extensive and high-quality training datasets and the interpretability of complex models.
* **Extreme Rainfall:** In the world of weather forecasting, extreme rainfall prediction has never been easy due to its chaotic nature. An **Anomaly Frequency Method (AFM)-SVM framework** and the **Affinity Propagation algorithm** along with **K-means clustering** have been adopted.

---

## 3.4 Regional-Scale Weather and Climate

These models have the benefit of making predictions based solely on historical data and employing a physics-free approach.

* **ML/Statistical Algorithms:** Algorithms utilized include **DT**, **Arbitrary DT**, **RF**, **SVM**, **Naive Bayes classification**, **SVR**, **Lasso Regression**, and **Multiple Linear Regression (MLR)**.
* **DL Architectures:** Employing a **bidirectional LSTM (BILSTM)** model alongside adaptive learning significantly enhances temperature forecasting accuracy. **CNNs on a cubed sphere grid** can produce global forecasts in an ensemble Deep Learning Weather Prediction (DLWP) system. A hybrid model converted spherical global data into a cubed sphere grid by using a modified **DLWP-CS** architecture to improve short-range global precipitation forecasts.
* **Other Time Series Models:** **ARIMA** can forecast future climate (rainfall, maximum, and minimum temperature). **Auto encoder and MLP** have been used for predicting the accumulated rainfall for the next day, by using data of previous days. **Reservoir Computing (RC)** is a supervised learning strategy for RNNs that employs **Echo State Networks (ESNs)** and **Deep ESNs**.

### Data Downscaling and Climatological Events
* **Downscaling:** Downscaling of low-resolution data into high-resolution observation data has recently become a popular approach. DL approaches used over the Indian region include **Super-resolution CNN (SRCNN), stacked-SRCNN, DeepSD, SR-GAN, ConvLSTM, and U-Net**.
* **Climatological Analysis:** DL frameworks have been used for the analysis and forecasting of extreme events like floods (using **LSTM combined with CNN**), droughts (using **conjugated CNN-LSTM** and **LSTM**), and El Niño (using a **CNN framework**). Forecasting the Indian summer monsoon rainfall (ISMR) has also been attempted using frameworks like **ConvLSTM** and a **stacked auto-encoder and ensemble regression model**.

---

## 4. Overview

The data-driven methods detailed have proven capable of handling non-linear interactions and overcoming some traditional limitations of NWP models, particularly in short-term weather prediction and forecasting extreme events.

This review details on the application of ML/DL to **extreme weather events** (e.g., TCs, thunderstorms, heavy rainfall), but dedicated less extensive discussion to long-term **climate change** applications as a distinct area of study. While the importance of climate forecasting, decadal trends like global warming, and downscaling global climate models were mentioned, the depth of application examples focused more on weather and immediate hazards. This confirms that the detailed application of ML/DL in the moderate to long-term climate change area may be a fruitful avenue for next projects, aiming to fill this application gap.

I agree with the document's emphasis on the need for continued research into the **interpretability** of complex ML/DL models. Given that these models often operate as "black boxes", enhancing transparency (Explainable AI/XAI) is essential, not just for thunderstorm prediction, but across all applications to ensure confidence in climate and weather insights.

---

## 5. Reference

Machine Learning and Deep Learning Applications in Weather and Climate Studies: A Systematic Review.
