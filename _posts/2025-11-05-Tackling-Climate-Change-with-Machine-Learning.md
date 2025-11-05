---
layout: post
title: "Tackling Climate Change with Machine Learning"
date: 2025-11-05 11:00:00 +0900
tags: [Article]
---


# Climate Prediction

The predominant predictive tools informing climate policy and risk assessment are climate models, known as **General Circulation Models (GCMs)** or **Earth System Models (ESMs)**. Opportunities for ML to advance climate prediction are significant, driven by recent trends: satellites are creating **petabytes of climate observation data**; massive climate modeling projects are generating **petabytes of simulated climate data**; and ML methods are becoming increasingly fast to train and run.

---

## 1. Data for Climate Models

ML techniques can resolve many challenging data-related problems in climate science. These include:
* Resolving "black box" problems, such as **sensor calibration**.
* Performing classification of observational data, for instance, classifying **crop cover** or identifying **pollutant sources in satellite imagery**.

Numerous authors have identified geoscience problems that would be aided by the development of **benchmark datasets**, citing examples such as EnviroNet and IS-GEO.

---

## 2. Accelerating Climate Models

Many climate prediction problems are **irremediably data-limited**; regardless of the number of sensors deployed, the Earth generates at most one year of new climate data per year. Existing climate models mitigate this limitation by relying heavily on **physical laws**, such as thermodynamics.

### 2.1 Clouds and Aerosols

Recent work has demonstrated how deep neural networks can be combined with existing thermodynamics knowledge to address the largest source of uncertainty in current climate models: **clouds**. Bright clouds cool the Earth, while dark clouds catch outgoing heat; these effects are controlled by small-scale processes like **cloud convection** and **atmospheric aerosols**. The physical models for these small-scale processes are far too computationally expensive to include in global models, but ML models are not. Replacing select climate model components with **neural network approximators** may thus improve both the cost and the accuracy of global climate models.

### 2.2 Ice Sheets and Sea Level Rise

Models of mass loss from the **Antarctic ice sheet** are highly uncertain, and models of the extent of Antarctic sea ice often do not match reality well. The most uncertain parts of these models—and thus the best targets for improvement—are **snow reflectivity**, **sea ice reflectivity**, **ocean heat mixing**, and **ice sheet grounding line migration rates**.

### 2.3 Working with Climate Models

**Pattern recognition** and **feature extraction techniques** could enable the identification of more useful connections in the climate system. Regression models could then quantify **non-linear relationships** between connected variables. For instance, Nowack et al. demonstrated that ozone concentrations could be computed as a function of temperature, rather than relying on complex physical transport laws, leading to considerable computational savings.

Making good **ensemble predictions** is an excellent ML problem. **Random forests** can be utilized to make high-resolution predictions from a mix of high- and low-resolution models.

---

## 3. Forecasting Extreme Events

Weather models are optimized to track the **rapid, chaotic changes of the atmosphere**; since these changes are fast, forecasts are made and tested daily. Climate models, conversely, are chaotic on short timescales, but their long-term trends are driven by the slow, predictable changes of the ocean, land, and ice. Consequently, climate model output can only be tested against long-term observations over years to decades.

Identifying **extreme events** in climate model output presents a unique challenge: it is a **classification problem** where the available datasets are strongly skewed because extreme events are, by definition, rare. Despite this, ML has been used successfully to classify some extreme weather events. Researchers have applied **deep learning** to classify, detect, and segment phenomena such as **cyclones**, **atmospheric rivers**, and **tornadoes**.

---

## 4. Local Forecasts

ML is widely used to create **local forecasts** from coarse $10$-$100 \text{ km}$ climate or weather model predictions. Various authors have attempted this downscaling using **support vector machines**, **autoencoders**, **Bayesian deep learning**, and **super-resolution convolutional neural networks**. Several groups are now focused on translating high-resolution climate forecasts into **risk scenarios**.

---

## 5. Discussion

Many components of large climate models can be replaced with ML models at lower computational costs. Recent research has explored **data-efficient techniques** for learning dynamical systems, including **physics-informed neural networks** and **neural ordinary differential equations**. ML approaches are being developed for a wide range of scientific modeling challenges, including **crash prediction**, **adaptive numerical meshing**, **uncertainty quantification**, and **performance optimization**.

---

## 6. Overview

This article highlights the application of machine learning to critical climate change problems. Although specific methodologies or algorithms are not detailed, the text inspires the components where ML can be applied. ML plays a crucial role in collecting and improving data quality, and in creating computationally effective and high-accuracy results from such data. Keeping these capabilities in mind, the next step should involve seeking out other components where ML can be applied to climate change, especially concerning **data and model improvement** and **extreme event forecasting**.
