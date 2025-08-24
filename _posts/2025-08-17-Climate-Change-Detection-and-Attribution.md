---
layout: post
title: "Climate Change Detection and Attribution"
date: 2025-08-17 16:00:00 +0900
tags: [Climate Dynamics, D&A]
---

To detect the **anthropogenic fingerprints** within observed climate change, researchers typically compare results derived from climate models with real-world observations. This comparative approach aims to robustly establish that greenhouse gas emissions stemming from human activities are the primary contributors to global warming. Furthermore, the increasing frequency and intensity of extreme events, such as heavy rainfall and heatwaves, necessitate precise detection and attribution studies. Such efforts are crucial for understanding the causes of specific climate changes and for making more accurate predictions of future climate evolution based on these insights.

The scientific community identifies the anthropogenic fingerprint on climate change by systematically comparing climate model simulations with observational data. According to Figure 1, current global warming trends and patterns can only be accurately replicated by models when human-induced forcings (such as greenhouse gas emissions) are included in their simulations. Conversely, models that consider only natural forcings (like volcanic activity or solar variations) consistently fail to reproduce the observed warming, particularly the rapid and widespread changes seen since the mid-20th century. This discrepancy strongly indicates that human activities are the dominant driver of recent climate change.

{% raw %}
<img src="/images/Climate_Change_Detection_Attribution/Fig10-21-1.jpg" alt="Figure 1">
{% endraw %}
**Figure 1:** Observed and simulated changes in global land temperature and ocean heat content during the past 100 years and regional changes. (Adapted from IPCC, 2013)

---

## 1. Climate Model


---

### Momentum Equation for Zonal Wind (U-component)

This equation describes the rate of change of the **zonal (east-west) wind component** ($U$) over time. It accounts for various forces and processes influencing horizontal air movement within a spherical coordinate system.

$$
\frac{\partial U}{\partial t}+\frac{1}{a~cos^{2}\theta}\{U\frac{\partial U}{\partial\lambda}+V~cos~\theta\frac{\partial U}{\partial\theta}\}+\dot{\eta}\frac{\partial U}{\partial\eta}-fV+\frac{1}{a}\{\frac{\partial\phi}{\partial\lambda}+R_{dry}T_{v}\frac{\partial}{\partial\lambda}(ln~p)\}=P_{U}+K_{U}
$$

* $\frac{\partial U}{\partial t}$: Local rate of change of zonal wind.
* $\frac{1}{a~cos^{2}\theta}\{U\frac{\partial U}{\partial\lambda}+V~cos~\theta\frac{\partial U}{\partial\theta}\}$: Horizontal advection (transport) of zonal wind, where $a$ is the Earth's radius, $\theta$ is latitude, and $\lambda$ is longitude.
* $\dot{\eta}\frac{\partial U}{\partial\eta}$: Vertical advection of zonal wind, with $\dot{\eta}$ representing vertical velocity in the $\eta$-coordinate system.
* $fV$: Coriolis force term affecting zonal wind, where $f$ is the Coriolis parameter and $V$ is the meridional wind.
* $\frac{1}{a}\{\frac{\partial\phi}{\partial\lambda}+R_{dry}T_{v}\frac{\partial}{\partial\lambda}(ln~p)\}$: Pressure gradient force in the zonal direction, related to geopotential $\phi$, the gas constant for dry air $R_{dry}$, virtual temperature $T_v$, and pressure $p$.
* $P_U$: Source/sink term due to parameterized physical processes (e.g., friction, convection).
* $K_U$: Term representing horizontal diffusion.

---

### Momentum Equation for Meridional Wind (V-component)

This equation describes the rate of change of the **meridional (north-south) wind component** ($V$) over time, considering similar forces and processes as the U-component, along with terms related to the curvature of spherical coordinates.

$$
\frac{\partial V}{\partial t}+\frac{1}{a~cos^{2}\theta}\{U\frac{\partial V}{\partial\lambda}+V~cos~\theta\frac{\partial V}{\partial\theta}+sin~\theta(U^{2}+V^{2})\}+\dot{\eta}\frac{\partial V}{\partial\eta}+fU+\frac{cos~\theta}{a}\{\frac{\partial\phi}{\partial\theta}+R_{dry}T_{v}\frac{\partial}{\partial\theta}(ln~p)\}=P_{V}+K_{V}
$$

* $\frac{\partial V}{\partial t}$: Local rate of change of meridional wind.
* $\frac{1}{a~cos^{2}\theta}\{U\frac{\partial V}{\partial\lambda}+V~cos~\theta\frac{\partial V}{\partial\theta}+sin~\theta(U^{2}+V^{2})\}$: Horizontal advection and curvature terms affecting meridional wind.
* $\dot{\eta}\frac{\partial V}{\partial\eta}$: Vertical advection of meridional wind.
* $fU$: Coriolis force term affecting meridional wind.
* $\frac{cos~\theta}{a}\{\frac{\partial\phi}{\partial\theta}+R_{dry}T_{v}\frac{\partial}{\partial\theta}(ln~p)\}$: Pressure gradient force in the meridional direction.
* $P_V$: Source/sink term due to parameterized physical processes.
* $K_V$: Term representing horizontal diffusion.

---

### Virtual Temperature Definition

**Virtual temperature ($T_v$)** is an adjusted temperature used in atmospheric equations to account for the effect of water vapor on the density of air. Moist air is less dense than dry air at the same temperature and pressure, and using virtual temperature allows for a simpler application of the ideal gas law to moist air.

$$
T_{v}=T[1+\{(R_{vap}/R_{dry})-1\}q]
$$

* $T_v$: Virtual temperature.
* $T$: Actual temperature.
* $R_{vap}$: Gas constant for water vapor.
* $R_{dry}$: Gas constant for dry air.
* $q$: Specific humidity (mass of water vapor per unit mass of air).

---

### Thermodynamic Equation

This equation describes how **temperature ($T$)** changes in the atmosphere due to processes like heat advection (transport), vertical motion (adiabatic heating/cooling), and heating/cooling from various physical processes such as radiation and latent heat release.

$$
\frac{\partial T}{\partial t}+\frac{1}{a~cos^{2}\theta}\{U\frac{\partial T}{\partial\lambda}+V~cos~\theta\frac{\partial T}{\partial\theta}\}+\dot{\eta}\frac{\partial T}{\partial\eta}-\frac{\kappa T_{v}\omega}{(1+(\delta-1)q)p}=P_{T}+K_{T}
$$

* $\frac{\partial T}{\partial t}$: Local rate of change of temperature.
* $\frac{1}{a~cos^{2}\theta}\{U\frac{\partial T}{\partial\lambda}+V~cos~\theta\frac{\partial T}{\partial\theta}\}$: Horizontal advection of temperature.
* $\dot{\eta}\frac{\partial T}{\partial\eta}$: Vertical advection of temperature.
* $\frac{\kappa T_{v}\omega}{(1+(\delta-1)q)p}$: Term related to adiabatic heating/cooling due to vertical motion (where $\kappa$ and $\delta$ are constants, and $\omega$ is pressure-coordinate vertical velocity).
* $P_T$: Source/sink term due to parameterized physical processes (e.g., radiation, latent heat release).
* $K_T$: Term representing horizontal diffusion.

---

### Moisture Equation

This equation governs the change in **specific humidity ($q$)** (the amount of water vapor in the air) over time, influenced by its transport through horizontal and vertical motion, as well as various moisture-related physical processes.

$$
\frac{\partial q}{\partial t}+\frac{1}{a~cos^{2}\theta}\{U\frac{\partial q}{\partial\lambda}+V~cos~\theta\frac{\partial q}{\partial\theta}\}+\dot{\eta}\frac{\partial q}{\partial\eta}=P_{q}+K_{q}
$$

* $\frac{\partial q}{\partial t}$: Local rate of change of specific humidity.
* $\frac{1}{a~cos^{2}\theta}\{U\frac{\partial q}{\partial\lambda}+V~cos~\theta\frac{\partial q}{\partial\theta}\}$: Horizontal advection of specific humidity.
* $\dot{\eta}\frac{\partial q}{\partial\eta}$: Vertical advection of specific humidity.
* $P_q$: Source/sink term due to parameterized physical processes (e.g., condensation, evaporation).
* $K_q$: Term representing horizontal diffusion.

---

### Continuity Equation

This equation expresses the fundamental principle of **mass conservation** in the atmosphere within the chosen vertical coordinate system ($\eta$). It states that the local change in mass within a volume is balanced by the net flow of mass into or out of that volume.

$$
\frac{\partial}{\partial t}(\frac{\partial p}{\partial\eta})+\nabla\cdot(v_{H}\frac{\partial p}{\partial\eta})+\frac{\partial}{\partial\eta}(\dot{\eta}\frac{\partial p}{\partial\eta})=0
$$

* $\frac{\partial}{\partial t}(\frac{\partial p}{\partial\eta})$: Local rate of change of pressure in $\eta$-coordinates.
* $\nabla\cdot(v_{H}\frac{\partial p}{\partial\eta})$: Horizontal divergence of mass flux, where $\nabla$ is the horizontal gradient operator and $v_H=(u,v)$ is the horizontal wind vector.
* $\frac{\partial}{\partial\eta}(\dot{\eta}\frac{\partial p}{\partial\eta})$: Vertical divergence of mass flux.

---

### Hydrostatic Equation

This equation describes the **hydrostatic balance**, a fundamental equilibrium in the atmosphere where the upward pressure gradient force is balanced by the downward force of gravity. It defines how geopotential ($\phi$) changes with altitude.

$$
\frac{\partial\phi}{\partial\eta}=-\frac{R_{dry}T_{v}}{p}\frac{\partial p}{\partial\eta}
$$

* $\frac{\partial\phi}{\partial\eta}$: Vertical gradient of geopotential.
* $R_{dry}$: Gas constant for dry air.
* $T_v$: Virtual temperature.
* $p$: Pressure.
* $\frac{\partial p}{\partial\eta}$: Vertical gradient of pressure in $\eta$-coordinates.

---

### Pressure-Coordinate Vertical Velocity ($\omega$)

This equation defines the **vertical velocity in pressure coordinates ($\omega$)**, which is a common way to express vertical motion in atmospheric dynamics, representing the change in pressure over time experienced by a parcel of air.

$$
\omega=-\int_{0}^{\eta}\nabla\cdot(v_{H}\frac{\partial p}{\partial\eta})d\eta+v_{H}\cdot\nabla p
$$

* $\omega$: Pressure-coordinate vertical velocity.
* $-\int_{0}^{\eta}\nabla\cdot(v_{H}\frac{\partial p}{\partial\eta})d\eta$: Integrated horizontal divergence of mass flux from the top of the atmosphere to a given $\eta$-level.
* $v_{H}\cdot\nabla p$: Horizontal advection of pressure.

---

### Rate of Change of Surface Pressure ($p_s$)

This equation relates the change in **surface pressure ($p_s$)** over time to the integrated effect of horizontal mass divergence throughout the entire atmospheric column.

$$
\frac{\partial p_{s}}{\partial t}=-\int_{0}^{1}\nabla\cdot(v_{H}\frac{\partial p}{\partial\eta})d\eta
$$

* $\frac{\partial p_{s}}{\partial t}$: Local rate of change of surface pressure.
* $-\int_{0}^{1}\nabla\cdot(v_{H}\frac{\partial p}{\partial\eta})d\eta$: Vertically integrated horizontal divergence of mass flux across the entire atmospheric column (from top $\eta=0$ to bottom $\eta=1$).

---

### Vertical Velocity in $\eta$-Coordinates ($\dot{\eta}$)

This equation provides an expression for the **vertical velocity within the $\eta$-coordinate system ($\dot{\eta}$) multiplied by the pressure gradient in $\eta$-coordinates**. It is derived from the continuity equation and is fundamental for calculating vertical motion in this specific coordinate system.

$$
\dot{\eta}\frac{\partial p}{\partial\eta}=-\frac{\partial p}{\partial t}-\int_{0}^{\eta}\nabla\cdot(v_{R}\frac{\partial p}{\partial\eta})d\eta
$$

* $\dot{\eta}\frac{\partial p}{\partial\eta}$: Term representing vertical velocity in $\eta$-coordinates multiplied by the vertical pressure gradient.
* $-\frac{\partial p}{\partial t}$: Local rate of change of pressure.
* $-\int_{0}^{\eta}\nabla\cdot(v_{R}\frac{\partial p}{\partial\eta})d\eta$: Integrated horizontal divergence of mass flux from the top of the atmosphere to a given $\eta$-level (note: $v_R$ should likely be $v_H$ for consistency with other equations for horizontal velocity).

---

### Logarithm of Surface Pressure Change

This is an alternative, often more numerically stable, formulation for the rate of change of the natural logarithm of **surface pressure ($ln~p_s$)** with respect to time. It is derived directly from the surface pressure tendency equation.

$$
\frac{\partial}{\partial t}(ln~p_{s})=-\frac{1}{p_{s}}\int_{0}^{1}\nabla\cdot(v_{H}\frac{\partial p}{\partial\eta})d\eta
$$

* $\frac{\partial}{\partial t}(ln~p_{s})$: Local rate of change of the natural logarithm of surface pressure.
* This equation is equivalent to Equation 2.8, scaled by $1/p_s$.


---

## 2. Detection & Attribution

**Detection** in climate change refers to identifying whether observed climate shifts fall outside the bounds of natural variability. By precisely defining this natural range, researchers can determine if an event stems from natural or external (e.g., anthropogenic) forcing. However, accurately quantifying natural variability remains challenging due to insufficient observational data. **Attribution**, on the other hand, aims to ascertain the specific causes of a detected change. In this context, anthropogenic and natural forcings act as signals, while natural variabilities are considered noise, framing the climate detection problem as a signal-to-noise challenge.

Optimal Fingerprinting
Optimal fingerprinting is a widely employed statistical method for comparing observed climate patterns with those simulated by climate models. A "fingerprint" represents the characteristic spatial and temporal pattern induced by a specific external forcing. This method compares observed patterns ($Y$) with model-derived fingerprint patterns ($X$) using total least-squares regression.

If the confidence interval of the regression coefficient $\beta$ is greater than zero, it indicates a statistically significant correlation between observations and the model's fingerprint, signifying successful detection. Furthermore, if the confidence interval of $\beta$ includes 1, it implies that the magnitude of the observed change is consistent with the model's simulated response to that forcing. Therefore, determining the appropriate $\beta$ value is central to this analytical process.

For a detailed examination of how models' fingerprints, influenced by greenhouse gases, aerosols, and stratospheric ozone depletion, are compared, one can refer to Santer et al. (1996).

{% raw %}
<img src="/images/Climate_Change_Detection_Attribution/Santer_1996.png" alt="Figure 2">
{% endraw %}

**Figure 2:** Modelled and moserved zonal-mean annually averaged changes in the thermal structure of the atmosphere.(j) observed change.(Adapted from Santer et al.1996)


## Overview
Previously, climate change research mainly focused on atmospheric and ocean warming. Now, studies increasingly attribute shifts in atmospheric circulation, precipitation, regional water cycles, sea ice, and extreme events to human influence. Event attribution (Stott et al. 2004; Herring et al. 2014; Min et al. 2015) actively assesses if human activities increase the probability of specific extreme events.

However, detection and attribution research faces significant limitations. For instance, understanding East Asian precipitation changes requires better comprehension of monsoon variability, currently insufficient in models and physical understanding. Climate models' limited ability to simulate regional-scale variability means natural variability still heavily influences results, hindering human influence identification. Thus, continuous climate model improvements, sophisticated analysis methods, and international collaboration are crucial for effective climate change mitigation and adaptation.


---

## References
- 민승기, 기후변화 탐지와 원인규명, 기후과학자가 쓴 기후역학 교과서, 309-331.
- IPCC, 2013: Climate Change 2013: The Physical Science Basis. Contribution of Working Group I to the Fifth Assessment Report of the Intergovernmental Panel on Climate Change, Stocker, T. F. et al. Eds., Cambridge University Press, Cambridge, United Kingdom and New York, NY, USA
- Santer, B., Taylor, K., Wigley, T. et al. A search for human influences on the thermal structure of the atmosphere. Nature 382, 39–46 (1996).
- ECMWF, IFS DOCUMENTATION - Cy43rl Operational implementation 22 Nov 2016, PART III: DYNAMICS AND NUMERICAL PROCEDURES
