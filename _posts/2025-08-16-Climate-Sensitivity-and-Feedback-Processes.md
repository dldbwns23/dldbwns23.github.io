---
layout: post
title: "Climate Sensitivity and Feedback Processes"
date: 2025-08-16 14:00:00 +0900
tags: [Climate Dynamics]
---

Climate change is generally associated with changes in the average temperature of a certain region's surface or atmosphere. Changes in humidity, precipitation, and ecology are also included due to their relationship with temperature. Global climate models produce greenhouse gas emission scenarios, enabling researchers to predict climate change with these useful tools. Improving global climate models leads to more precise predictions; however, this does not mean that a more complex model is necessarily better. The interaction between the atmosphere, ocean, land surface, ecology, and cryosphere drives future climate, which means the **feedback processes** among them have become a crucial concept.

Future climate prediction is a problem that requires decades of long-term integration of differential equations. Under the governing equations of the atmosphere, climate reacts sensitively to external forcings such as greenhouse gases and solar cycles, and internal forcings such as volcanoes, geological activity, and periodic El Niño. **Climate sensitivity** refers to the degree to which climate reacts to these forcings. Climate sensitivity is controlled by feedback processes.

---

## 1. Climate Sensitivity and Feedback Concept

{% raw %}
<img src="/images/Climate_Sensitivity_Feedback_Processes/Lindzen_2001.png" alt="Figure 1.1 Schematic illustrating operation of feedbacks">
{% endraw %}
**Figure 1.1:** Schematic illustrating the operation of feedback and no-feedback cases. (Adapted from Lindzen et al., 2001)

Figure 1.1 presents a conceptual model of feedback and no-feedback cases.

In (a) No Feedback Case, $\Delta T_{0} = G_{0}\Delta Q$, while in (b) Feedback Case, $\Delta T = G_{0}(\Delta Q + F\Delta T)$.

Therefore, $\Delta T_{0} = \Delta T(1-G_{0}F)$. **F is called the feedback parameter**, with unit $W m^{-2}K^{-1}$, which signifies the degree of feedback using energy imbalance at the Top of Atmosphere (TOA) under a change in unit surface temperature. For simplicity, the **feedback factor ($f$)** is defined as $G_{0}F$.

We can derive $\Delta T = G_{0}\Delta Q / (1-G_{0}F) = \Delta T_{0}/(1-f)$. If $f$ is between 0 and 1, it indicates **positive feedback**, and if $f$ is less than 0, it indicates **negative feedback**. It is physically undefined when $f$ is greater than or equal to 1, as $\Delta T$ becomes negative.

Temperature, water vapor, clouds, and albedo are the main contributors to feedback. They are nonlinearly connected. For example, if cloud feedback is negative, energy emitted by water vapor is exposed to the TOA, reinforcing water vapor feedback. Therefore, it is inappropriate to separate the effect of each component. However, it is known that the entire feedback factor $f$ can be represented by the addition of individual $f$ values for temperature, water vapor, clouds, and albedo.

Figure 1.2 illustrates the relationship between climate sensitivity and the feedback factor. The feedback factor on the x-axis assumes that feedback factors from prior research follow a Gaussian distribution. It can be easily shown that climate models with a feedback factor near 0.65 exhibit a significant difference in climate sensitivity.

{% raw %}
<img src="/images/Climate_Sensitivity_Feedback_Processes/Roe_and_Baker_2007.png" alt="Figure 1.2 Demonstration of the relationships linking climate sensitivity to feedback factor">
{% endraw %}
**Figure 1.2:** Demonstration of the relationships linking climate sensitivity to the feedback factor. (Adapted from Roe and Baker, 2007)

## 2. How To Measure Climate Sensitivity and Feedback

Climate sensitivity can be utilized as a factor for measuring impact of certain meteorological-physical processes on climate change and as an index for climate models. Also, sensitivity numerically refers to how much an objective observation of climate changes as one of multiple independent variables changes. For instance, let's consider surface temperature $T_{s}$. Let $S_{0}$ be solar constant. If $S_{0}$ changes, so does $T_{s}$. Since $T_{s}$ is a function of other variables such as $y_{i}$, we can get the following:

$$\frac{dT}{dS_{0}} = \frac{\partial T_{s}}{\partial S_{0}}+ \sum_{j=1}^{N} \frac{\partial T_{s}}{\partial y_{i}}\frac{dy_{i}}{dS_{0}}$$

By defining forcing $dQ$ ($Wm^{-2}$) and the ratio $\frac{dT_{s}}{dQ} \equiv \lambda_{R}$, we can check how average surface temperature changes are related to the degree of climate change.

The radiative balance at the Top of the Atmosphere ($R_{TOA}$) is given by:
$$R_{TOA} = \frac{S_{0}}{4}(1-\alpha_{p}) - F^{\uparrow}(\infty)=0$$
Here $\alpha_{p}$ is the planet's albedo. If a certain forcing $dQ$ is given, the climate will transit to a new average surface temperature equilibrium.

The change in $R_{TOA}$ with respect to the forcing and temperature change can be expressed as:
$$\frac{dR_{TOA}}{dQ} = \frac{\partial R_{TOA}}{\partial Q} + \frac {\partial R_{TOA}}{\partial T_{s}}\frac{dT_{s}}{dQ} = 0$$
$\partial R_{TOA} / \partial T_{s}$ represents the energy imbalance at TOA relying on surface temperature, which is equivalent to the feedback parameter $F$. Also, $\partial R_{TOA} / \partial Q=1$ since $R_{TOA}$ is always same as the forcing itself. Therefore, the above can be simplified as $1+F\cdot\lambda_{R}=0$.
Thus, the climate sensitivity $\lambda_R$ can be derived as:
$$\frac{dT_{s}}{dQ} = -(\frac{\partial R_{TOA}}{\partial T_{s}})^{-1} = (\frac{S_{0}}{4}\frac{\partial \alpha_{p}}{\partial T_{s}}+\frac{\partial F^{\uparrow}(\infty)}{\partial T_{s}})^{-1} = \lambda_{R}$$
In a simple model, climate sensitivity is primarily determined by two basic feedback processes: the derivative of outgoing Earth radiation flux with respect to surface temperature, and the derivative of surface albedo with respect to surface temperature.

{% raw %}
<img src="/images/Climate_Sensitivity_Feedback_Processes/Choi_2011.png" alt="Figure 1.3 Schematic relation between sea surface temperature change and outgoing total radiative flux change at the top of the atmosphere. The slope represents the sign and magnitude of total climate feedback. ZFB stands for zero-feedback system. Negative slope corresponds to positive climate feedback of current climate models. The slope over that of ZFB corresponds to negative climate feedback.">
{% endraw %}
**Figure 1.3:** Schematic relation between sea surface temperature change and outgoing total radiative flux change at the top of the atmosphere. The slope represents the sign and magnitude of total climate feedback. ZFB stands for zero-feedback system. A negative slope corresponds to positive climate feedback in current climate models. The slope over that of ZFB corresponds to negative climate feedback. (Adapted from Choi, 2011)

## 3. Basic Feedback Processes

The following are basic feedback processes in climate:

- Stefan-Boltzmann Feedback
- Temperature Lapse Rate Feedback
- Water Vapor Feedback
- Ice-Albedo Feedback
- Dynamic Feedback and North-South Energy Transfer
- Cloud Feedback
- Biogeochemistry Feedback

Details of these feedback processes can be found in many references.

### Stefan-Boltzmann Feedback

>This is a negative feedback. As the Earth's surface and atmosphere warm, they radiate more longwave (infrared) energy back to space. This increased outgoing radiation acts to cool the planet, partially offsetting the initial warming. It's an inherent property of any body that emits thermal radiation.

### Temperature Lapse Rate Feedback

>The lapse rate refers to the rate at which temperature decreases with altitude in the atmosphere. This feedback can be positive or negative. If the upper atmosphere warms more than the lower atmosphere, the lapse rate decreases (becomes more stable), which can lead to more outgoing longwave radiation and thus a negative feedback. Conversely, if the lower atmosphere warms more, it can lead to a positive feedback. The exact nature depends on how warming is distributed vertically in the atmosphere.

### Water Vapor Feedback

>This is a strong positive feedback. As the atmosphere warms, its capacity to hold water vapor increases. Water vapor is a potent greenhouse gas, so more water vapor in the atmosphere traps more heat, leading to further warming. This additional warming, in turn, allows the atmosphere to hold even more water vapor, creating a reinforcing cycle.

### Ice-Albedo Feedback

>This is a powerful positive feedback. As global temperatures rise, ice and snow cover melt. Ice and snow have a high albedo, meaning they reflect a large portion of incoming solar radiation back to space. When they melt, they expose darker land or ocean surfaces, which absorb more solar radiation. This increased absorption leads to further warming, which causes more ice to melt, amplifying the initial warming.

### Dynamic Feedback and North-South Energy Transfer

>This category encompasses feedbacks related to changes in atmospheric and oceanic circulation patterns. These dynamics can influence how heat is distributed across the planet, particularly between the tropics and the poles. Changes in ocean currents, for instance, can alter the rate at which heat is transported from warmer to colder regions, potentially leading to regional warming or cooling that then feeds back into the global climate system. The complexity of these interactions means they can contribute to both positive and negative feedbacks.

### Cloud Feedback

>This is one of the most uncertain and complex feedbacks. Clouds can exert both cooling and warming effects.
> - Cooling effect: High albedo clouds reflect incoming solar radiation back to space (negative feedback).
> - Warming effect: Clouds can also trap outgoing longwave radiation, acting like a blanket (positive feedback).
>
>The net effect of cloud feedback depends on changes in cloud type, altitude, and coverage, making it a major source of uncertainty in climate models.

### Biogeochemistry Feedback

>This refers to feedbacks involving interactions between the climate system and biological and geological processes, particularly those affecting the cycles of greenhouse gases like carbon dioxide and methane.
> - Carbon Cycle Feedback: As the Earth warms, natural carbon sinks (like oceans and forests) may become less efficient at absorbing CO2, or even release stored carbon, leading to higher atmospheric CO2 levels and further warming (positive feedback). For example, thawing permafrost can release large amounts of CH4.
>
> - Vegetation Changes: Shifts in vegetation patterns due to climate change can alter surface albedo and evapotranspiration, influencing regional and global temperatures.

## Overview
This discussion has provided a foundational understanding of climate sensitivity and feedback processes, two cornerstone concepts in climate dynamics. We established how climate change, driven by various forcings, is critically influenced by complex interactions among Earth's systems. Note that feedback processes play a defining role in amplifying or dampening initial changes. The mathematical flow illustrated the direct linkage between climate sensitivity and the feedback factor. While these fundamental feedback mechanisms have been outlined, their quantification, particularly their non-linear interactions and spatiotemporal variability, remains vague. Future efforts must continue to refine climate models to better constrain the magnitude and sign of these feedbacks.

## References
- 최용상, 기후민감도와 되먹임, 기후과학자가 쓴 기후역학 교과서, 333-372
- Lindzen, R. S., M.-D. Chou, and A. Y. Hou, 2001: Does the Earth Have an Adaptive Infrared Iris? Bull.amer.Meteorol. Soc., 82(3), 417-432
- Roe, G.H. and Baker, M.B., 2007: Why is climate sensitivity so unpredictable? Science, 318, 629-632
- Choi, Y.S., 2011: How Sensitive is the Earth Climate to a Runaway Carbon Dioxide? J.Korean Earth Science Soc., 32, 239-247
