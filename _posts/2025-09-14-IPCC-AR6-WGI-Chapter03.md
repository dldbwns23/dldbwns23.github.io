---
layout: post
title: "IPCC AR6 WGI Chapter03"
date: 2025-09-14 9:00:00 +0900
tags: [Article, IPCC]
---

## 0. Executive Summary

This chapter provides an updated assessment of human influence on the climate system, focusing on large-scale indicators of climate change. It synthesizes information from paleoclimate records, modern observations, and climate models.

### Synthesis Across the Climate System

It is **unequivocal** that human influence has warmed the atmosphere, ocean, and land since pre-industrial times. Combining evidence from across the climate system increases confidence in the attribution of observed changes to human activities. The latest generation of climate models from the Coupled Model Intercomparison Project Phase 6 (CMIP6) shows improved simulation of the recent mean climate for most large-scale indicators compared to the previous generation (CMIP5) (*high confidence*). Overall, the multi-model mean captures most aspects of observed climate change well (*high confidence*).

### Human Influence on the Atmosphere and Surface

The likely range of human-induced warming in global-mean surface air temperature (GSAT) in 2010–2019 relative to 1850–1900 is $0.8^{\circ}C$–$1.3^{\circ}C$, which encompasses the observed warming of $0.9^{\circ}C$–$1.2^{\circ}C$. The change attributable to natural forcings is only $-0.1^{\circ}C$ to $+0.1^{\circ}C$. It is *very likely* that human-induced greenhouse gas increases were the main driver of tropospheric warming since satellite observations began in 1979, and *extremely likely* that human-induced stratospheric ozone depletion was the main driver of cooling in the lower stratosphere between 1979 and the mid-1990s.

The CMIP6 model ensemble reproduces the observed historical global surface temperature trend and variability with biases small enough to support detection and attribution of human-induced warming (*very high confidence*). For upper-air temperature, there is *medium confidence* that most CMIP5 and CMIP6 models overestimate observed warming in the upper tropical troposphere by at least $0.1^{\circ}C$ per decade over the period 1979 to 2014. The slower rate of global mean surface temperature (GMST) increase observed over 1998–2012 was a temporary event, followed by a strong increase (*very high confidence*).

It is *likely* that human influence has contributed to observed large-scale precipitation changes since the mid-20th century. This includes strengthening the zonal mean precipitation contrast between the wet tropics and dry subtropics. Human-induced greenhouse gas forcing is the **main driver** of observed changes in hot and cold extremes on a global scale (*virtually certain*) and on most continents (*very likely*).

It is *likely* that human influence has contributed to the poleward expansion of the Hadley cell in the Southern Hemisphere since the 1980s. While CMIP6 models capture the general characteristics of the large-scale tropospheric circulation (*high confidence*), systematic biases persist in the mean frequency of atmospheric blocking events, especially in the Euro-Atlantic sector.

### Human Influence on the Cryosphere

It is *very likely* that anthropogenic forcing, primarily from greenhouse gas increases, was the **main driver of Arctic sea ice loss** since the late 1970s. It is also *very likely* that human influence contributed to the observed reductions in Northern Hemisphere spring snow cover since 1950.

### Human Influence on the Ocean

It is *extremely likely* that human influence was the **main driver of the increase in ocean heat content** observed since the 1970s, a warming that extends into the deeper ocean (*very high confidence*). Updated observations and model simulations confirm that this warming extends throughout the entire water column (*high confidence*).

It is *extremely likely* that human influence has contributed to observed near-surface and subsurface ocean salinity changes since the mid-20th century. The associated pattern shows fresh regions becoming fresher and salty regions becoming saltier (*high confidence*), driven by changes in the atmospheric water cycle (evaporation and precipitation) (*high confidence*).

Combining contributions from glacier melt, ice-sheet surface mass balance, and thermal expansion, it is *very likely* that human influence was the **main driver of the observed global mean sea level rise** since at least 1971.

### Human Influence on the Biosphere

The observed increase in the amplitude of the seasonal cycle of atmospheric $CO_2$ is primarily driven by **enhanced fertilization of plant growth** from the increasing concentration of atmospheric $CO_2$ (*medium confidence*).

### Human Influence on Modes of Climate Variability

It is *very likely* that human influence has contributed to the observed trend toward the **positive phase of the Southern Annular Mode (SAM)** since the 1970s and the associated strengthening and southward shift of the Southern Hemispheric extratropical jet in austral summer.

Human influence has **not affected the principal tropical modes** of interannual climate variability (like ENSO) or their regional teleconnections beyond the range of internal variability (*high confidence*). There is *medium confidence* that anthropogenic and volcanic aerosols contributed to observed changes in the Atlantic Multi-decadal Variability (AMV) index since the 1960s, but there is *low confidence* in the magnitude of this influence.

---
## 1. Scope

This chapter assesses the degree to which human influence has affected the climate system and the ability of climate models to simulate observed mean climate, its changes, and its variability. This assessment provides a foundation for understanding the current impacts of anthropogenic climate change and informs confidence in future climate projections. To attribute observed climate change, an estimate of the expected response to human influence is required, along with estimates of the climate's evolution due to natural forcings and internal variability.

Unlike previous IPCC reports, which separated model evaluation and attribution into different chapters, AR6 integrates these comparisons. This approach allows for a more holistic discussion of inconsistencies between simulations and observations, considering factors like missing forcings, errors in the simulated response, and observational errors. However, if a model's simulation of historical climate has been tuned to match observations, or if models in an attribution study were selected based on the realism of their climate response, these factors must be considered, and attribution results tempered accordingly. This chapter assesses the detection of anthropogenic influence on large spatial and long temporal scales, which is distinct from the emergence of climate change signals from internal variability on local scales and shorter timescales.

---
## 2. Methods

### 2.1 Methods Based on Regression

Regression-based techniques, also known as **fingerprinting methods**, are widely used to detect climate change and attribute it to different external drivers. In this approach, an observed change is assumed to be a linear combination of externally forced signals (the "fingerprints") and internal variability. A regression coefficient that is significantly greater than zero implies a detectable change has been identified in observations. When the confidence interval for this coefficient includes one and is inconsistent with zero, the magnitude of the model-simulated response is considered consistent with observations, implying the change can be attributed in part to that forcing. To ensure robustness, the model-simulated internal variability is typically checked against the observed residual variance using a residual consistency test.

Recent methodological advances include:
* **Integrated Approaches:** Procedures that treat all sources of uncertainty (observational error, model error, internal variability) within a single statistical model, avoiding preliminary dimension-reduction steps. Similar integrated approaches have been developed using Bayesian model averaging.
* **Improved Confidence Intervals:** Bootstrap methods have been suggested to better estimate the confidence intervals of scaling factors, particularly in weak-signal regimes.
* **Pattern Similarity:** An alternative fingerprinting approach uses pattern similarity, where observations are projected onto a time-evolving fingerprint (often the leading empirical orthogonal function from a multi-model forced simulation) to measure the degree of similarity with the expected physical response. To handle high-dimensional daily data, statistical learning techniques like regularized linear regression have been proposed to optimize the signal by down-weighting regions with large internal variability.

### 2.2 Other Probabilistic Approaches

Other probabilistic methods have been introduced to better account for climate modeling uncertainties. These include statistical frameworks based on likelihood maximization, which test whether observations are consistent with the expected response from models, and the application of causal theory to calculate the probability of causation for a given forcing.

---
## 3. Human Influence on the Atmosphere and Surface

### 3.1 Temperature

#### 3.1.1 Surface Temperature

##### 3.1.1.1 Model Evaluation

To be fit for attribution, climate models must realistically represent both the response of surface temperature to external forcings and its internal variability across various timescales, based on physical principles.

* **Paleoclimate Performance:** CMIP6 models generally simulate regional temperature responses during the Last Interglacial (LIG) that are within the uncertainty ranges of reconstructions, though mismatches can occur where processes like meltwater or vegetation changes are unresolved. Biases in mid-Holocene simulations appear to have common causes with those in pre-industrial simulations. Over the last millennium, model agreement with reconstructions is good for Northern Hemisphere continents but poor for the Southern Hemisphere, partly due to lower-quality reconstructions there.
* **Historical Period Performance:** Temperature biases in CMIP6 models are similar to those in CMIP5, with persistent errors in regions like the North Atlantic and in areas of upwelling. The fundamental causes of these biases are still elusive but are linked to errors in clouds, ocean circulation, and surface energy budgets. Models are evaluated using **temperature anomaly time series**, which focus on the change over time rather than absolute temperature, as this is more relevant for climate change evaluation and helps isolate long-term trends.
* **Trends and Biases:** While most CMIP6 modeling groups report improvements in simulating the present-day climate, this does not always translate to improved simulation of temperature trends. The CMIP6 multi-model mean is notably cooler than both observations and CMIP5 during the 1980–2000 period. Temperature biases are driven by a combination of model physics and the prescribed forcings, making them a complex challenge to resolve.
* **Internal Variability:** The way models represent natural internal variability is crucial for attribution. The influence of phenomena like the El Niño-Southern Oscillation (ENSO) and the Atlantic Multi-decadal Oscillation (AMO) on global temperature is significant, especially on timescales shorter than 30–40 years. Some models, particularly in CMIP6, exhibit much higher multi-decadal variability than was seen in CMIP5 or in proxy reconstructions. For example, the high variability in the CNRM models is attributed to a strong simulated AMV, which is difficult to rule out due to the short observational record. Importantly, however, the spatial patterns of this simulated internal variability differ from the pattern of forced temperature change.

Despite model imperfections, there is *very high confidence* that the biases in temperature trends and variability simulated by the CMIP5 and CMIP6 ensembles are small enough to support the detection and attribution of human-induced warming.


{% raw %}
<img src="/images/IPCC_AR6_Ch3/Ch3_1.png" alt="Figure 3.2">
{% endraw %}


{% raw %}
<img src="/images/IPCC_AR6_Ch3/Ch3_2.png" alt="Figure 3.3">
{% endraw %}

{% raw %}
<img src="/images/IPCC_AR6_Ch3/Ch3_3.png" alt="Figure 3.4">
{% endraw %}

{% raw %}
<img src="/images/IPCC_AR6_Ch3/Ch3_4.png" alt="Figure 3.5">
{% endraw %}

{% raw %}
<img src="/images/IPCC_AR6_Ch3/Ch3_5.png" alt="Figure 3.6">
{% endraw %}

##### 3.1.1.2 Detection and Attribution

* **Long-Term Context:** The warming of the 20th century reversed a 5000-year cooling trend in the Northern Hemisphere mid- to high latitudes that was caused by orbital forcing; this reversal is attributed to anthropogenic forcing with *high confidence*. Over the last millennium, observed temperature changes were mostly driven by volcanic forcing, with smaller roles for greenhouse gases and solar forcing.
* **Methodological Advances:** Since AR5, attribution studies have improved by better accounting for observational and model uncertainties, using counterfactual frameworks, and reducing dependence on uncertain climate sensitivity estimates. Alternative techniques, such as statistical and econometric approaches that do not rely on climate models, have also been applied and yield results that match physically-based methods.
* **Disentangling Forcings:** While the net human influence is clear, separating the contributions of greenhouse gases and aerosols remains challenging. Model responses to individual forcings can be inconsistent, and detection of the aerosol signal often fails in single-model analyses. This highlights the need to use multi-model means to better constrain the attributable warming from different anthropogenic sources.
* **Key Attribution Statement:** Using multiple lines of evidence, the assessed human contribution to global warming in 2010–2019 relative to 1850–1900 is **$1.07^{\circ}C$**, with a likely range of **$0.8^{\circ}C$ to $1.3^{\circ}C$**. Progress in attribution techniques allows this statement to be made for warming since 1850–1900, an important advance from AR5, which attributed warming since 1951. The contribution of internal variability to warming over this period is assessed as likely being between **$-0.2^{\circ}C$ and $+0.2^{\circ}C$**.


{% raw %}
<img src="/images/IPCC_AR6_Ch3/Ch3_6.png" alt="Figure 3.7">
{% endraw %}

{% raw %}
<img src="/images/IPCC_AR6_Ch3/Ch3_7.png" alt="Figure 3.8">
{% endraw %}

{% raw %}
<img src="/images/IPCC_AR6_Ch3/Ch3_8.png" alt="Figure 3.9">
{% endraw %}

{% raw %}
<img src="/images/IPCC_AR6_Ch3/Ch3_table_1.png" alt="Table 3.1">
{% endraw %}


#### 3.1.2 Upper-air Temperature

##### 3.1.2.1 Tropospheric temperature

Studies continue to show an inconsistency between simulated and observed temperature trends in the tropical troposphere, with models generally simulating more warming than has been observed. This warming bias persists in CMIP6 models despite the use of updated forcing estimates. The discrepancy is linked to several factors:
* **Climate Sensitivity and Feedbacks:** Differences in climate sensitivity among models may be an important factor. One study suggested that the absence of a hypothesized negative tropical cloud feedback could explain half of the warming bias in one model.
* **Sea Surface Temperature (SST) Biases:** Much of the difference in tropospheric temperature trends is due to corresponding differences in SST warming trends in regions of deep convection. The bias in tropospheric warming is partly linked to biases in surface temperature warming.

Taking these factors into account, there is *medium confidence* that models overestimate observed warming in the upper tropical troposphere over the 1979–2014 period by at least $0.1^{\circ}C$ per decade.

{% raw %}
<img src="/images/IPCC_AR6_Ch3/Ch3_9.png" alt="Figure 3.10">
{% endraw %}

##### 3.1.2.2 Stratospheric temperature

Increased greenhouse gases cause warming near the surface but cooling in the stratosphere. It was assessed in AR5 that anthropogenic forcing, dominated by ozone-depleting substances (ODS), *very likely* contributed to the cooling of the lower stratosphere since 1979.

Since AR5, research has shown that in the middle and upper stratosphere, the cooling since 1979 is mainly due to changes in greenhouse gas concentrations. Volcanic eruptions and the solar cycle have short-term influences but do not affect long-term trends. Based on this stronger evidence, it is now assessed as *extremely likely* that stratospheric ozone depletion from ODS was the **main driver of the cooling of the lower stratosphere between 1979 and the mid-1990s**.

---
## Cross-Chapter Box 3.1 | Global Surface Warming Over the Early 21st Century

The AR5 found that the rate of GMST increase from 1998–2012 was slower than the rate from 1951–2012 and slower than projected by CMIP5 models. This apparent slowdown was assessed with *medium confidence* to be caused in roughly equal measure by a cooling contribution from internal variability and a reduced trend in external forcing (from solar and volcanic activity).

Since AR5, updated observational datasets consistently find **stronger positive trends for 1998–2012** than those available for AR5, mainly due to improved SST datasets and better coverage of the rapidly warming Arctic. The updated observed trends now lie within the range of simulated trends from both CMIP5 and CMIP6 ensembles (*high confidence*).

The slowdown was a **temporary event** (*very high confidence*), induced by a combination of internal variability (particularly a negative phase of Pacific Decadal Variability) and natural forcing (moderate volcanic eruptions and a prolonged solar minimum) that partly offset the ongoing anthropogenic warming trend (*high confidence*). Throughout this period, the planet's total heat content continued to rise, as evidenced by uninterrupted warming of the global ocean (*very high confidence*). A major El Niño event in 2014–2016 marked the end of the slower warming period, leading to three consecutive years of record annual GMST.

{% raw %}
<img src="/images/IPCC_AR6_Ch3/Ch3_CB_1.png" alt="CB 3.1 Figure 1">
{% endraw %}

---

### 3.2 Precipitation, Humidity and Streamflow

#### 3.2.1 Paleoclimate Context

Detection and attribution studies for precipitation and other hydrological variables are hindered by their large internal variability relative to the anthropogenic signal. This low signal-to-noise ratio makes it difficult for a clear human-caused signal to emerge from the background of natural fluctuations. Furthermore, the sign of any change depends heavily on the specific location and time of year. While hydrological proxy data represent regional conditions, they can reveal large-scale patterns when synthesized.

For example, tree-ring records show that recent prolonged dry spells in the Levant and Chile are unprecedented in the last millennium (*high confidence*). Similarly, the 2012–2014 drought in the southwestern United States was exceptionally severe and likely exacerbated by anthropogenic temperature rise (*medium confidence*). While this may be a signature of anthropogenic forcing, the signal cannot yet be definitively distinguished from natural variability. Indeed, paleoclimate records also indicate that megadroughts more severe than 20th and 21st-century events have occurred naturally in the past, driven by internal variability. This demonstrates that large-magnitude changes in the water cycle can occur irrespective of human influence.

Paleoclimate records also serve as a benchmark for climate model evaluation. One analysis concluded that CMIP5 models do not perform better than earlier model versions in simulating rainfall during the Last Glacial Maximum (LGM) and mid-Holocene, despite their higher resolution and complexity. However, some studies found that prescribing changes in vegetation and dust improves the match with paleoclimate records, suggesting that vegetation feedbacks in the models may be too weak (*low confidence*). In the tropics, the variable response across models is partly related to how they handle the exposure of continental shelves during glacial times, which can either intensify or weaken convection in the Walker circulation depending on the model's parameterization.


{% raw %}
<img src="/images/IPCC_AR6_Ch3/Ch3_10.png" alt="Figure 3.11">
{% endraw %}


{% raw %}
<img src="/images/IPCC_AR6_Ch3/Ch3_11.png" alt="Figure 3.12">
{% endraw %}

#### 3.2.2 Atmospheric Water Vapour

Changes in atmospheric water vapour are a critical component of the climate system, particularly in the upper troposphere, where water vapour strongly regulates the strength of the water-vapour feedback. However, model simulations of water vapour are not perfect; biases are dominated by errors in relative humidity throughout the troposphere, which are closely linked to errors in simulating large-scale circulation. At the same time, it must be noted that many well-established observational water vapour records are affected by inhomogeneity issues and should be used with caution.

Despite these challenges, a human fingerprint on atmospheric moisture is detectable. Early studies by Santer et al. (2007) provided evidence for human-induced moistening, and subsequent work with CMIP5 models confirmed that these detection and attribution conclusions were not sensitive to model quality. These robust results demonstrate that the human influence on atmospheric water vapour is governed by fundamental physical processes. Using CMIP5 simulations, Chung et al. (2014) showed that the observed moistening of the upper troposphere from 1979–2005 could not be explained by internal variability alone but is attributable to a combination of anthropogenic and natural forcings.

In summary, it is *likely* that human influence has contributed to moistening in the upper troposphere since 1979. Additionally, there is *medium confidence* that human influence has contributed to a global increase in annual surface specific humidity and to a decrease in surface relative humidity over mid-latitude Northern Hemisphere continents during summertime.

#### 3.2.3 Precipitation

While climate models capture the large-scale patterns of annual mean precipitation, the amplitude of simulated trends is often much smaller than observed. Models have struggled to reproduce the observed zonal mean trends in the magnitude of the seasonal cycle, although these differences might be explainable by internal variability. It is also important to note that observed trends in seasonality depend on the dataset used. One study found that observed changes in the *phase* of the annual cycle are consistent with forced changes in models, characterized by an earlier onset of the wet season in the extratropics, particularly in the Southern Hemisphere.

CMIP6 models show minor improvements over CMIP5, with better spatial correlations and day-to-day variability, but persistent biases like the double Inter-Tropical Convergence Zone (ITCZ) remain. Recent studies have shown that increasing atmospheric resolution can lead to a strong decrease in precipitation biases, particularly in the tropical Atlantic ITCZ. Nevertheless, despite some progress, CMIP6 models still have deficiencies in simulating precipitation patterns, especially over the tropical ocean (*high confidence*).

An intensification of the wet-dry zonal mean pattern—whereby wet tropical and mid-latitude bands get wetter while dry subtropical bands get drier—is an expected response to greenhouse gas and ozone changes. Detecting this pattern is complicated by model errors in the location of major rainfall features. To overcome this, researchers have used pattern-based fingerprinting that tracks the movement and amplification of these wet and dry zones, finding a detectable human signal in observations. While the observed strengthening of the wet-dry contrast is detectable, its magnitude is significantly larger in observations than in the multi-model mean.

Global land precipitation has *likely* increased since the mid-20th century (*medium confidence*), but there is *low confidence* in trends before 1950 and over the ocean due to disagreements between datasets. Consistent with models and our physical understanding of the Earth's energy budget, no significant trend has been found in global *mean* precipitation during the satellite era. This is because the expected increase from ocean warming is partly offset by the fast atmospheric adjustment to greenhouse gases and the cooling effect of anthropogenic aerosols. Over the ocean, observed salinity patterns provide powerful evidence of a changing water cycle. Coherent, large-scale patterns show that fresh ocean regions are becoming fresher and salty regions are becoming saltier, which is consistent with the "wet-gets-wetter, dry-gets-drier" paradigm driven by changes in precipitation minus evaporation.

New attribution studies have strengthened previous findings of a detectable increase in mid- to high-latitude land precipitation over the Northern Hemisphere (*high confidence*). There is *medium confidence* that human influence has contributed to the strengthening of the wet tropics-dry subtropics contrast. However, due to observational uncertainties and inconsistent results, there is *low confidence* in the attribution of changes in the seasonality of precipitation.

{% raw %}
<img src="/images/IPCC_AR6_Ch3/Ch3_12.png" alt="Figure 3.13">
{% endraw %}

{% raw %}
<img src="/images/IPCC_AR6_Ch3/Ch3_13.png" alt="Figure 3.14">
{% endraw %}

{% raw %}
<img src="/images/IPCC_AR6_Ch3/Ch3_14.png" alt="Figure 3.15">
{% endraw %}


#### 3.2.4 Streamflow

Streamflow is the only component of the terrestrial water cycle with sufficient in-situ observations for global-scale detection and attribution analysis. AR5 concluded with *medium confidence* that human influence had affected streamflow in some mid- and high-latitude regions.

More recent work has reinforced these findings. A pan-European assessment attributed the observed pattern of decreasing streamflow in southern Europe and increasing streamflow in northern Europe to anthropogenic climate change, while noting that the effects of human water withdrawals could not be excluded. A global study by Gudmundsson et al. (2021) found that the observed global pattern of streamflow trends—with some regions drying and others wetting—is consistent with simulations only when externally forced climate change is included.

In summary, there is *medium confidence* that anthropogenic climate change has altered local and regional streamflow in various parts of the world and that the associated global-scale pattern of trends is inconsistent with internal variability alone. Furthermore, while direct human interventions like water management and withdrawals affect streamflow, they cannot by themselves explain the observed spatio-temporal trends (*medium confidence*).

---

TBC


---


## Reference

Eyring, V., N.P. Gillett, K.M. Achuta Rao, R. Barimalala, M. Barreiro Parrillo, N. Bellouin, C. Cassou, P.J. Durack, Y. Kosaka, S. McGregor, S. Min, O. Morgenstern, and Y. Sun, 2021: Human Influence on the Climate System. In Climate Change 2021: The Physical Science Basis. Contribution of Working Group I to the Sixth Assessment Report of the Intergovernmental Panel on Climate Change [Masson-Delmotte, V., P. Zhai, A. Pirani, S.L. Connors, C. Péan, S. Berger, N. Caud, Y. Chen, L. Goldfarb, M.I. Gomis, M. Huang, K. Leitzell, E. Lonnoy, J.B.R. Matthews, T.K. Maycock, T. Waterfield, O. Yelekçi, R. Yu, and B. Zhou (eds.)]. Cambridge University Press, Cambridge, United Kingdom and New York, NY, USA, pp. 423–552, doi: 10.1017/9781009157896.005.
