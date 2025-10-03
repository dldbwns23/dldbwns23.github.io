---
layout: post
title: "IPCC AR6 WGI Chapter03"
date: 2025-09-14 9:00:00 +0900
tags: [Article, IPCC]
---

# 0. Executive Summary

This chapter provides an updated assessment of human influence on the climate system, focusing on large-scale indicators of climate change. It synthesizes information from paleoclimate records, modern observations, and climate models.

## Synthesis Across the Climate System

It is **unequivocal** that human influence has warmed the atmosphere, ocean, and land since pre-industrial times. Combining evidence from across the climate system increases confidence in the attribution of observed changes to human activities. The latest generation of climate models from the Coupled Model Intercomparison Project Phase 6 (CMIP6) shows improved simulation of the recent mean climate for most large-scale indicators compared to the previous generation (CMIP5) (*high confidence*). Overall, the multi-model mean captures most aspects of observed climate change well (*high confidence*).

## Human Influence on the Atmosphere and Surface

The likely range of human-induced warming in global-mean surface air temperature (GSAT) in 2010–2019 relative to 1850–1900 is $0.8^{\circ}C$–$1.3^{\circ}C$, which encompasses the observed warming of $0.9^{\circ}C$–$1.2^{\circ}C$. The change attributable to natural forcings is only $-0.1^{\circ}C$ to $+0.1^{\circ}C$. It is *very likely* that human-induced greenhouse gas increases were the main driver of tropospheric warming since satellite observations began in 1979, and *extremely likely* that human-induced stratospheric ozone depletion was the main driver of cooling in the lower stratosphere between 1979 and the mid-1990s.

The CMIP6 model ensemble reproduces the observed historical global surface temperature trend and variability with biases small enough to support detection and attribution of human-induced warming (*very high confidence*). For upper-air temperature, there is *medium confidence* that most CMIP5 and CMIP6 models overestimate observed warming in the upper tropical troposphere by at least $0.1^{\circ}C$ per decade over the period 1979 to 2014. The slower rate of global mean surface temperature (GMST) increase observed over 1998–2012 was a temporary event, followed by a strong increase (*very high confidence*).

It is *likely* that human influence has contributed to observed large-scale precipitation changes since the mid-20th century. This includes strengthening the zonal mean precipitation contrast between the wet tropics and dry subtropics. Human-induced greenhouse gas forcing is the **main driver** of observed changes in hot and cold extremes on a global scale (*virtually certain*) and on most continents (*very likely*).

It is *likely* that human influence has contributed to the poleward expansion of the Hadley cell in the Southern Hemisphere since the 1980s. While CMIP6 models capture the general characteristics of the large-scale tropospheric circulation (*high confidence*), systematic biases persist in the mean frequency of atmospheric blocking events, especially in the Euro-Atlantic sector.

## Human Influence on the Cryosphere

It is *very likely* that anthropogenic forcing, primarily from greenhouse gas increases, was the **main driver of Arctic sea ice loss** since the late 1970s. It is also *very likely* that human influence contributed to the observed reductions in Northern Hemisphere spring snow cover since 1950.

## Human Influence on the Ocean

It is *extremely likely* that human influence was the **main driver of the increase in ocean heat content** observed since the 1970s, a warming that extends into the deeper ocean (*very high confidence*). Updated observations and model simulations confirm that this warming extends throughout the entire water column (*high confidence*).

It is *extremely likely* that human influence has contributed to observed near-surface and subsurface ocean salinity changes since the mid-20th century. The associated pattern shows fresh regions becoming fresher and salty regions becoming saltier (*high confidence*), driven by changes in the atmospheric water cycle (evaporation and precipitation) (*high confidence*).

Combining contributions from glacier melt, ice-sheet surface mass balance, and thermal expansion, it is *very likely* that human influence was the **main driver of the observed global mean sea level rise** since at least 1971.

## Human Influence on the Biosphere

The observed increase in the amplitude of the seasonal cycle of atmospheric $CO_2$ is primarily driven by **enhanced fertilization of plant growth** from the increasing concentration of atmospheric $CO_2$ (*medium confidence*).

## Human Influence on Modes of Climate Variability

It is *very likely* that human influence has contributed to the observed trend toward the **positive phase of the Southern Annular Mode (SAM)** since the 1970s and the associated strengthening and southward shift of the Southern Hemispheric extratropical jet in austral summer.

Human influence has **not affected the principal tropical modes** of interannual climate variability (like ENSO) or their regional teleconnections beyond the range of internal variability (*high confidence*). There is *medium confidence* that anthropogenic and volcanic aerosols contributed to observed changes in the Atlantic Multi-decadal Variability (AMV) index since the 1960s, but there is *low confidence* in the magnitude of this influence.

---
# 1. Scope

This chapter assesses the degree to which human influence has affected the climate system and the ability of climate models to simulate observed mean climate, its changes, and its variability. This assessment provides a foundation for understanding the current impacts of anthropogenic climate change and informs confidence in future climate projections. To attribute observed climate change, an estimate of the expected response to human influence is required, along with estimates of the climate's evolution due to natural forcings and internal variability.

Unlike previous IPCC reports, which separated model evaluation and attribution into different chapters, AR6 integrates these comparisons. This approach allows for a more holistic discussion of inconsistencies between simulations and observations, considering factors like missing forcings, errors in the simulated response, and observational errors. However, if a model's simulation of historical climate has been tuned to match observations, or if models in an attribution study were selected based on the realism of their climate response, these factors must be considered, and attribution results tempered accordingly. This chapter assesses the detection of anthropogenic influence on large spatial and long temporal scales, which is distinct from the emergence of climate change signals from internal variability on local scales and shorter timescales.

---
# 2. Methods

## 2.1 Methods Based on Regression

Regression-based techniques, also known as **fingerprinting methods**, are widely used to detect climate change and attribute it to different external drivers. In this approach, an observed change is assumed to be a linear combination of externally forced signals (the "fingerprints") and internal variability. A regression coefficient that is significantly greater than zero implies a detectable change has been identified in observations. When the confidence interval for this coefficient includes one and is inconsistent with zero, the magnitude of the model-simulated response is considered consistent with observations, implying the change can be attributed in part to that forcing. To ensure robustness, the model-simulated internal variability is typically checked against the observed residual variance using a residual consistency test.

Recent methodological advances include:
* **Integrated Approaches:** Procedures that treat all sources of uncertainty (observational error, model error, internal variability) within a single statistical model, avoiding preliminary dimension-reduction steps. Similar integrated approaches have been developed using Bayesian model averaging.
* **Improved Confidence Intervals:** Bootstrap methods have been suggested to better estimate the confidence intervals of scaling factors, particularly in weak-signal regimes.
* **Pattern Similarity:** An alternative fingerprinting approach uses pattern similarity, where observations are projected onto a time-evolving fingerprint (often the leading empirical orthogonal function from a multi-model forced simulation) to measure the degree of similarity with the expected physical response. To handle high-dimensional daily data, statistical learning techniques like regularized linear regression have been proposed to optimize the signal by down-weighting regions with large internal variability.

## 2.2 Other Probabilistic Approaches

Other probabilistic methods have been introduced to better account for climate modeling uncertainties. These include statistical frameworks based on likelihood maximization, which test whether observations are consistent with the expected response from models, and the application of causal theory to calculate the probability of causation for a given forcing.

---
# 3. Human Influence on the Atmosphere and Surface

## 3.1 Temperature

### 3.1.1 Surface Temperature

#### 3.1.1.1 Model Evaluation

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

#### 3.1.1.2 Detection and Attribution

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


### 3.1.2 Upper-air Temperature

#### 3.1.2.1 Tropospheric temperature

Studies continue to show an inconsistency between simulated and observed temperature trends in the tropical troposphere, with models generally simulating more warming than has been observed. This warming bias persists in CMIP6 models despite the use of updated forcing estimates. The discrepancy is linked to several factors:
* **Climate Sensitivity and Feedbacks:** Differences in climate sensitivity among models may be an important factor. One study suggested that the absence of a hypothesized negative tropical cloud feedback could explain half of the warming bias in one model.
* **Sea Surface Temperature (SST) Biases:** Much of the difference in tropospheric temperature trends is due to corresponding differences in SST warming trends in regions of deep convection. The bias in tropospheric warming is partly linked to biases in surface temperature warming.

Taking these factors into account, there is *medium confidence* that models overestimate observed warming in the upper tropical troposphere over the 1979–2014 period by at least $0.1^{\circ}C$ per decade.

{% raw %}
<img src="/images/IPCC_AR6_Ch3/Ch3_9.png" alt="Figure 3.10">
{% endraw %}

#### 3.1.2.2 Stratospheric temperature

Increased greenhouse gases cause warming near the surface but cooling in the stratosphere. It was assessed in AR5 that anthropogenic forcing, dominated by ozone-depleting substances (ODS), *very likely* contributed to the cooling of the lower stratosphere since 1979.

Since AR5, research has shown that in the middle and upper stratosphere, the cooling since 1979 is mainly due to changes in greenhouse gas concentrations. Volcanic eruptions and the solar cycle have short-term influences but do not affect long-term trends. Based on this stronger evidence, it is now assessed as *extremely likely* that stratospheric ozone depletion from ODS was the **main driver of the cooling of the lower stratosphere between 1979 and the mid-1990s**.

---
# Cross-Chapter Box 3.1 | Global Surface Warming Over the Early 21st Century

The AR5 found that the rate of GMST increase from 1998–2012 was slower than the rate from 1951–2012 and slower than projected by CMIP5 models. This apparent slowdown was assessed with *medium confidence* to be caused in roughly equal measure by a cooling contribution from internal variability and a reduced trend in external forcing (from solar and volcanic activity).

Since AR5, updated observational datasets consistently find **stronger positive trends for 1998–2012** than those available for AR5, mainly due to improved SST datasets and better coverage of the rapidly warming Arctic. The updated observed trends now lie within the range of simulated trends from both CMIP5 and CMIP6 ensembles (*high confidence*).

The slowdown was a **temporary event** (*very high confidence*), induced by a combination of internal variability (particularly a negative phase of Pacific Decadal Variability) and natural forcing (moderate volcanic eruptions and a prolonged solar minimum) that partly offset the ongoing anthropogenic warming trend (*high confidence*). Throughout this period, the planet's total heat content continued to rise, as evidenced by uninterrupted warming of the global ocean (*very high confidence*). A major El Niño event in 2014–2016 marked the end of the slower warming period, leading to three consecutive years of record annual GMST.

{% raw %}
<img src="/images/IPCC_AR6_Ch3/Ch3_CB_1.png" alt="CB 3.1 Figure 1">
{% endraw %}

---

## 3.2 Precipitation, Humidity and Streamflow

### 3.2.1 Paleoclimate Context

Attributing changes in precipitation and other hydrological variables is uniquely challenging because their **large internal variability** often masks the anthropogenic signal. This low signal-to-noise ratio makes it difficult for a clear human-caused signal to emerge from the background of natural fluctuations. Furthermore, the sign of any change depends heavily on the specific location and time of year. While hydrological proxy data represent regional conditions, they can be synthesized to reveal large-scale patterns and provide crucial long-term context.

* **Evidence of Unprecedented Change:** Tree-ring records show that recent prolonged dry spells in the Levant and Chile are unprecedented in the last millennium (*high confidence*). This may be a signature of anthropogenic forcing, but the signal cannot yet be fully distinguished from natural variability.
* **Large Natural Swings:** Paleorecords also show that **large-magnitude changes in the water cycle can occur irrespective of human influence**. For instance, western North America experienced prolonged megadroughts throughout the last millennium that were more severe than 20th and 21st-century events and were associated with internal climate variability.
* **Model Performance:** Climate models are only partially successful at simulating past hydrological changes. One analysis concluded that CMIP5 models show little improvement over earlier versions in simulating rainfall during the Last Glacial Maximum (LGM) and mid-Holocene, despite having higher resolution. However, some studies suggest that model performance improves when changes in vegetation and dust are prescribed, indicating that **vegetation feedbacks in models may be too weak** (*low confidence*). The wide range of model responses, particularly during glacial times, is also linked to how different models parameterize the effect of exposed tropical shelves on atmospheric convection.

{% raw %}
<img src="/images/IPCC_AR6_Ch3/Ch3_10.png" alt="Figure 3.11">
{% endraw %}

{% raw %}
<img src="/images/IPCC_AR6_Ch3/Ch3_11.png" alt="Figure 3.12">
{% endraw %}

### 3.2.2 Atmospheric Water Vapour

Changes in atmospheric water vapour are a critical component of the climate system, particularly in the upper troposphere, where water vapour strongly regulates the strength of the **water-vapour feedback**.

* **Model and Observational Challenges:** Model biases in simulating water vapour are dominated by errors in relative humidity, which are closely linked to errors in simulating large-scale circulation. At the same time, many key observational records of water vapour are affected by data inhomogeneity issues and must be used with caution.
* **A Detectable Human Fingerprint:** Despite these challenges, the human fingerprint on atmospheric moisture is clear and governed by robust physical processes. Attribution studies using CMIP5 models found that conclusions about human influence were not sensitive to model quality. Work by Chung et al. (2014) showed that the observed moistening of the upper troposphere from 1979–2005 could not be explained by internal variability alone and is attributable to a combination of anthropogenic and natural forcings.

In summary, the assessment is that it is ***likely*** **that human influence has contributed to moistening in the upper troposphere since 1979**. Additionally, there is *medium confidence* that human influence has contributed to a global increase in annual surface specific humidity and to a decrease in surface relative humidity over mid-latitude Northern Hemisphere continents in summer.

### 3.2.3 Precipitation

While climate models successfully capture the large-scale patterns of annual mean precipitation, significant challenges remain in simulating the magnitude and seasonality of recent changes.

* **Model Performance:** The amplitude of simulated precipitation trends is often **much smaller than observed**. While models are consistent with observed changes in the *phase* of the annual cycle (e.g., an earlier onset of the wet season in the extratropics), they struggle to reproduce the magnitude of seasonal changes. CMIP6 models show minor improvements over CMIP5, but persistent biases like the **double Inter-Tropical Convergence Zone (ITCZ)** remain. Increasing model resolution has been shown to reduce some of these biases, but overall, models still have deficiencies in simulating precipitation patterns, particularly over the tropical oceans (*high confidence*).
* **Intensifying Water Cycle Patterns:** An expected response to warming is an intensification of the wet-dry zonal pattern (wet tropics and mid-latitudes getting wetter, dry subtropics getting drier). Detecting this is complicated by model errors, but pattern-based fingerprinting studies that track the movement of these zones have successfully found a detectable human signal. The observed strengthening of this precipitation contrast is significantly larger than what is simulated in the multi-model mean.
* **Global Trends:** Global land precipitation has *likely* increased since the mid-20th century (*medium confidence*). However, there has been **no significant trend in global *mean* precipitation** during the satellite era. This is consistent with our physical understanding, as the warming-induced increase is partly offset by the cooling effect of anthropogenic aerosols and the fast atmospheric adjustment to greenhouse gases. Evidence for a changing water cycle is also clear over the ocean, where coherent patterns of **fresh regions getting fresher and salty regions getting saltier** are consistent with the "wet-gets-wetter, dry-gets-drier" paradigm.

New attribution studies have strengthened previous findings of a detectable increase in mid- to high-latitude land precipitation over the Northern Hemisphere (*high confidence*). There is *medium confidence* that human influence has contributed to a strengthening of the contrast between the wet tropics and dry subtropics. However, due to observational uncertainties and inconsistent results, there is *low confidence* in attributing changes in the seasonality of precipitation.

{% raw %}
<img src="/images/IPCC_AR6_Ch3/Ch3_12.png" alt="Figure 3.13">
{% endraw %}

{% raw %}
<img src="/images/IPCC_AR6_Ch3/Ch3_13.png" alt="Figure 3.14">
{% endraw %}

{% raw %}
<img src="/images/IPCC_AR6_Ch3/Ch3_14.png" alt="Figure 3.15">
{% endraw %}

### 3.2.4 Streamflow

Streamflow is the best-observed variable within the terrestrial water cycle for conducting global-scale detection and attribution analyses.

* **Regional Attribution:** The AR5 concluded with *medium confidence* that human influence had affected streamflow in some mid- and high-latitude regions. More recent work supports this, such as a pan-European assessment that attributed the pattern of decreasing streamflow in southern Europe and increasing streamflow in northern Europe to anthropogenic climate change, while acknowledging that direct human water withdrawals also play a role.
* **Global Attribution:** A recent global study found that the observed global pattern of streamflow trends—with some regions drying and others wetting—is consistent with model simulations **only when anthropogenic climate change is included**.

In summary, there is ***medium confidence*** **that anthropogenic climate change has altered local and regional streamflow** in various parts of the world. The associated global-scale trend pattern is inconsistent with internal variability. Furthermore, while direct human interventions like water withdrawals affect streamflow, they cannot explain the observed large-scale spatio-temporal trends (*medium confidence*).

## 3.3 Atmospheric Circulation

### 3.3.1 The Hadley and Walker Circulations

The tropical troposphere is dominated by two major overturning circulations: the meridional Hadley circulation and the zonal Walker circulation. AR5 found *medium confidence* that stratospheric ozone depletion had contributed to a widening of the Hadley cell in the Southern Hemisphere summer. It also noted a contradiction for the Walker circulation: while models simulated a weakening in response to greenhouse gases, observations since the early 1990s showed a strengthening, precluding any clear detection of human influence.

The latest assessment builds on these findings:

* **Hadley Cell Expansion:** It is now considered ***likely*** **that human influence has contributed to the poleward expansion of the Hadley cell in the Southern Hemisphere since the 1980s**. During the period of strong ozone depletion (1981–2000), this human influence is detectable in the summertime expansion with *medium confidence*. In contrast, the observed expansion of the Northern Hemisphere Hadley cell is assessed to be within the range of internal variability (*medium confidence*), with contributions from Pacific Decadal Variability (PDV) and other natural patterns.
* **Walker Circulation Strengthening:** The causes of the observed strengthening of the Pacific Walker circulation from 1980–2014 are still not well understood. The observed strengthening trend lies **outside the range of variability simulated in coupled models** (*medium confidence*). Confidence in these assessments is limited by large observational uncertainty, a lack of complete understanding of the mechanisms driving Hadley cell expansion, and contradicting theories regarding the influence of greenhouse gases and aerosols on the Walker circulation's strength.

{% raw %}
<img src="/images/IPCC_AR6_Ch3/Ch3_15.png" alt="Figure 3.16">
{% endraw %}

### 3.3.2 Global Monsoon

The global monsoon system is a dominant mode of annual climate variation in the tropics. Its behavior is shaped by a combination of natural and anthropogenic factors.

* **Paleoclimate Context:** In past climates, it is assessed with *high confidence* that **orbital forcing produced strong interhemispheric rainfall variability**, which is evident in multiple proxy records. For the warmer mid-Pliocene period, both orbital forcing and higher $CO_2$ concentrations played an important role in monsoon expansion and intensification. However, while models capture these qualitative changes, they tend to **underestimate the magnitude of monsoon expansion** found in proxy reconstructions. This may be linked to biases in the models' representation of the monsoon domain, which could be improved by imposing changes in vegetation and dust.
* **Forcing in the Last Millennium:** Studies of the last millennium show that while simulated global monsoon precipitation increases with global mean temperature, the specific response of circulation and regional rainfall depends on the forcing. **Volcanic forcing is found to be more effective** than greenhouse gas or solar variations in driving changes in global monsoon precipitation over this period.
* **Historical Simulation & Biases:** Reproducing the precise domain, amount, and timing of monsoons in the historical period remains difficult for models. CMIP5 models broadly capture the global monsoon domain but tend to underestimate its extent in some regions (East Asia, North America) while overestimating it in others (tropical western North Pacific). Even with increased resolution, improvements in monsoon simulation are not uniform across all regions and models.
* **Attribution of Recent Trends:**
    * Instrumental records show that global summer monsoon precipitation **decreased from the 1950s to the 1980s, followed by an increase**, driven mainly by the Northern Hemisphere land monsoons.
    * Modeling studies indicate that while greenhouse gas increases act to *enhance* monsoon precipitation, this effect was **overwhelmed by the influence of anthropogenic aerosols** in both CMIP5 and CMIP6 models, which drove the mid-century weakening.
    * The recent intensification since the 1980s appears to be dominated by **internal variability**, including the PDV and the Atlantic Multi-decadal Variability (AMV). The observed strengthening of the monsoon circulation is opposite to the simulated response to greenhouse gas forcing, further pointing to a dominant role for natural patterns. However, it is noted that CMIP5 models tend to under-represent the magnitude of the PDV, which may lead to overconfidence in detecting the forced aerosol signal.

In summary, while greenhouse gas increases acted to enhance global land monsoon precipitation over the 20th century (*medium confidence*), this was **overwhelmed by anthropogenic aerosols from the 1950s to the 1980s** (*medium confidence*). The subsequent intensification since the 1980s is considered to be dominated by internal variability. Models, including those in CMIP6, still fail to fully capture the observed variations of the Northern Hemisphere summer monsoon circulation.

{% raw %}
<img src="/images/IPCC_AR6_Ch3/Ch3_16.png" alt="Figure 3.17">
{% endraw %}

### 3.3.4 Sudden Stratospheric Warming Activity

Sudden stratospheric warmings (SSWs) are major meteorological events in the winter polar stratosphere, characterized by a rapid rise in temperature that can disrupt the polar vortex and influence surface weather for weeks to months.

* **Model Performance:** The ability of models to simulate SSWs is highly dependent on their representation of the stratosphere. Models with limited stratospheric resolution, which constituted more than half of the CMIP5 ensemble, **underestimate the frequency of SSWs**. Even in stratosphere-resolving models, the surface pressure response during an SSW can differ substantially from that seen in reanalysis products.
* **Attribution:** The observational record is short and shows substantial decadal variability, leading to low confidence in any observed trends. To date, an **anthropogenic influence on the frequency or other aspects of SSWs has not been robustly detected**.


---


# 4. Human Influence on the Cryosphere

## 4.1 Sea Ice

### 4.1.1 Arctic Sea Ice

The AR5 concluded that anthropogenic forcings were *very likely* to have contributed to Arctic sea ice loss since 1979. The evidence for this has since strengthened.

* **Role of Internal Variability:** While human forcing is the main driver, internal variability plays an important role, accounting for an estimated **40–50% of the observed decline in September sea ice**. This variability is linked to summer atmospheric circulation patterns, such as enhanced ridges over Greenland, and teleconnections from decadal variability in the tropics. However, extreme events, like the record low sea ice extent in 2012, are inconsistent with internal climate variability alone.
* **Model Performance:** All CMIP5 and CMIP6 models capture the Arctic sea ice loss. However, there is a large spread in the rate of decline among models. This has been attributed to several factors, including the model's initial simulated sea ice state, the magnitude of simulated ocean heat transport, and the model's overall rate of global warming.
* **Overall Assessment:** It is ***very likely*** **that anthropogenic forcing, mainly from greenhouse gas increases, was the main driver of Arctic sea ice loss since 1979**.

{% raw %}
<img src="/images/IPCC_AR6_Ch3/Ch3_19.png" alt="Figure 3.20">
{% endraw %}


### 4.1.2 Antarctic Sea Ice

The Antarctic presents a more complex picture. Over the 1979–2017 period, Antarctic sea ice cover showed a small, statistically insignificant increase, followed by an anomalous and rapid decrease starting in late 2016. This recent decrease has been linked to anomalous atmospheric conditions driven by teleconnections from the tropics.

* **Model-Observation Discrepancy:** Unlike in the Arctic, most climate models simulate an overall *decrease* in Antarctic sea ice over the satellite era, in direct contrast to the observed slight increase through 2015. This discrepancy is a major challenge.
* **Potential Drivers and Uncertainties:**
    * The observed trends in some seasons and regions appear to be **outside the range of simulated internal variability**, suggesting that models may have an unrealistic forced response or are underestimating the true range of natural variability.
    * Model discrepancies have been linked to biases in simulating Southern Ocean winds, upper ocean temperatures, and the sea ice response to the Southern Annular Mode.
    * One proposed mechanism for the past sea ice expansion is an **increase in freshwater fluxes** from the melting Antarctic Ice Sheet, though there is no consensus on this.
* **Overall Assessment:** A combination of large multi-decadal variability, a short observational record, and limited model performance has resulted in a weak signal-to-noise ratio, precluding a confident attribution of Antarctic sea ice trends.


{% raw %}
<img src="/images/IPCC_AR6_Ch3/Ch3_20.png" alt="Figure 3.21">
{% endraw %}


## 4.2 Snow Cover

Seasonal snow cover is a defining feature of the northern continents and a key component of the climate system.

* **Model Performance:** CMIP5 models noted a strong linear correlation between Northern Hemisphere snow cover extent (SCE) and surface air temperature. However, they tended to overestimate the albedo of snow-covered boreal forests, which may have led to an overestimation of the snow-albedo feedback. CMIP6 models have improved, simulating a slightly stronger decline in spring SCE since 1981 that is in better agreement with observations. A remaining concern is the failure of most CMIP6 models to correctly represent the relationship between snow cover *extent* and snow *mass*.
* **Attribution:** It is ***very likely*** **that anthropogenic influence contributed to the observed reductions in Northern Hemisphere springtime snow cover since 1950**. Both CMIP5 and CMIP6 models simulate strong declines in spring SCE in recent decades, in general agreement with observations.

{% raw %}
<img src="/images/IPCC_AR6_Ch3/Ch3_21.png" alt="Figure 3.22">
{% endraw %}


## 4.3 Glaciers and Ice Sheets

### 4.3.1 Glaciers

The AR5 assessed that human influence had *likely* contributed to the retreat of glaciers since the 1960s. The evidence is now stronger. Glacier models driven by historical simulations that include only natural forcings show a net glacier *growth*, in stark contrast to the observed widespread retreat. When driven by simulations including both natural and anthropogenic forcings, the models show mass loss. One study estimated that **at least 85% of cumulative glacier mass loss since 1850 is attributable to anthropogenic influence**.

### 4.3.2 Ice Sheets

#### 4.3.2.1 Greenland Ice Sheet

The AR5 assessed it was *likely* that anthropogenic forcing contributed to Greenland's surface melting since 1993. The evidence has since mounted. The rate of mass loss observed in the 21st century appears unprecedented over long timescales; proxy evidence and modeling suggest the last time the rate was similar was during the early Holocene.

While internal variability, such as the summer North Atlantic Oscillation (NAO), plays an important role in year-to-year variations, the long-term trend of increased surface melting is strongly associated with warming. The observed trend in surface mass balance loss from 2007–2017 was at the top of the range projected by AR5 models, providing further evidence of a consistent, anthropogenically forced trend.

#### 4.3.2.2 Antarctic Ice Sheet

The primary driver of mass loss from the Antarctic Ice Sheet is the acceleration, retreat, and thinning of major outlet glaciers in West Antarctica, which is in turn driven by the melting of ice shelves by warm ocean waters.

* **Attribution Challenges:** Attributing these changes is difficult. The basal melt that drives ice discharge is influenced by wind-driven ocean currents, which may be partly anthropogenic, but there is no consensus on whether the recent changes are a forced trend or a manifestation of internal variability.
* **Model Confidence:** While models have *medium confidence* in representing many ice-sheet processes, there is *low confidence* in their ability to simulate the ocean forcing that controls basal melt rates.


---


# 5. Human Influence on the Ocean

The global ocean plays a critical role in the climate system by storing and transporting vast amounts of heat, freshwater, and carbon. Accurate simulation of the ocean is therefore essential for realistic climate projections. This section assesses the performance of models in simulating key ocean properties and the attribution of observed changes to human influence.

## 5.1 Ocean Temperature

This section assesses the performance of climate models in representing the mean state ocean temperature and heat content.

### 5.1.1 Sea Surface and Zonal Mean Ocean Temperature Evaluation

There has been growing evidence that increasing the horizontal resolution of both the ocean and atmosphere components in models can improve the simulation of mean ocean temperatures. Higher resolution allows for the explicit representation of processes like mesoscale eddies, which can improve vertical heat transport and reduce temperature drifts in the deep ocean. Despite this, there is **little overall improvement in the multi-model mean biases** for sea surface and zonal mean ocean temperatures between CMIP5 and CMIP6 (*medium confidence*).

{% raw %}
<img src="/images/IPCC_AR6_Ch3/Ch3_22.png" alt="Figure 3.23">
{% endraw %}

{% raw %}
<img src="/images/IPCC_AR6_Ch3/Ch3_23.png" alt="Figure 3.24">
{% endraw %}

{% raw %}
<img src="/images/IPCC_AR6_Ch3/Ch3_2.png" alt="Figure 3.25">
{% endraw %}

### 5.1.2 Tropical Sea Surface Temperature Evaluation

#### 5.1.2.1 Tropical Pacific Ocean

Persistent biases in the tropical Pacific, such as the excessive equatorial cold tongue, have been linked to misrepresentations of the ocean-atmosphere Bjerknes feedback, vertical mixing, and surface winds.

#### 5.1.2.2 Tropical Atlantic Ocean

Major biases in the tropical Atlantic, such as the east-west temperature gradient, also persist in CMIP6. While higher resolution can help, some lower-resolution models also perform well, suggesting that **resolution is not the only factor** responsible for these biases.

#### 5.1.2.3 Tropical Indian Ocean

A common problem in both CMIP5 and CMIP6 is a **warm bias in the subsurface ocean**, particularly around the depth of the thermocline.

#### 5.1.2.4 Summary

The structure and magnitude of multi-model mean ocean temperature biases have not changed substantially between CMIP5 and CMIP6 (*medium confidence*). Nevertheless, CMIP5 and CMIP6 models are considered appropriate tools for investigating ocean temperature and heat content responses to forcing.

### 5.1.3 Ocean Heat Content Change Attribution

The ocean has acted as the Earth's primary energy store, absorbing an estimated **91% of the total energy change** in the climate system from 1971 to 2018.

* **Warming Throughout the Water Column:** It is *virtually certain* that the global upper ocean (0–700 m) has warmed substantially since 1971, and *very likely* that the intermediate ocean (700–2000 m) has also warmed. The rate of ocean heat uptake has roughly **doubled in the past few decades** compared to the rate over the full 20th century.
* **Model-Observation Consistency:** Both CMIP5 and CMIP6 simulations show robust increases in ocean heat content in the upper and intermediate layers (*high confidence*), consistent with observations. The simulated warming is also reflected in thermal expansion (thermosteric sea level rise).
* **Attribution:** A response to **anthropogenic forcing is clearly detectable** in ocean heat content, and natural forcing alone cannot explain the observed changes. The influence of greenhouse gas forcing is detectable independently from other anthropogenic forcings.

It is ***extremely likely*** **that human influence was the main driver of the ocean heat content increase observed since the 1970s**.

{% raw %}
<img src="/images/IPCC_AR6_Ch3/Ch3_25.png" alt="Figure 3.26">
{% endraw %}


## 5.2 Ocean Salinity

The AR5 assessed that it was *very likely* that anthropogenic forcings contributed to ocean salinity changes since the 1960s. The SROCC strengthened this, noting that high-latitude freshening and warming have *very likely* made the surface ocean less dense, increasing stratification. While salinity does not have a direct feedback to the atmosphere, some studies have identified indirect feedbacks through sea-salt aerosol interactions that can influence precipitation and tropical cyclones.

### 5.2.1 Sea Surface and Depth-profile Salinity Evaluation

Models generally capture the large-scale patterns of ocean salinity, with features like the salty Atlantic and fresher Pacific aligning with patterns of evaporation minus precipitation. However, persistent biases exist, particularly a fresh bias in the upper ocean. These biases are primarily linked to biases in the simulated precipitation field. Despite these issues, the biases are considered sufficiently small to allow for the use of these models in detection and attribution studies.

### 5.2.2 Salinity Change Attribution

It is ***extremely likely*** **that human influence has contributed to observed near-surface and subsurface salinity changes** across the globe since the mid-20th century. All available assessments confirm that the pattern of change corresponds to **fresh regions becoming fresher and salty regions becoming saltier** (*high confidence*), providing powerful evidence of an intensifying global water cycle.


{% raw %}
<img src="/images/IPCC_AR6_Ch3/Ch3_26.png" alt="Figure 3.27">
{% endraw %}

{% raw %}
<img src="/images/IPCC_AR6_Ch3/Ch3_27.png" alt="Figure 3.28">
{% endraw %}

## 5.3 Sea Level

Global Mean Sea Level (GMSL) rise is a result of both the thermal expansion of ocean water (thermosteric change) and the addition of mass from melting glaciers, ice sheets, and changes in terrestrial water storage.

### 5.3.1 Sea Level Evaluation

The current generation of climate models does not fully resolve all components needed to close the sea level budget (e.g., dynamic ice sheets, glaciers). Therefore, analyses often focus on the thermosteric component, which is well-simulated, and use offline models driven by climate model output to estimate contributions from ice melt. A key advance in CMIP6 is the coordinated inclusion of dynamic ice sheet models through the Ice Sheet Model Intercomparison Project (ISMIP6), which will help to provide more comprehensive projections.

### 5.3.2 Sea Level Attribution

The SROCC concluded with *high confidence* that the dominant cause of GMSL rise since 1970 is anthropogenic forcing.

* **Thermosteric Component:** Studies show that **anthropogenic forcings are essential for explaining the magnitude of observed thermosteric sea level rise**. One study found that human influence explains 87% of the thermosteric rise in the top 700 m from 1950–2005. It is ***very likely*** **that anthropogenic forcing was the main driver** of the observed global mean thermosteric sea level increase since 1970.
* **Cryosphere Contribution:** One study found that about 69% of glacier mass loss from 1991-2010 was anthropogenic.
* **Total Sea Level:** Synthesizing these points to explain the final assessment: **it is *very likely* that anthropogenic forcing is the main driver of GMSL rise since at least 1971.**


{% raw %}
<img src="/images/IPCC_AR6_Ch3/Ch3_28.png" alt="Figure 3.29">
{% endraw %}

## 5.4 Ocean Circulation

Ocean circulation is crucial for transporting heat and freshwater throughout the climate system. The AR5 reported that while models suggested an AMOC slowdown would occur due to anthropogenic forcing, the observational record was too short to confirm this.

### 5.4.1 Atlantic Meridional Overturning Circulation (AMOC)

The AMOC mean state is reasonably well simulated in models, but there is a large spread in its strength, and models tend to underestimate its depth and its interannual-to-decadal variability. While observations show a weakening from the mid-2000s to mid-2010s, this is largely attributed to internal variability, as the forced response in models is much weaker. Due to the short and sparse observational record and significant model biases, there is *low confidence* that anthropogenic forcing has had a significant influence on observed AMOC changes.


{% raw %}
<img src="/images/IPCC_AR6_Ch3/Ch3_29.png" alt="Figure 3.30">
{% endraw %}


### 5.4.2 Southern Ocean Circulation

The representation of the Antarctic Circumpolar Current (ACC) has improved from CMIP5 to CMIP6, converging toward observed estimates. However, models still struggle with the strength of the upper and lower overturning cells. The lack of interactive ice-sheet coupling is another important limitation. Due to substantial observational uncertainty and remaining model challenges, attribution of observed Southern Ocean circulation changes is not yet possible.


# 6. Human Influence on the Biosphere

## 6.1 Terrestrial Carbon Cycle

The IPCC Special Report on Climate Change and Land (SRCCL) assessed with *high confidence* that global vegetation photosynthetic activity has increased over the last two to three decades. This "greening" of the Earth has been attributed to a combination of drivers, including direct land use and management changes, nitrogen deposition, increased diffuse radiation, climate change, and importantly, $CO_2$ fertilization (*high confidence*).

* **Model Performance and Carbon Sinks:** While earlier literature emphasized the importance of nitrogen limitation on plant growth, more recent studies note that models without a nitrogen cycle can still be consistent with historical carbon cycle trends. Earth system models simulate the **globally averaged land carbon sink** within the range of observation-based estimates (*high confidence*). However, this global agreement masks **large regional disagreements**, with models tending to underestimate the Northern Hemisphere sink. Spatially, the land carbon uptake is dominated by the extratropical northern latitudes, partly because the tropics may have become a net source of carbon due to deforestation.
* **The Atmospheric $CO_2$ Seasonal Cycle:** The seasonal cycle of atmospheric $CO_2$, driven by vegetation growth in summer and respiration in winter, has **increased in amplitude** since systematic monitoring began. This trend, first reported by Keeling et al. (1996) for Mauna Loa, is larger at higher latitudes of the Northern Hemisphere and has continued.
* **Attribution of Seasonal Change:** The primary cause of this increasing seasonal amplitude has been identified through attribution studies. The assessment is that **fertilization of plant growth by anthropogenic increases in atmospheric $CO_2$ is the main driver** (*medium confidence*).


{% raw %}
<img src="/images/IPCC_AR6_Ch3/Ch3_30.png" alt="Figure 3.31">
{% endraw %}

{% raw %}
<img src="/images/IPCC_AR6_Ch3/Ch3_31.png" alt="Figure 3.32">
{% endraw %}
## 6.2 Ocean Biogeochemical Variables

The ocean's biogeochemistry is undergoing significant changes as a direct consequence of anthropogenic emissions.

* **Ocean Deoxygenation:** Oxygen concentrations in the open ocean have decreased since 1960. While temperature-driven solubility changes play a role, the main drivers of observed deoxygenation appear to be changes in **ocean circulation, mixing, and/or biochemical processes**. Attribution studies conclude that this deoxygenation is due in part to anthropogenic forcing (*medium confidence*).
* **Ocean Carbon Uptake:** In the subtropical and equatorial North Atlantic, warming has been a significant contributor to the observed increase in the partial pressure of $CO_2$ ($pCO_2$) since the mid-2000s. This long-term warming is leading to a **reduction in the ocean's capacity to absorb carbon**. In other regions, like the Southern Ocean, the signal is harder to detect due to strong natural variability.
* **Ocean Acidification:** The uptake of anthropogenic $CO_2$ is the most significant driver of chemical changes in the ocean. The AR5 concluded that this had *very likely* resulted in the acidification of surface waters. The evidence is now stronger. It is ***virtually certain*** **that the uptake of anthropogenic $CO_2$ was the main driver of the observed acidification of the global surface open ocean**. This acidification is not just a surface phenomenon but has been observed down to depths of 3000 m in some regions. The observed increase in acidification in parts of the North Atlantic since 2000 is *likely* associated in part with rising ocean temperatures, which corresponds to the expected weakening of the ocean carbon sink as the climate warms.

# 7. Human Influence on Modes of Climate Variability

## 7.1 Northern Annular Mode (NAM)
* **The Northern Annular Mode (NAM)**, also known as the Arctic Oscillation, describes an oscillation of atmospheric mass between the Arctic and the northern mid-latitudes. It is the leading mode of wintertime variability in the northern extratropics. The **North Atlantic Oscillation (NAO)** is considered the regional expression of the NAM over the North Atlantic/European sector.

There is limited evidence for a significant role for anthropogenic forcings in driving the observed multi-decadal variations of the NAM/NAO since the mid-20th century. **Confidence in attribution is low** for several key reasons:
1.  There is a large spread in the modelled forced responses, which is overwhelmed by strong internal variability.
2.  There is an apparent "signal-to-noise paradox," which suggests the response of the NAM/NAO to forcings may be too weak in models.
3.  Models have a chronic inability to produce a range of long-term trends that encompasses the observed estimates over the last 60 years.

{% raw %}
<img src="/images/IPCC_AR6_Ch3/Ch3_32.png" alt="Figure 3.33">
{% endraw %}

{% raw %}
<img src="/images/IPCC_AR6_Ch3/Ch3_33.png" alt="Figure 3.34">
{% endraw %}

## 7.2 Southern Annular Mode (SAM)
* **The Southern Annular Mode (SAM)** consists of a north-south redistribution of atmospheric mass around Antarctica, associated with a corresponding shift in the jet stream and surface westerlies over the Southern Ocean.

In contrast to its northern counterpart, the human fingerprint on the SAM is clear. It is ***very likely*** **that anthropogenic forcings have contributed to the observed trend of the summer SAM toward its positive phase since the 1970s**. This assessment is supported by improved models since AR5. The primary drivers have been identified:
* **Stratospheric ozone depletion** was the dominant driver of the positive trend from the 1970s to the 1990s (*medium confidence*).
* Since 2000, the influence of ozone depletion has been small (due to the effects of the Montreal Protocol), leading to a **weaker summertime SAM trend** over the 2000–2019 period (*medium confidence*).


{% raw %}
<img src="/images/IPCC_AR6_Ch3/Ch3_34.png" alt="Figure 3.35">
{% endraw %}

## 7.3 El Niño-Southern Oscillation (ENSO)
* **The El Niño-Southern Oscillation (ENSO)**, generated by coupled ocean-atmosphere interactions in the tropical Pacific, is the world's most dominant mode of interannual climate variability, influencing weather patterns globally.

While proxy records suggest ENSO amplitude since 1950 has been higher than in the period from 1400–1850 (*medium confidence*), large-ensemble model simulations suggest that such changes are consistent with internal variability. Key findings include:
* **Attribution of Variability:** Models do not show strong trends in ENSO variability over the historical period in response to external forcing. This suggests that the observed changes have not yet exceeded the magnitude of natural variability. Therefore, there is ***low confidence*** **that anthropogenic forcing has led to the changes in ENSO variability** inferred from paleo-proxy evidence.
* **Teleconnections:** Models are reasonably successful at simulating ENSO's remote impacts (teleconnections). While simulating the exact spatial patterns is challenging due to noise from internal atmospheric variability, models show high agreement with observations in the **sign and regional average strength** of these connections.
* **Model Biases:** A persistent issue in models is that key physical feedbacks are not well balanced. The atmospheric **Bjerknes feedback** (which promotes event growth) is often too weak, while the **surface heat flux feedback** (which dampens events) is also often too weak. These compensating errors allow models to produce ENSO-like variability, but for the wrong reasons.

{% raw %}
<img src="/images/IPCC_AR6_Ch3/Ch3_35.png" alt="Figure 3.36">
{% endraw %}

{% raw %}
<img src="/images/IPCC_AR6_Ch3/Ch3_36.png" alt="Figure 3.37">
{% endraw %}

{% raw %}
<img src="/images/IPCC_AR6_Ch3/Ch3_37.png" alt="Figure 3.38">
{% endraw %}

## 7.4 Indian Ocean Basin and Dipole Modes (IOB, IOD)

* **The Indian Ocean Basin (IOB) and Dipole (IOD)** modes are the two leading patterns of interannual sea surface temperature (SST) variability in the tropical Indian Ocean. The IOB mode involves basin-wide warming/cooling, while the IOD features an east-west dipole of SST anomalies.

There is *medium confidence* that the observed changes in interannual IOD variability in the late 20th century are **within the range of internal variability**. There is no evidence of an anthropogenic influence on the interannual IOB. On longer, multi-decadal timescales, there is *low confidence* that human influence has caused any change in the relationship between the IOB and other climate modes.

## 7.5 Atlantic Meridional and Zonal Modes (AMM, AZM)

* **The Atlantic Zonal Mode (AZM)**, or Atlantic Niño, and the **Atlantic Meridional Mode (AMM)** are the two leading modes of climate variability in the tropical Atlantic.

Based on both CMIP5 and CMIP6 results, there is **no robust evidence that observed changes in either the AMM or AZM are beyond the range of internal variability** or have been influenced by external forcing. Attribution is severely limited by major, persistent biases in how climate models simulate the tropical Atlantic mean state and its variability.

## 7.6 Pacific Decadal Variability

* **Pacific Decadal Variability (PDV)** is a generic term for decadal-to-interdecadal modes of variability in the Pacific Ocean, which includes the well-known Pacific Decadal Oscillation (PDO) and the Inter-decadal Pacific Oscillation (IPO). PDV was implicated in the temporary slowdown of global surface warming in the early 2000s.

While some studies have suggested a potential link between anthropogenic aerosols and the recent negative phase of the PDV, the evidence remains limited. The temporal correlation between the forced multi-model mean PDV index and observations is negligible, suggesting any externally-driven component is weak. The primary assessment is that **PDV is mostly driven by internal variability**.

Therefore, there is ***high confidence*** **that internal variability has been the main driver of the PDV since pre-industrial times**, despite some modeling evidence for a potential external influence.


{% raw %}
<img src="/images/IPCC_AR6_Ch3/Ch3_38.png" alt="Figure 3.39">
{% endraw %}


## 7.7 Atlantic Multi-decadal Variability

* **Atlantic Multi-decadal Variability (AMV)** refers to basin-wide, multi-decadal fluctuations in North Atlantic surface temperatures. It is associated with a wide range of physical processes, including changes in the Atlantic Meridional Overturning Circulation (AMOC). Along with the PDV, the AMV has modulated global surface temperature on multi-decadal timescales.

There is **robust evidence that external forcings have modulated the AMV** over the historical period. In particular, it is thought that **anthropogenic and volcanic aerosols played a role** in the timing and intensity of the negative (cold) phase of the AMV from the mid-1960s to mid-1990s and the subsequent warming (*medium confidence*).

However, confidence in the *magnitude* of this human influence remains low due to several challenges:
* **Model Evaluation:** It is difficult to evaluate model performance due to short observational records, particularly for key processes like the AMOC. On average, modeled AMV episodes are too short and too weak.
* **Observational Uncertainty:** There is a lack of detailed, long-term, process-based observations (e.g., of AMOC strength, aerosol properties, surface fluxes) that limit our physical understanding of the drivers.
* **Methodological Issues:** Many studies rely on simplistic SST indices that may mask critical inconsistencies between models and observations.

{% raw %}
<img src="/images/IPCC_AR6_Ch3/Ch3_39.png" alt="Figure 3.40">
{% endraw %}

# 8. Synthesis Across Earth System Components

## 8.1 Multivariate Attribution of Climate Change

When the evidence from individual components of the climate system is considered together, the scientific case for human influence becomes overwhelmingly strong. Combining the evidence increases the level of confidence in attribution and reduces the uncertainties associated with looking at any single variable. From this combined evidence, **it is unequivocal that human influence has warmed the atmosphere, ocean and land**.

{% raw %}
<img src="/images/IPCC_AR6_Ch3/Ch3_40.png" alt="Figure 3.41">
{% endraw %}

## 8.2 Multivariate Model Evaluation

This section provides a system-wide assessment of model performance using integrative measures that combine multiple diagnostic fields.


### 8.2.1 Integrative Measures of Model Performance

The purpose of multivariate analysis is to assess how well models simulate present-day and historical climate across a wide range of variables, using metrics like pattern correlation or root-mean-square difference against observations.

* **Model Errors vs. Observational Uncertainty:** As in AR5, the analysis shows that model errors are generally larger than the differences between various observational reference datasets.
* **Progress Across Generations:** An analysis of dozens of atmospheric, land, and ocean variables shows clear progress across model generations, from CMIP3 to CMIP5 to CMIP6. The improved performance across so many different fields enhances confidence that the progress reflects greater physical realism, rather than just a cancellation of errors from model tuning.
* **Complexity and Performance:** High-complexity Earth System Models (ESMs), which include additional biogeochemical feedbacks, often perform as well as or better than their lower-complexity counterparts. This indicates that the interactive simulation of these complex processes is a mature and valuable development in climate modeling.

{% raw %}
<img src="/images/IPCC_AR6_Ch3/Ch3_41.png" alt="Figure 3.42">
{% endraw %}

{% raw %}
<img src="/images/IPCC_AR6_Ch3/Ch3_42.png" alt="Figure 3.43">
{% endraw %}

{% raw %}
<img src="/images/IPCC_AR6_Ch3/Ch3_43.png" alt="Figure 3.44">
{% endraw %}

### 8.2.2 Process Representation in Different Classes of Models

This section explores the influence of two key aspects of model development: resolution and complexity. A key advance in CMIP6 is the systematic inclusion of high-resolution models.

* **Influence of Resolution:** Increasing horizontal resolution improves many aspects of climate simulation, including the representation of ocean currents, atmospheric storms, and sea surface temperatures. However, it is not a panacea. In some regions, like parts of the Southern Ocean, higher resolution can lead to *inferior* performance, highlighting underlying deficiencies in model physics that are not simply related to resolution.
* **Influence of Complexity:** The inclusion of more complex Earth system processes (e.g., nitrogen cycles, interactive atmospheric chemistry, permafrost) and enhanced coupling between components has been a major focus of recent model development. This allows for the investigation of crucial feedbacks (like carbon-nutrient limitations) and interactions (like between climate and air quality policy), which improves confidence in key projections, such as remaining carbon budgets.

In summary, both high-resolution and high-complexity models have been evaluated as part of CMIP6. Higher resolution improves many, but not all, aspects of climate simulation (*high confidence*). At the same time, high-complexity ESMs perform as well as their lower-complexity counterparts, demonstrating that the interactive simulation of key Earth system components is now well established.


---


# Reference

Eyring, V., N.P. Gillett, K.M. Achuta Rao, R. Barimalala, M. Barreiro Parrillo, N. Bellouin, C. Cassou, P.J. Durack, Y. Kosaka, S. McGregor, S. Min, O. Morgenstern, and Y. Sun, 2021: Human Influence on the Climate System. In Climate Change 2021: The Physical Science Basis. Contribution of Working Group I to the Sixth Assessment Report of the Intergovernmental Panel on Climate Change [Masson-Delmotte, V., P. Zhai, A. Pirani, S.L. Connors, C. Péan, S. Berger, N. Caud, Y. Chen, L. Goldfarb, M.I. Gomis, M. Huang, K. Leitzell, E. Lonnoy, J.B.R. Matthews, T.K. Maycock, T. Waterfield, O. Yelekçi, R. Yu, and B. Zhou (eds.)]. Cambridge University Press, Cambridge, United Kingdom and New York, NY, USA, pp. 423–552, doi: 10.1017/9781009157896.005.
