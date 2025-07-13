---
layout: post
title: "Visualization Practice Climatology"
date: 2025-07-13 19:00:00 +0900
tags: [Visualization, Python, Cartopy, Xarray]
---

In addition to previous post, I practiced visualization with python libraries using dataset from PSL in NOAA. All dataset is available in [Physical Sciences Laboratory](https://psl.noaa.gov/data/gridded/).

---

## 1. Setup
```
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import xarray as xr

path = "./Visualization_tutorial_Anomaly/"
```

---

## 2. Draw Zonal Mean of Uwind
```
# Draw Zonal mean

var = xr.open_dataset('../Data/uwnd.mon.mean.nc')
#print(var.uwnd.shape)

mean = var.uwnd.mean(dim=['lon']) # zonal mean
mean.shape

# Plot
mean.sel(time='2010-01-01').plot.contourf(levels=50, cmap='jet')
plt.gca().invert_yaxis()
plt.savefig(path+'Uwnd_Zonal_Mean.png')
plt.show()
```
{% raw %}
<img src="/images/Visualization_Practice_anomaly/Uwnd_Zonal_Mean.png" alt="Uwind Zonal Mean at 2010-01-01">
{% endraw %}

---

## 3. Draw Global Mean Timeseries of SST
```
# Draw Global Mean Timeseries

var = xr.open_dataset('../Data/sst.mnmean.nc')
#print(var.sst.shape)

mean = var.sst.mean(dim=['lat', 'lon'])

# Plot
fig = plt.figure(figsize=(10,5), dpi=200)
ax = plt.axes()

plt.plot(var.time.values, mean, color='black')

ax.set_xlabel('Time')
ax.set_ylabel('$\degree$C')
ax.set_title('Global Mean SST')
plt.savefig(path+'Viz_tutorial_Anomaly_02.png')
plt.show()
```
{% raw %}
<img src="/images/Visualization_Practice_anomaly/Viz_tutorial_Anomaly_02.png" alt="Global Mean Sea Surface Temperature Timeseries 1983-2023">
{% endraw %}

---

## 4. Draw Weighted Global Mean Timeseries of SST
```
# Draw Weighted Global Mean Timeseries

var = xr.open_dataset('../Data/sst.mnmean.nc')

# Weights = cosine of latitude (proportional to grid cell area)
weights = np.cos(np.deg2rad(var.lat))
weights.name = "weights"

# Mean and Weighted Mean
# Mean
mean = var.sst.mean(dim=['lat','lon'])

# Weighted Mean
var_weighted = var.sst.weighted(weights)
weighted_mean = var_weighted.mean(dim=['lat', 'lon'])

# Plot
fig = plt.figure(figsize=(10,5), dpi=200)
ax = plt.axes()

plt.plot(var.time.values, mean, color='black')
plt.plot(var.time.values, weighted_mean, color='blue')

ax.set_xlabel('Time')
ax.set_ylabel('$\degree$C')
ax.set_title('Global Mean SST')
plt.savefig(path+'Weighted_Global_Mean_Timeseries')
plt.show()
```
{% raw %}
<img src="/images/Visualization_Practice_anomaly/Weighted_Global_Mean_Timeseries.png" alt="Weighted Global Mean Sea Surface Temperature Timeseries (cosine of latitude)">
{% endraw %}


### 4-a. About the weights in section 4

Weights defined in section 4 is cosine of latitude of each grid. That means, we have 0 at polars and 1 at equator. At first I assumed this was intended to apply the difference of "importance" of each regions. However, there is a mathematical proof for doing so.

Weights are defined as such manner because the actual area represented by the gird decreases as its latitude go far from 0. It is because the a degree of longitude spans up to 111 km at the equator, but 0 km at the poles. Thus, the grid cells at hight latitude cover much less area than those at low latitude.


### 4-b. Concepts

The east–west (longitude) width of the cell **shrinks with** $\cos(\phi)$ because lines of longitude converge toward the poles.


Let's compute the **approximate area** of a small rectangle on a sphere.

- **R**: Radius of Earth  
- **$\phi$**: Latitude in **radians**  
- **$\Delta\phi$**: Latitude step (in radians)  
- **$\Delta\lambda$**: Longitude step (in radians)


A grid cell on a sphere is approximately:

$$ \text{Area} = R^2 \cdot \Delta\phi \cdot \Delta\lambda \cdot \cos(\phi) $$



### 4-c. Visual Explanation

You can imagine Earth as sliced into **thin horizontal rings** (by latitude):

- Each ring (from latitude $\phi$ to $\phi + \Delta\phi$) forms a **band or strip**.
- The **length around the ring** (i.e., the circumference) is:  

$$ 2\pi R \cos(\phi) $$

So, a small patch at that latitude has area:  

$$ \text{(ring width)} \times \text{(ring height)} = (R \cos(\phi) \cdot \Delta\lambda) \cdot (R \cdot \Delta\phi) = R^2 \cdot \cos(\phi) \cdot \Delta\phi \cdot \Delta\lambda $$



The **area of a grid cell at latitude $\phi$** is:  


$$ \text{Area} \propto \cos(\phi) $$
This is why we use **$$\cos(\phi)$$ as weights** in global averages to correctly account for the **shrinking surface area** of grid cells near the poles.
---

## 5. Draw Climatology of SST
```
# Draw Climatology

var = xr.open_dataset('../Data/sst.mnmean.nc')

# Climatology
clim = var.sst.sel(time=slice('1990-01','2021-12')).groupby('time.month').mean(dim='time')

# Plot
fig = plt.figure(figsize=(18,14), dpi=100)
clevs = np.arange(-3,33,1)

ax1 = plt.subplot(221, projection=ccrs.Robinson(180))
a = clim.isel(month=0).plot.contourf(
    ax=ax1,
    levels=clevs,
    cmap='jet',
    transform=ccrs.PlateCarree(),
    cbar_kwargs={'extendrect':'True', 
                 'orientation':'horizontal', 
                 'pad':0.06, 
                 'aspect':30}    
)

ax1.set_title('SST Climatology(1990-2021)(JAN)')
ax1.coastlines()
ax1.add_feature(cfeature.NaturalEarthFeature('physical', 
                                              'land', 
                                              '50m', 
                                              edgecolor='face', 
                                              facecolor='black'))
ax1.gridlines(draw_labels=True, 
             linestyle=':', 
             color='gray', 
             x_inline=False)

ax2 = plt.subplot(222, projection=ccrs.Robinson(180))
b = clim.isel(month=3).plot.contourf(
    ax=ax2, 
    levels=clevs, 
    cmap='jet',
    transform=ccrs.PlateCarree(),
    cbar_kwargs={'extendrect':'True', 
                 'orientation':'horizontal', 
                 'pad':0.06, 
                 'aspect':30} 
)
ax2.set_title('SST Climatology(1990-2021)(APR)')
ax2.coastlines()
ax2.add_feature(cfeature.NaturalEarthFeature('physical',
                                             'land',
                                             '50m',
                                             edgecolor='face',
                                             facecolor='black'))
ax2.gridlines(draw_labels=True,
              linestyle=':',
              color='gray')

ax3 = plt.subplot(223, projection=ccrs.Robinson(180))
c = clim.isel(month=6).plot.contourf(
    ax=ax3, 
    levels=clevs, 
    cmap='jet',
    transform=ccrs.PlateCarree(),
    cbar_kwargs={'extendrect':'True', 
                 'orientation':'horizontal', 
                 'pad':0.06, 
                 'aspect':30} 
)
ax3.set_title('SST Climatology(1990-2021)(JUL)')
ax3.coastlines()
ax3.add_feature(cfeature.NaturalEarthFeature('physical',
                                             'land',
                                             '50m',
                                             edgecolor='face',
                                             facecolor='black'))
ax3.gridlines(draw_labels=True,
              linestyle=':',
              color='gray')


ax4 = plt.subplot(224, projection=ccrs.Robinson(180))
d = clim.isel(month=9).plot.contourf(
    ax=ax4, 
    levels=clevs, 
    cmap='jet',
    transform=ccrs.PlateCarree(),
    cbar_kwargs={'extendrect':'True', 
                 'orientation':'horizontal', 
                 'pad':0.06, 
                 'aspect':30} 
)
ax4.set_title('SST Climatology(1990-2021)(OCT)')
ax4.coastlines()
ax4.add_feature(cfeature.NaturalEarthFeature('physical',
                                             'land',
                                             '50m',
                                             edgecolor='face',
                                             facecolor='black'))
ax4.gridlines(draw_labels=True,
              linestyle=':',
              color='gray')

plt.savefig(path+'SST_Monthly_CLimatology.png')
plt.show()
```
{% raw %}
<img src="/images/Visualization_Practice_anomaly/SST_Monthly_Climatology.png" alt="Monthly Climatology of Sea Surface Temperature">
{% endraw %}

---

## 6. Draw Nino 3.4 Index 
```
# Draw Nino Index

var = xr.open_dataset('../Data/sst.mnmean.nc')
clim = var.sst.sel(time=slice('1990-01','2021-12')).groupby('time.month').mean(dim=['time'])
anom = var.sst.groupby('time.month') - clim

# Select Nino area
area_nino34 = anom.where((anom.lat<5) & (anom.lat>-5) & (anom.lon>190) & (anom.lon<240), drop=True)

# Calculate Mean over Nino area
nino34 = area_nino34.mean(dim=['lat','lon'])

# 3-month moving average
nino3mon = nino34.rolling(time=3, center=True).mean()

# Plot
fig = plt.figure(figsize=(10,5), dpi=200)
ax = fig.subplots()

ax.fill_between(
    nino3mon.time.values, 
    nino3mon.where(nino3mon>=0.0).values,
    0.0,
    color='red',
    alpha=0.8)
ax.fill_between(
    nino3mon.time.values,
    nino3mon.where(nino3mon<=0.0).values,
    0.0,
    color='blue',
    alpha=0.8
)
ax.plot(nino3mon.time.values,nino3mon,color='black')
ax.axhline(0, color='black', linewidth=0.5)
ax.axhline(1.0, color='black', linewidth=0.5, linestyle='dashed')
ax.axhline(-1.0, color='black', linewidth=0.5, linestyle='dashed')
ax.axhline(1.5, color='black', linewidth=0.6, linestyle='dashed')
ax.axhline(-1.5, color='black', linewidth=0.6, linestyle='dashed')
ax.axhline(2.0, color='black', linewidth=0.7, linestyle='dashed')
ax.axhline(-2.0, color='black', linewidth=0.7, linestyle='dashed')

ax.set_xlabel('Time')
ax.set_ylabel('$\degree$C')
ax.set_title('Nino 3.4 index')

plt.savefig(path+'Nino_34_Index.png')
plt.show()
```
{% raw %}
<img src="/images/Visualization_Practice_anomaly/Nino_34_Index.png" alt="Nino 3.4 Index">
{% endraw %}

---

## 7. Draw Seasonal Nino 3.4 Index using Moving Average

```
# Draw Seasonal Nino Index

var = xr.open_dataset('../Data/sst.mnmean.nc')

# Calculate Climatology and anomaly
clim = var.sst.sel(time=slice('1990-01', '2021-12')).groupby('time.month').mean(dim='time')
anom = var.sst.groupby('time.month') - clim

# Select Nino 3.4 area
area_nino34 = anom.where((anom.lat < 5) & (anom.lat > -5) & (anom.lon>190) & (anom.lon<240), drop=True)
nino34 = area_nino34.mean(dim=['lat', 'lon'])

# 3-month moving average
nino3mon = nino34.rolling(time=3, center=True).mean()

# Pick specific season
# starting from Jan, add 12months(1year) -> every Jan
# nino3mon is moving average with center true, we selected Dec, Jan, Feb
ninos = nino3mon[1::12] 
print(ninos.std().values)

# Plot
fig = plt.figure(figsize=(6,4), dpi=200)
ax = fig.subplots()

print(ninos)

ax.bar(list(range(1982, 2024)), ninos, color='g', width=0.7)
ax.axhline(0,color='black',linewidth=0.5)
ax.axhline(1.0,color='black',linewidth=0.5,linestyle='dashed')
ax.axhline(-1.0,color='black',linewidth=0.5,linestyle='dashed')
ax.axhline(1.5,color='black',linewidth=0.6,linestyle='dashed')
ax.axhline(-1.5,color='black',linewidth=0.6,linestyle='dashed')
ax.axhline(2.0,color='black',linewidth=0.7,linestyle='dashed')
ax.axhline(-2.0,color='black',linewidth=0.7,linestyle='dashed')
ax.set_xlabel('Time')
ax.set_ylabel('$\degree$C')
ax.set_title('Nino 3.4 index (DJF)')

plt.savefig(path+'Nino34_index_DJF_OISST.png')
plt.show()
```
{% raw %}
<img src="/images/Visualization_Practice_anomaly/Nino34_index_DJF_OISST.png" alt="Nino 3.4 Index(DJF) Moving Average">
{% endraw %}

---

## 8. Draw Overlay of Wind Vector on Prcp Climatology
```
# Climatological wind vector overlaid on climatological precipitation

prcp = xr.open_dataset('../Data/precip.mon.mean.nc').precip
uwnd = xr.open_dataset('../Data/uwnd.mon.mean.nc').uwnd
vwnd = xr.open_dataset('../Data/vwnd.mon.mean.nc').vwnd

prcp_clim = prcp.sel(time=slice('1990-01', '2021-12')).groupby('time.month').mean(dim='time')
uwnd_clim = uwnd.sel(time=slice('1990-01', '2021-12')).groupby('time.month').mean(dim='time')
vwnd_clim = vwnd.sel(time=slice('1990-01', '2021-12')).groupby('time.month').mean(dim='time')
uvwnd_clim = xr.merge([uwnd_clim[:,:,::3,::3], vwnd_clim[:,:,::3,::3]]).sel(level='850.0')

# Plot
fig = plt.figure(figsize=(18,14), dpi=200)

clevs=np.arange(-5,25,3)


# DJF
# prcp
ax1 = plt.subplot(221, projection=ccrs.Robinson(180))
a = prcp_clim.isel(month=[-1,0,1]).mean(dim='month').plot.contourf(
    ax=ax1,
    levels=clevs,
    cmap='jet',
    transform=ccrs.PlateCarree(),
    cbar_kwargs={'extendrect':'True',
                 'orientation':'horizontal', 
                 'pad':0.06, 
                 'aspect':30}
)
# uvwind
aa = uvwnd_clim.isel(month=[-1,0,1]).mean(dim='month').plot.quiver(
    x='lon',
    y='lat',
    u='uwnd',
    v='vwnd',
    color='green',
    transform=ccrs.PlateCarree()
)

ax1.set_title('Wind(850hpa) on Precipitation Climatology(1990-2021) DJF')
ax1.coastlines()
ax1.add_feature(cfeature.NaturalEarthFeature('physical',
                                             'land',
                                             '50m',
                                             edgecolor='face',
                                             facecolor='black'))
ax1.gridlines(draw_labels=True, linestyle=':', color='gray', x_inline=False)

# MAM
# prcp
ax2 = plt.subplot(222, projection=ccrs.Robinson(180))
b = prcp_clim.isel(month=[2,3,4]).mean(dim='month').plot.contourf(
    ax=ax2,
    levels=clevs,
    cmap='jet',
    transform=ccrs.PlateCarree(),
    cbar_kwargs={'extendrect':'True',
                 'orientation':'horizontal', 
                 'pad':0.06, 
                 'aspect':30}
)

# uvwnd
bb = uvwnd_clim.isel(month=[2,3,4]).mean(dim='month').plot.quiver(
    x='lon',
    y='lat',
    u='uwnd',
    v='vwnd',
    color='green',
    transform=ccrs.PlateCarree()
)

ax2.set_title('Wind(850hpa) on Precipitation Climatology(1990-2021) MAM')
ax2.coastlines()
ax2.add_feature(cfeature.NaturalEarthFeature('physical',
                                             'land',
                                             '50m',
                                             edgecolor='face',
                                             facecolor='black'))
ax2.gridlines(draw_labels=True, linestyle=':', color='gray', x_inline=False)

# JJA
# prcp
ax3 = plt.subplot(223, projection=ccrs.Robinson(180))
c = prcp_clim.isel(month=[5,6,7]).mean(dim='month').plot.contourf(
    ax=ax3,
    levels=clevs,
    cmap='jet',
    transform=ccrs.PlateCarree(),
    cbar_kwargs={'extendrect':'True',
                 'orientation':'horizontal', 
                 'pad':0.06, 
                 'aspect':30}
)

# uvwnd
cc = uvwnd_clim.isel(month=[5,6,7]).mean(dim='month').plot.quiver(
    x='lon',
    y='lat',
    u='uwnd',
    v='vwnd',
    color='green',
    transform=ccrs.PlateCarree()
)

ax3.set_title('Wind(850hpa) on Precipitation Climatology(1990-2021) JJA')
ax3.coastlines()
ax3.add_feature(cfeature.NaturalEarthFeature('physical',
                                             'land',
                                             '50m',
                                             edgecolor='face',
                                             facecolor='black'))
ax3.gridlines(draw_labels=True, linestyle=':', color='gray', x_inline=False)

# SON
# prcp
ax4 = plt.subplot(224, projection=ccrs.Robinson(180))
d = prcp_clim.isel(month=[8,9,10]).mean(dim='month').plot.contourf(
    ax=ax4,
    levels=clevs,
    cmap='jet',
    transform=ccrs.PlateCarree(),
    cbar_kwargs={'extendrect':'True',
                 'orientation':'horizontal', 
                 'pad':0.06, 
                 'aspect':30}
)

# uvwnd
dd = uvwnd_clim.isel(month=[8,9,10]).mean(dim='month').plot.quiver(
    x='lon',
    y='lat',
    u='uwnd',
    v='vwnd',
    color='green',
    transform=ccrs.PlateCarree()
)

ax4.set_title('Wind(850hpa) on Precipitation Climatology(1990-2021) SON')
ax4.coastlines()
ax4.add_feature(cfeature.NaturalEarthFeature('physical',
                                             'land',
                                             '50m',
                                             edgecolor='face',
                                             facecolor='black'))
ax4.gridlines(draw_labels=True, linestyle=':', color='gray', x_inline=False)

plt.savefig(path+'Wind_overlaid_on_prcp_climatology.png')
plt.show()
```
{% raw %}
<img src="/images/Visualization_Practice_anomaly/Wind_overlaid_on_prcp_climatology.png" alt="Wind (850hpa) Overlaid on Precipitation Climatology(1990-2021) 3-month Moving Average">
{% endraw %}

# Overview
 One of the most important concepts in climate change research would be climatology, a long-term weather patterns of a specific region, is dealt with python. Using rolling method in xarray, moving average in terms of month is also visualized. The workflow of visualization and key concepts in research will be keep practiced. 

 Before proving the weights defined to handle global mean sst in section 4, I attempted just simple and shallow approach, averaging whole gird without weights. However, as soon as the idea that the gird is just a way how you represent your data and rectangle gird does not mimic real Earth, I googled it to set correct weights. Thanks to this issue, "think before you do" came up with my mind. 
