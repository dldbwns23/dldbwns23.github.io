---
layout: post
title: "Composite Analysis Practice"
date: 2025-07-21 20:48:00 +0900
tags: [Visualization, Composite, Python, Cartopy, Xarray]
---

To understand composite analysis, I practiced the workflow of composite analysis and visualization.

In meteorology, composite analysis is a powerful statistical analysis technique used to identify the typical characteristics or structure of a specific meteorological phenomenon or climate variability mode.
Simply put, it's about gathering similar phenomena that have occurred multiple times and averaging them to create a "typical" pattern.
The reason why we use composite analysis is that when analyzing multiple events, each of them has its own small variations.
For example, El Nino events might vary in intensity or pattern every time they occur. 
Composite anlaysis helps to cancel out the small variations so that analysis can reveal the crux of characteristics or physical mechanisms to the phenomenon.

- Identifying Typical Patterns
- Studying Casue and Effect
- Improving Predicability
---

## 1. Setup
```
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy import stats
```

---

## 2. Compositing SST Based on ENSO Index Method 1
```
# Compositing SST Based on ENSO Index Method 1

data = xr.open_dataset('../Data/sst.mnmean.nc')
sst = data.sst

# Climatology 1982-2018
clim = sst.sel(time=slice('1982-01','2018-12')).groupby('time.month').mean(dim='time')

# Anomaly
anom = sst.groupby('time.month') - clim

# Select El Nino years
onset_elnino = [1982, 1986, 1991, 1997, 2009, 2015] # onset D(0) strong elnino years
end_elnino = [1983, 1987, 1992, 1998, 2010, 2016] # end JF(+1) strong el nino years

# Composite El Nino years
# All the December
dec = anom.sel(time=anom['time.month'].isin([12]))
dec0 = dec.sel(time=dec['time.year'].isin(onset_elnino))

jf = anom.sel(time=anom['time.month'].isin([1,2]))
jf1 = jf.sel(time=jf['time.year'].isin(end_elnino))

# Merge all months and take seasonal mean
djf = xr.merge([dec0, jf1])
# xarrays automoatically assign 'DJF', 'MAM', 'JJA', 'SON' labels
# based on month of each time
djf_mean = djf.groupby('time.season').mean('time')

compo = djf_mean.to_array()

# Plot
fig = plt.figure(figsize=(9,7))
ax = plt.axes(projection=ccrs.PlateCarree(180))

clevs = np.around(np.arange(-3, 3.1, 0.1),1)
ticks = np.around(np.arange(-3, 3.1, 0.5),1)

compo.isel(variable=0, season=0).plot.contourf(
    ax=ax,
    levels=clevs,
    cmap='bwr',
    transform=ccrs.PlateCarree(),
    cbar_kwargs={'orientation':'horizontal',
                 'pad':0.06,
                 'aspect':30,
                 'extend':'neither',
                 'ticks':ticks}
)
ax.set_title('El Nino Composite | SST | DJF | Method1')
ax.coastlines()
ax.gridlines(draw_labels=True, linestyle=':', color='gray')
ax.add_feature(cfeature.NaturalEarthFeature('physical', 
                                            'land', 
                                            '50m', 
                                            edgecolor='face', 
                                            facecolor='black'))
plt.savefig('./Composite_Analysis/Nino_Composite_SST_DJF.png')
plt.show()
```
{% raw %}
<img src="/images/Composite_Analysis/Nino_Composite_SST_DJF.png" alt="El Nino Composite of SST During Winter">
{% endraw %}

---

## 3. Compositing SST Based on ENSO Index Method 2
```
# Compositing SST Based on ENSO Index Method 2

data = xr.open_dataset('../Data/sst.mnmean.nc')
sst = data.sst

# Climatology
clim = sst.sel(time=slice('1982-01','2018-12')).groupby('time.month').mean(dim='time')

# Anomaly
anomt = sst.groupby('time.month') - clim

# 3-month Moving Average
anom = anomt.rolling(time=3, center=True).mean('time')

# Calculate DJF Nino Index
nino34 = anom.where((anom.lat<5) & (anom.lat>-5) & (anom.lon>190) & (anom.lon<240), drop=True).mean(dim=['lat','lon'])
ninos = nino34[1::12] # Select Every DJF Nino 3.4 index

# Composite El Nino years
std = ninos.std(axis=0).values # <=> std = ninos.std(dim='time').values
compo = anom.where(ninos > std).mean(dim='time')

# Print El Nino years
# elyr = ninos.where(ninos > std, drop=True).time.values
# import pandas as pd
# dates = pd.to_datetime(elyr)
# print(dates)

# Plot
fig = plt.figure(figsize=(9,7))
ax = plt.axes(projection=ccrs.PlateCarree(180))

clevs = np.around(np.arange(-3,3.1,0.1),1)
ticks = np.around(np.arange(-3,3.1,0.5),1)

compo.plot.contourf(
    ax=ax,
    levels=clevs,
    cmap='bwr',
    transform=ccrs.PlateCarree(),
    cbar_kwargs={'orientation':'horizontal',
                 'pad':0.06,
                 'aspect':30,
                 'extend':'neither',
                 'ticks':ticks}
)
ax.set_title('El Nino Composite | SST | DJF | Method2')
ax.coastlines()
ax.gridlines(draw_labels=True, linestyle=':', color='gray')
ax.add_feature(cfeature.NaturalEarthFeature('physical', 
                                            'land', 
                                            '50m', 
                                             edgecolor='face', 
                                            facecolor='black'))
plt.savefig('./Composite_Analysis/Nino_Composite_SST_DJF_2.png')
plt.show()
```
{% raw %}
<img src="/images/Composite_Analysis/Nino_Composite_SST_DJF_2.png" alt="Other method for El Nino Composite of SST During Winter">
{% endraw %}

---

## 4.  Compositing SST Based on Enso Index with Significance Test
```
# Compositing SST Based on Enso Index

data = xr.open_dataset('../Data/sst.mnmean.nc')
sst = data.sst

# Climatology
clim = sst.sel(time=slice('1982-01', '2018-12')).groupby('time.month').mean(dim='time')

# Anomaly
anomt = sst.groupby('time.month') - clim
anom = anomt.rolling(time=3, center=True).mean(dim=['lat','lon'])

# DJF Nino Index
nino34 = anom.where((anom.lat<5) & (anom.lat>-5) & (anom.lon>190) & (anom.lon<240),drop=True).mean(dim=['lat','lon'])
ninos = nino34[1::12]

# El Nino years
std = ninos.std(dim='time').values
compo = anom.where(ninos>std).mean(dim='time')

# Significance Test for Composite
s1 = anom.where(ninos>std)
s2 = anom.where(ninos<=std)
t, p = stats.ttest_ind(s1, s2, axis=0, equal_var=False, nan_policy='omit', alternative='two-sided')

# Plot
fig = plt.figure(figsize=(9,7))
ax = plt.axes(projection=ccrs.PlateCarree(180))

clevs = np.around(np.arange(-3,3.1,0.1),1)
ticks = np.around(np.arange(-3,3.1,0.5),1)
compo_sig = compo.where(p<0.05) # Select the grid only statistically significantly different than non-El-Nino years
compo_sig.plot.contourf(
    ax=ax,
    levels=clevs,
    cmap='bwr',
    transform=ccrs.PlateCarree(),
    cbar_kwargs={'orientation':'horizontal',
                 'extend':'neither',
                 'pad':0.06,
                 'aspect':30,
                 'ticks':ticks}
)
ax.set_title('El Nino Composite | SST | DJF')
ax.coastlines()
ax.gridlines(draw_labels=True, linestyle=':', color='gray')
ax.add_feature(cfeature.NaturalEarthFeature('physical','land','50m',edgecolor='face',facecolor='black'))
plt.savefig('./Composite_Analysis/Nino_Composite_SST_DJF_pval.png')
plt.show()
```
{% raw %}
<img src="/images/Composite_Analysis/Nino_Composite_SST_DJF_pval.png" alt="El Nino Composite of SST During Winter showing only statistically significant at p-value less than 0.05">
{% endraw %}

---

## 5. Compositing SST Based Winter Precipitation in Pohang
```
# Compositing SST Based Winter Precipitation in Pohang
# Pohang monthly prcp 1980-01 - 2025-07
data = pd.read_csv('../Data/rn_20250720201413.csv',encoding='euc-kr', sep=',', header=6)
data['년월'] = pd.to_datetime(data['년월'])
dec = data.loc[data['년월'].dt.month == 12]
prcp = dec['강수량(mm)']
dates = pd.DatetimeIndex(dec['년월'])

df = pd.DataFrame({'time':dates,'강수량(mm)':prcp}).set_index(['time'])

xarr = df.to_xarray().to_array()

# Select time period and anomaly for Pohang precipitation
prcp_sliced = xarr.sel(time=slice('1981-12-01','2018-12-01'))

# Calculate anomaly relative to its own mean
anom_PH_values = prcp_sliced - prcp_sliced.mean(dim='time')

# standard deviation of the anomaly
std_PH = anom_PH_values.std(dim='time')

if 'variable' in anom_PH_values.dims and anom_PH_values['variable'].size == 1:
    anom_PH_values = anom_PH_values.squeeze('variable')
if 'variable' in std_PH.dims and std_PH['variable'].size == 1:
    std_PH = std_PH.squeeze('variable')

# Map data for composite
global_data = xr.open_dataset('../Data/sst.mnmean.nc')
sst = global_data.sst

for dim in sst.dims:
    if sst[dim].size == 1 and dim not in ['lat', 'lon', 'time']: # Avoid squeezing essential dims
        sst = sst.squeeze(dim)

# Climatology of SST for the reference period
clim = sst.sel(time=slice('1981-12', '2018-12')).groupby('time.month').mean(dim='time')

# Anomaly
anom_all_months = sst.groupby('time.month') - clim

# Select December 
anom_dec = anom_all_months.sel(time=anom_all_months['time'].dt.month == 12)

for dim in anom_dec.dims:
    if anom_dec[dim].size == 1 and dim not in ['lat', 'lon', 'time']:
        anom_dec = anom_dec.squeeze(dim)


# Composite:
compo = anom_dec.where(anom_PH_values > std_PH).mean(dim='time')

# Explicitly squeeze `compo` if it still has a singleton dimension
for dim in compo.dims:
    if compo[dim].size == 1:
        compo = compo.squeeze(dim)

# Significance test
sample1 = anom_dec.where(anom_PH_values > std_PH)
sample2 = anom_dec.where(anom_PH_values <= std_PH)

# ttest_ind computes along axis=0 (time)
t, p = stats.ttest_ind(sample1, sample2, axis=0, equal_var=False, nan_policy='omit', alternative='two-sided')

# Convert p (NumPy array) to an Xarray DataArray with proper coordinates for plotting
# If `compo` has been squeezed to (lat, lon), then `p_xr` also needs to match that.
p_xr = xr.DataArray(p, coords={'lat': compo['lat'], 'lon': compo['lon']}, dims=['lat', 'lon'])


# Plot
fig = plt.figure(figsize=(16,7))
ax1 = plt.subplot(121,projection=ccrs.PlateCarree(180))

clevs=np.around(np.arange(-1.5, 1.6, 0.1), 1)
ticks=np.around(np.arange(-1.5, 1.6, 0.5), 1)

compo.plot.contourf(
    ax=ax1,
    levels = clevs,
    cmap='bwr',
    transform=ccrs.PlateCarree(),
    cbar_kwargs={'orientation': 'horizontal',
                 'extend':'neither',
                 'pad': 0.06,
                 'aspect': 30,
                 'ticks': ticks}
)

ax1.set_title('Pohang High prcp composite | SST | Dec')
ax1.coastlines()
ax1.gridlines(draw_labels=True, linestyle=':', color='gray')
ax1.add_feature(cfeature.NaturalEarthFeature('physical',
                                             'land',
                                             '50m',
                                             edgecolor='face',
                                             facecolor='black'))

ax2 = plt.subplot(122,projection=ccrs.PlateCarree(180))

compo.where(p_xr < 0.05).plot.contourf(
    ax=ax2,
    levels = clevs,
    cmap='bwr',
    transform=ccrs.PlateCarree(),
    cbar_kwargs={'orientation': 'horizontal',
                 'extend':'neither',
                 'pad': 0.06,
                 'aspect': 30,
                 'ticks': ticks}
)

ax2.set_title('Pohang High prcp composite pval<0.05 | SST | Dec')
ax2.coastlines()
ax2.gridlines(draw_labels=True, linestyle=':', color='gray')
ax2.add_feature(cfeature.NaturalEarthFeature('physical',
                                             'land',
                                             '50m',
                                             edgecolor='face',
                                             facecolor='black'))
plt.savefig('./Composite_Analysis/Pohang_Composite_SST_Dec.png')
plt.show()
```
{% raw %}
<img src="/images/Composite_Analysis/Pohang_Composite_SST_Dec.png" alt="Composite Analysis of SSt during the period when Pohang, South Korea experienced high precipitation than reference period">
{% endraw %}

---

## 6. Compositing Winter Precipitation Over the East Asia Using Winter ENSO Index
```
# Compositing Winter Precipitation Over the East Asia Using Winter ENSO Index
data = xr.open_dataset('../Data/sst.mnmean.nc')
sst = data.sst

# Climatology
clim = sst.sel(time=slice('1982-01', '2018-12')).groupby('time.month').mean(dim='time')

# Anomaly
anomt = sst.groupby('time.month') - clim
anom = anomt.rolling(time=3, center=True).mean(dim='time')

# Select Winter
nino34 = anom.where((anom.lat<5) & (anom.lat>-5) & (anom.lon>190) & (anom.lon<240),drop=True).mean(dim=['lat','lon'])
ninos = nino34[1::12]

# East Asia Precipitation
prcp = xr.open_dataset('../Data/precip.mon.mean.nc').precip
east_asia = prcp.where((prcp.lat>20) & (prcp.lat<60) & (prcp.lon>90) & (prcp.lon<155),drop=True)
east_asia_clim = east_asia.sel(time=slice('1982-01', '2018-12')).groupby('time.month').mean(dim='time')
east_asia_anomt = (east_asia.groupby('time.month') - east_asia_clim)
east_asia_anom = east_asia_anomt.rolling(time=3, center=True).mean(dim='time')

east_asia_anom_winter = east_asia_anom.mean(dim=['lat','lon'])
east_asia_anom_winter = east_asia_anom_winter[1::12] # Starts from Dec -> select 1::12
ea_std = east_asia_anom_winter.std()

# Composite
compo = anom.where(east_asia_anom_winter>ea_std).mean(dim='time')

s1 = anom.where(east_asia_anom_winter>ea_std)
s2 = anom.where(east_asia_anom_winter<=ea_std)
t, p = stats.ttest_ind(s1, s2, axis=0, equal_var=False, nan_policy='omit', alternative='two-sided')

p_xr = xr.DataArray(p, coords={'lat': compo['lat'], 'lon': compo['lon']}, dims=['lat', 'lon'])


# Plot
fig = plt.figure(figsize=(16,7))
ax1 = plt.subplot(121,projection=ccrs.PlateCarree(180))

clevs=np.around(np.arange(-2, 2.1, 0.1), 1)
ticks=np.around(np.arange(-2, 2.1, 0.5), 1)

compo.plot.contourf(
    ax=ax1,
    levels = clevs,
    cmap='bwr',
    transform=ccrs.PlateCarree(),
    cbar_kwargs={'orientation': 'horizontal',
                 'extend':'neither',
                 'pad': 0.06,
                 'aspect': 30,
                 'ticks': ticks}
)

ax1.set_title('East Asia High prcp composite | SST | DJF')
ax1.coastlines()
ax1.gridlines(draw_labels=True, linestyle=':', color='gray')
ax1.add_feature(cfeature.NaturalEarthFeature('physical',
                                             'land',
                                             '50m',
                                             edgecolor='face',
                                             facecolor='black'))

ax2 = plt.subplot(122,projection=ccrs.PlateCarree(180))

compo.where(p_xr < 0.05).plot.contourf(
    ax=ax2,
    levels = clevs,
    cmap='bwr',
    transform=ccrs.PlateCarree(),
    cbar_kwargs={'orientation': 'horizontal',
                 'extend':'neither',
                 'pad': 0.06,
                 'aspect': 30,
                 'ticks': ticks}
)

ax2.set_title('East Asia High prcp composite pval<0.05 | SST | DJF')
ax2.coastlines()
ax2.gridlines(draw_labels=True, linestyle=':', color='gray')
ax2.add_feature(cfeature.NaturalEarthFeature('physical',
                                             'land',
                                             '50m',
                                             edgecolor='face',
                                             facecolor='black'))
plt.savefig('./Composite_Analysis/East_Asia_prcp_Composite_SST_DJF.png')
plt.show()
```
{% raw %}
<img src="/images/Composite_Analysis/East_Asia_prcp_Composite_SST_DJF.png" alt="Compositing Winter Precipitation Over the East Asia Using Winter ENSO Index With p-value less than 0.05">
{% endraw %}

---


# Overview
