---
layout: post
title: Visualization Practice
date: 2025-07-09 14:19:00 +0900
tags: [Visualization, Python, Cartopy, Xarray]
---


## 1. Setup
```
import matplotlib.pyplot as plt
import xarray as xr
import cartopy as ct
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import cartopy.feature as cfeature
from matplotlib.colors import BoundaryNorm
```

## 2. Basic SST Visualization
```
var = xr.open_dataset('../Data/sst.mnmean.nc')
sst = var.sst

fig = plt.figure(figsize=(9,7))
ax = plt.axes(projection = ccrs.PlateCarree())

sst.isel(time=0).plot.contourf(ax=ax, levels=20, cmap='jet') # Shading plot

# Option 1
ax.coastlines()
plt.show()

# Option 2
ax.add_feature(ct.feature.GSHHSFeature(edgecolor='k'))
plt.show()
```

## 3. SST Visualization using levels
```
fig = plt.figure(figsize=(9,7))
ax = plt.axes(projection=ccrs.PlateCarree())

sst.isel(time=0).plot.contourf(
    ax=ax,
    levels=[-3,0,3,6,9,12,15,18,21,24,27,30],
    transform=ccrs.PlateCarree()
)

ax.set_title('Sea Surface Temperature (SST)')
ax.coastlines(resolution='10m')
plt.show()
```

## 4. SST Visualization Using Different Projections
```
fig = plt.figure(figsize=(9,7))
#ax = plt.axes(projection=ccrs.PlateCarree())
#ax = plt.axes(projection=ccrs.Mercator())
#ax = plt.axes(projection=ccrs.Mollweide())
ax = plt.axes(projection=ccrs.Robinson())
#ax = plt.axes(projection=ccrs.LambertConformal(130, 40))
#ax = plt.axes(projection=ccrs.Orthographic())

#ax.set_extent([90, -90, 0, 360], crs=ccrs.PlateCarree())

sst.isel(time=0).plot.contourf(
    ax=ax,
    levels=[-3,0,3,6,9,12,15,18,21,24,27,30],
    transform=ccrs.PlateCarree()
)

ax.set_title('Sea Surface Temperature (SST)')
ax.coastlines(resolution='10m')
ax.gridlines()
plt.show()
```

## 5. Adding Physical Features
```
var = xr.open_dataset('../Data/sst.mnmean.nc')
sst = var.sst

fig = plt.figure(figsize=(9,7))
ax = plt.axes(projection=ccrs.Robinson(central_longitude=180))

clevs = np.arange(-3, 33, 1)

sst.isel(time=0).plot.contourf(
    ax=ax,
    levels=clevs,
    cmap='jet',
    transform=ccrs.PlateCarree(),
    cbar_kwargs={'extendrect':'True', 'orientation':'horizontal'}
)

ax.set_title('Sea Surface Temperature (SST)')
ax.coastlines()
ax.gridlines(draw_labels=True,
             linewidth=1,
             linestyle=':',
             color='gray')
ax.add_feature(cfeature.NaturalEarthFeature('physical',
                                             'land',
                                             '50m',
                                             edgecolor='face',
                                             facecolor='black'))
plt.show()
```

## 6. Subplot Using Different Projections
```
from matplotlib.colors import BoundaryNorm

var = xr.open_dataset('../Data/sst.mnmean.nc')
sst = var.sst

fig = plt.figure(figsize=(12,12), dpi=100)
clevs = np.arange(-3,33,1)

# Fig 1
ax1 = plt.subplot(221, projection=ccrs.Mollweide(180))
a = sst.isel(time=0).plot.contourf(
    ax=ax1,
    levels=clevs,
    transform=ccrs.PlateCarree(),
    cbar_kwargs={'extendrect':'True',
                 'orientation':'horizontal',
                 'pad':0.08,
                 'aspect':30}
)
ax1.set_title('A. Sea Surface Temperature (SST)')
ax1.coastlines()
ax1.gridlines(draw_labels=True)
ax1.add_feature(cfeature.NaturalEarthFeature(
    'physical',
    'land',
    '50m',
    edgecolor='face',
    facecolor='black'
))

# Fig 2
ax2 = plt.subplot(223, projection=ccrs.NearsidePerspective(130, 40, satellite_height=1500000))
b = sst.isel(time=0).plot.pcolormesh(
    ax=ax2,
    levels=clevs,
    transform=ccrs.PlateCarree(),
    cbar_kwargs={'extendrect':'True',
                 'orientation':'horizontal',
                 'pad':0.08,
                 'fraction':0.045}
)

ax2.set_title('B. Sea Surface Temperature (SST)')
ax2.coastlines()
ax2.gridlines(draw_labels=True)
ax2.add_feature(cfeature.NaturalEarthFeature(
    'physical',
    'land',
    '50m',
    edgecolor='face',
    facecolor='black'
))

# Fig 3
ax3 = plt.subplot(122,projection=ccrs.Orthographic(130,40))
c = sst.isel(time=0).plot.contourf(
    ax=ax3,
    levels=clevs,
    transform=ccrs.PlateCarree(),
    cbar_kwargs={'extendrect':'True',
                 'orientation':'vertical',
                 'shrink': 0.45}
)
ax3.set_title('C. Sea Surface Temperature (SST)')
ax3.coastlines()
ax3.gridlines(draw_labels=True)
ax3.add_feature(cfeature.NaturalEarthFeature('physical',
                                             'land',
                                             '50m',
                                             edgecolor='face',
                                             facecolor='black'))
plt.show()
```

## 7. SST, Precipitation, Wind Visualization
```
sst = xr.open_dataset('../Data/sst.mnmean.nc').sst
prcp = xr.open_dataset('../Data/precip.mon.mean.nc').precip
uwnd = xr.open_dataset('../Data/uwnd.mon.mean.nc').uwnd
vwnd = xr.open_dataset('../Data/vwnd.mon.mean.nc').vwnd
uvwnd = xr.merge([uwnd[:,:,::3,::3], vwnd[:,:,::3,::3]])

fig = plt.figure(figsize=(9,7))
ax = plt.axes(projection=ccrs.Robinson(central_longitude=180))
clevs = np.arange(-3,33,1)

sst.isel(time=7).plot.contourf(
    ax=ax,
    levels=clevs,
    cmap='jet',
    transform=ccrs.PlateCarree(),
    cbar_kwargs={'extendrect':'True',
                 'orientation':'horizontal',
                 'pad':0.06}
)

uvwnd.isel(time=7,level=2).plot.quiver(
    x='lon',
    y='lat',
    u='uwnd',
    v='vwnd',
    color='green',
    transform=ccrs.PlateCarree()
)


prcp.isel(time=7).plot.contour(
    levels=12,
    transform=ccrs.PlateCarree()
)

ax.add_feature(cfeature.NaturalEarthFeature(
    'physical',
    'land',
    '50m',
    edgecolor='face',
    facecolor='gray'
))

ax.coastlines()
ax.gridlines(draw_labels=True, linestyle=':', color='gray')
ax.set_title('SST, Precipitation, Wind_850hpa')
plt.show()
```
