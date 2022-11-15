# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 14:40:15 2022

@author: lorinc
"""

#Import Packages
import matplotlib.pyplot as plt
import geopandas
from shapely.geometry import Polygon

##SHAPREFILE CLIP EXAMPLE

#Get or Create Example Data
capitals = geopandas.read_file(geopandas.datasets.get_path("naturalearth_cities"))
world = geopandas.read_file(geopandas.datasets.get_path("naturalearth_lowres"))

# Create a subset of the world data that is just the South American continent
south_america = world[world["continent"] == "South America"]

# Create a custom polygon
polygon = Polygon([(0, 0), (0, 90), (180, 90), (180, 0), (0, 0)])
poly_gdf = geopandas.GeoDataFrame([1], geometry=[polygon], crs=world.crs)


#Plot the Unclipped Data
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
world.plot(ax=ax1)
poly_gdf.boundary.plot(ax=ax1, color="red")
south_america.boundary.plot(ax=ax2, color="green")
capitals.plot(ax=ax2, color="purple")
ax1.set_title("All Unclipped World Data", fontsize=20)
ax2.set_title("All Unclipped Capital Data", fontsize=20)
ax1.set_axis_off()
ax2.set_axis_off()
plt.show()

#Clip the World Data
world_clipped = geopandas.clip(world, polygon)

# Plot the clipped data
# The plot below shows the results of the clip function applied to the world
# sphinx_gallery_thumbnail_number = 2
fig, ax = plt.subplots(figsize=(12, 8))
world_clipped.plot(ax=ax, color="purple")
world.boundary.plot(ax=ax)
poly_gdf.boundary.plot(ax=ax, color="red")
ax.set_title("World Clipped", fontsize=20)
ax.set_axis_off()
plt.show()

#Clip the Capitals Data
capitals_clipped = capitals.clip(south_america)

# Plot the clipped data
# The plot below shows the results of the clip function applied to the capital cities
fig, ax = plt.subplots(figsize=(12, 8))
capitals_clipped.plot(ax=ax, color="purple")
south_america.boundary.plot(ax=ax, color="green")
ax.set_title("Capitals Clipped", fontsize=20)
ax.set_axis_off()
plt.show()

##RASTER CLIP EXAMPLE

# import rioxarray and shapley
import rioxarray as riox
from shapely.geometry import Polygon
 
# Read raster using rioxarray
raster = riox.open_rasterio('cal_census.tiff')
 
# Shapely Polygon  to clip raster
geom = Polygon([[-13315253,3920415], [-13315821.7,4169010.0], [-13019053.84,4168177.65], [-13020302.1595,3921355.7391]])
 
# Use shapely polygon in clip method of rioxarray object to clip raster
clipped_raster = raster.rio.clip([geom])
 
# Save clipped raster
clipped_raster.rio.to_raster('clipped.tiff')
