# -*- coding: utf-8 -*-
"""
Model to assess the combined bio-physical and behavioural developments
along a river stretch in the coastal zone of Bangladesh.
This model has been implemented in Python.
This demontrsation has been developed as part of the 2022 research project: 
'Integrated Assessment Modelling of Tidal River Management in Bangladesh'.
"""
#set current working directory
import os
os.chdir(r'C:\Users\lorinc\OneDrive - Stichting Deltares\Documents\GitHub\trm-bangladesh\model')
import numpy as np
import math as math
import pcraster as pcr
from matplotlib import pyplot as plt 
import xarray as xr
import rioxarray
import rasterio
import pandas as pd
from household_agents import agent_functions 

#UserSettableParameters
model_params = {
    "slr2100": 1, #UserSettableParameter
    "subsidence": 0.005, #UserSettableParameter
    "sedrate": 0.1, #UserSettableParameter
    "trmsedrate": 0.4, #UserSettableParameter
}

#Strategies (1 - Business as Usual, 2 - nabaripoma)
strategy=2

#Options
plot = True
raster = True

#%%INITIALIZE

print('******** Initialization starts ********')

#Model parameters
slr2100 = model_params['slr2100']
subsidence = model_params['subsidence']
sedrate = model_params['sedrate']
trmsedrate = model_params['trmsedrate']
cellsize = 100 #100m
mslstart = 0.00
startyear = 2022
endyear = 2100
kslr = 0.02
mindraingrad = 0.1 / 1000. # 10cm per km minimum drainage gradient
year = startyear
msl = 0.00

#Read grid maps   

#Read rater metadata
with rasterio.open(r'p:\11208012-011-nabaripoma\Data\elevation.tif') as src:
    ras_meta = src.profile

# read the location of rivers and sea
rs_raster = rioxarray.open_rasterio(r'p:\11208012-011-nabaripoma\Data\rivers.tif')
rsmat = rs_raster.to_numpy().squeeze()
#rsmat[-5:,:]=2
#plot
plt.matshow(rsmat)
plt.title('location of rivers and sea')
plt.colorbar()
plt.show()

# Read the elevation (topography/bathymetry)
elev_raster = rioxarray.open_rasterio(r'p:\11208012-011-nabaripoma\Data\elevation_fabdem.tif')
elevmat = elev_raster.to_numpy().squeeze()
elevmat[rsmat == 1] = -5.0 - 1e-6 * np.indices(np.shape(elevmat))[0][rsmat == 1] #River bathymetry is -5 m

#plot
plt.matshow(elevmat)
plt.title('elevation')
plt.colorbar()
plt.show()

# read the location of polders
pol_raster = rioxarray.open_rasterio(r'p:\11208012-011-nabaripoma\Data\polders.tif')
polmat = pol_raster.to_numpy().squeeze()
#plot
plt.matshow(polmat)
plt.title('location of polders')
plt.colorbar()
plt.show()

#Numebr of polders
no_polder=np.max(polmat)

# read households per ha (per gridcell)
hh_raster = rioxarray.open_rasterio(r'p:\11208012-011-nabaripoma\Data\hh_perha.tif')
hhmat = hh_raster.to_numpy().squeeze()
#plot
plt.matshow(hhmat)
plt.title('households per ha')
plt.colorbar()
plt.show()

#initialize households
tot_pop_agr = {
"small": 47.0/100.,
"med": 8./100.,
"large": 1./100.,
"landless": 44./100.
}

land_ownership = {
"small": 
    {
    "landowner": 57./100.,
    "tenant": 43./100.,
    },
"med":
    {
    "landowner": 55./100.,
    "tenant": 45./100.,
    },
"large":
    {
    "landowner": 63./100.,
    "tenant": 37./100.,
    }
}

croppping_pattern = {
"small": 
    {
    "rice": 0.70,
    "fish-rice": 0.20,
    "fish": 0.10,
    "shrimp": 0.0
    },
"med":
    {
    "rice": 0.50,
    "fish-rice": 0.20,
    "fish": 0.25,
    "shrimp": 0.05
    },
"large":
    {
    "rice": 0.30,
    "fish-rice": 0.20,
    "fish": 0.20,
    "shrimp": 0.30
    }
}  

irrigation_perc = {
"small": np.random.normal(loc=0.113, scale=(0.113-0.0)/3),
"med": np.random.normal(loc=0.746, scale=(1-0.746)/3),
"large": np.random.normal(loc=0.703, scale=(1-0.703)/3)
}
    

#init arrays
landless_agents_perm = np.zeros(np.shape(elevmat))
landless_agents_seas = np.zeros(np.shape(elevmat))
landowner_agents_xy = np.zeros((np.shape(elevmat)[0], np.shape(elevmat)[1], 30))
#Calculate number of agents  
i=0  
for x in np.arange(0, np.shape(elevmat)[0]):
    for y in np.arange(0, np.shape(elevmat)[1]):
        #landlesss agents
        landless_agents_perm[x,y] = np.around(hhmat[x,y] * tot_pop_agr['landless']*0.5)
        landless_agents_seas[x,y] = np.around(hhmat[x,y] * tot_pop_agr['landless']*0.5)
        #landowner agents
        
        #i=0  rice_irrig_small                 #i=1  rice_irrig_med               #i=2  rice_irrig_large
        #i=3  rice_no_irrig_small              #i=4  rice_no_irrig_med            #i=5  rice_no_irrig_large
        #i=6  rice_irrig_landlease_small       #i=7  rice_irrig_landlease_med     #i=8  rice_irrig_landlease_large
        #i=9  rice_no_irrig_landlease_small    #i=10 rice_no_irrig_landlease_med  #i=11 rice_no_irrig_landlease_large
        #i=12 fish_landlease_small             #i=13 fish_landlease_med           #i=14 fish_landlease_large
        #i=15 fish_no_landlease_small          #i=16 fish_no_landlease_med        #i=17 fish_no_landlease_large
        #i=18 shrimp_landlease_small           #i=18 shrimp_landlease_med         #i=20 shrimp_landlease_large
        #i=21 shrimp_no_landlease_small        #i=22 shrimp_no_landlease_med      #i=23 shrimp_no_landlease_large
        #i=24 fish-rice_landlease_small        #i=25 fish-rice_landlease_med      #i=26 fish-rice_landlease_large
        #i=27 fish-rice_no_landlease_small     #i=28 fish-rice_no_landlease_med   #i=29 fish-rice_no_landlease_large
      
        
        for hh in ['small', 'med', 'large']:
            for crop in ["rice_irrig", "rice_no_irrig", "rice_irrig_landlease", "rice_no_irrig_landlease", "fish_landlease", "fish_no_landlease", "shrimp_landlease", "shrimp_no_landlease", "fish-rice_landlease", "fish-rice_no_landlease"]:
                if crop == "rice_irrig":
                    landowner_agents_xy[x,y,i] = np.around(hhmat[x,y] * tot_pop_agr[hh] * croppping_pattern[hh]['rice'] * irrigation_perc[hh] * land_ownership[hh]['landowner'])
                elif crop == "rice_no_irrig":
                    landowner_agents_xy[x,y,i] = np.around(hhmat[x,y] * tot_pop_agr[hh] * croppping_pattern[hh]['rice'] * (1.0 - irrigation_perc[hh]) * land_ownership[hh]['landowner'])
                elif crop == "rice_irrig_landlease":
                    landowner_agents_xy[x,y,i] = np.around(hhmat[x,y] * tot_pop_agr[hh] * croppping_pattern[hh]['rice'] * irrigation_perc[hh] * land_ownership[hh]['tenant'])
                elif crop == "rice_no_irrig_landlease":
                    landowner_agents_xy[x,y,i] = np.around(hhmat[x,y] * tot_pop_agr[hh] * croppping_pattern[hh]['rice'] * (1.0 - irrigation_perc[hh]) * land_ownership[hh]['tenant'])
                elif crop == "fish_landlease":
                    landowner_agents_xy[x,y,i] = np.around(hhmat[x,y] * tot_pop_agr[hh] * croppping_pattern[hh]['fish'] * land_ownership[hh]['tenant'])
                elif crop == "fish_no_landlease":
                    landowner_agents_xy[x,y,i] = np.around(hhmat[x,y] * tot_pop_agr[hh] * croppping_pattern[hh]['fish'] * land_ownership[hh]['landowner'])
                elif crop == "shrimp_landlease":
                    landowner_agents_xy[x,y,i] = np.around(hhmat[x,y] * tot_pop_agr[hh] * croppping_pattern[hh]['shrimp'] * land_ownership[hh]['tenant'])
                elif crop == "shrimp_no_landlease":
                    landowner_agents_xy[x,y,i] = np.around(hhmat[x,y] * tot_pop_agr[hh] * croppping_pattern[hh]['shrimp'] * land_ownership[hh]['landowner'])
                elif crop == "fish-rice_landlease":
                    landowner_agents_xy[x,y,i] = np.around(hhmat[x,y] * tot_pop_agr[hh] * croppping_pattern[hh]['fish-rice'] * land_ownership[hh]['tenant'])
                elif crop == "fish-rice_no_landlease":
                    landowner_agents_xy[x,y,i] = np.around(hhmat[x,y] * tot_pop_agr[hh] * croppping_pattern[hh]['fish-rice'] * land_ownership[hh]['landowner'])
                
                i = i + 1
        i = 0

# #verify agent numbers                    
# print(hhmat[x,y])
# print(landless_agents[x,y] + np.sum(landowner_agents_xy, axis = 2)[x,y])


#initial pcraster calculation
#set clonemap
height = np.shape(elevmat)[0]
width = np.shape(elevmat)[1]
pcr.setclone(height, width, 1, 0, 0)


#convert numpay arrays to PCR rasters
pcrelev = pcr.numpy2pcr(pcr.Scalar, elevmat, -999.)
pcr.report(pcrelev, r'p:\11208012-011-nabaripoma\Model\Python\results\real\maps\elevation.map')
pcrfloodelev = pcr.numpy2pcr(pcr.Scalar, elevmat, -999.)
pcr.report(pcrfloodelev, r'p:\11208012-011-nabaripoma\Model\Python\results\real\maps\floodelevation.map')
#create ldd
pcr.setglobaloption('lddin')
pcr.setglobaloption('unittrue')
pcrldd = pcr.lddcreate(pcrelev,1.0e+12,1.0e+12,1.0e+12,1.0e+12)
pcr.report(pcrldd, r'p:\11208012-011-nabaripoma\Model\Python\results\real\maps\ldd.map')
lddmat = pcr.pcr2numpy(pcrldd,-999)
#plot
plt.matshow(lddmat)
plt.title('ldd')
plt.colorbar()
plt.show()

#create river and sea array
rivmat = np.where(rsmat == 1, 1, 0)
#plot
plt.matshow(rivmat)
plt.title('river')
plt.colorbar()
plt.show()


#convert river and sea array to map
pcrriv = pcr.numpy2pcr(pcr.Boolean, rivmat, -999.)

#calculate distance to river over ldd
pcrdist2riv = pcr.ldddist(pcrldd,pcrriv,1.)
dist2rivmat = pcr.pcr2numpy(pcrdist2riv,-999)
#plot
plt.matshow(dist2rivmat)
plt.title('dist2riv_ldd')
plt.colorbar()
plt.show()
pcr.report(pcrdist2riv,r'p:\11208012-011-nabaripoma\Model\Python\results\real\maps\dist2riv.map')


#Masks
is_sea = rsmat==2
is_river = rsmat == 1
is_polder = polmat > 0
is_nopolder = polmat == 0
is_land = rsmat == 0

#%%RUN CALCULATION (Loop from 2022 to 2100)
#initialize arrays and lists
df = pd.DataFrame(columns=['Year', 'Strategy', 'Indicator', 'Polder','Value'])

#loop over timesteps
is_TRM = False
is_TRM_prev = False
trmlevelyear1 = np.full(np.shape(elevmat),0)
trmlevelyear2 = np.full(np.shape(elevmat),0)

print('******** Simulation starts ********')
for year in np.arange(startyear, endyear+1,1):
    """
    Run one iteration of the model. 
    """
    print(year)
    
    #BIOPHYSICAL
    
    # update topography
    # soil subsidence
    elevmat[is_land] = elevmat[is_land] - subsidence
    # #sedimentation on land outside polders
    # elevmat[(is_land & is_nopolder) | is_river] = elevmat[(is_land & is_nopolder) | is_river] + sedrate * 0.01
    
    #recalculate based on new elevation
    pcrelev = pcr.numpy2pcr(pcr.Scalar, elevmat, -999.)
    #create ldd
    pcrldd = pcr.lddcreate(pcrelev,1.0e+12,1.0e+12,1.0e+12,1.0e+12)
    lddmat = pcr.pcr2numpy(pcrldd,-999)

    #calculate distance to river over ldd
    pcrdist2riv = pcr.ldddist(pcrldd,pcrriv,1.)*cellsize
    dist2rivmat = pcr.pcr2numpy(pcrdist2riv,-999)

    # #calculate distance to sea over ldd
    # pcrdist2sea = pcr.ldddist(pcrldd,pcrsea,1.)
    # dist2seamat = pcr.pcr2numpy(pcrdist2sea,-999)
    
	#Create a map with unique values for te river cells (pcr.unigueid)
    pcrrivid = pcr.uniqueid(pcrriv)
    rividmat = pcr.pcr2numpy(pcrrivid,-999) 
    
	#Create subcatchments above the river cells (pcr.subcatchment(ldd, river cells)
    pcrsub = pcr.subcatchment(pcrldd, pcr.nominal(pcrrivid))

    #River bed level
    bedlevel=np.full_like(elevmat, -999)
    bedlevel[is_river]=elevmat[is_river]    
    
    pcrbedlevel = pcr.numpy2pcr(pcr.Scalar, bedlevel, -999.)
    pcrpolderbedlevel = pcr.areatotal(pcrbedlevel, pcrsub)
    polderbedlevelmat = pcr.pcr2numpy(pcrpolderbedlevel,0)
            
    #tidal range - for the moment a linear decrease with 2cm per km from 2m at sea
    #tidalrange = 2. - 0.02 * dist2seamat
    #tidalrange[tidalrange < 0.]= 0.
    tidalrange = np.full(np.shape(elevmat),2.5) #fixed tidal range  

    # Sea Level Rise        
    msl =  mslstart + slr2100 / (math.exp(kslr * (endyear - startyear)) - math.exp(0)) * (math.exp( kslr * ( year - startyear)) - 1)

    #Low tide levels
    #Upstream flow - dry season water level
    wl_dry=0.1
    #Dry low tide level
    lt_dry= np.maximum((msl + wl_dry - 0.5 * tidalrange), polderbedlevelmat)
    
    #Upstream flow - wet season water level
    wl_wet=1.0
    #Wet low tide level
    lt_wet= np.maximum((msl + wl_wet - 0.5 * tidalrange), polderbedlevelmat)
    
    #water logging - patches with gradient less than drainhead to low tide
    gradient_dry=np.full(np.shape(elevmat),np.nan)
    gradient_wet=np.full(np.shape(elevmat),np.nan)
    gradient_dry[is_polder] = (elevmat[is_polder] - lt_dry[is_polder]) / dist2rivmat[is_polder]
    gradient_wet[is_polder] = (elevmat[is_polder] - lt_wet[is_polder]) / dist2rivmat[is_polder]
       
    #Waterlogging
    is_waterlogged_dry = np.full(np.shape(elevmat),False)    
    is_waterlogged_wet = np.full(np.shape(elevmat),False) 
    is_waterlogged_dry[(is_polder) & (gradient_dry < mindraingrad)] = True    
    is_waterlogged_wet[(is_polder) & (gradient_wet < mindraingrad)] = True  
    
    #dry
    waterlogged_sev_dry = 1 - (gradient_dry / mindraingrad)
    waterlogged_sev_dry[waterlogged_sev_dry < 0.] = 0.
    waterlogged_sev_dry[waterlogged_sev_dry > 1.] = 1.
    
    #wet
    waterlogged_sev_wet = 1 - (gradient_wet / mindraingrad)
    waterlogged_sev_wet[waterlogged_sev_wet < 0.] = 0.
    waterlogged_sev_wet[waterlogged_sev_wet > 1.] = 1.

    #TRM
    is_TRM=False
        
    ht_wet= msl + wl_wet + 0.5 * tidalrange #calculate for each cell the high tide level of the nearest cell in the wet season

    #create ldd
    pcrldd = pcr.lddcreate(pcrelev,2*trmsedrate,1.0e+12,1.0e+12,1.0e+12)

    p_id_max = 0
    bheel_id_max = 0
    max_stored_volume = 0.0
    for p_id in np.arange(1,25): #for each polder: #create a list with bheels per polder, including their minimum elevation and the area and amount of sediment that can be stored with 80cm of sedimentation and average water logging severity
        poldermask = polmat == p_id
        pcrpoldermask = pcr.numpy2pcr(pcr.Boolean, poldermask, -999.)
        polderldd = pcr.lddmask(pcrldd, pcrpoldermask)
 
        #Determine pit cells
        pcrpits = pcr.pit(polderldd)
        pitsmat = pcr.pcr2numpy(pcrpits,0)
        
        n_bheels = np.max(pitsmat)
        
        #Make a map of upstream area of each bheel
        pcrpoldercatch = pcr.subcatchment(polderldd, pcrpits)
        poldercatch = pcr.pcr2numpy(pcrpoldercatch,0)
        
        for bheel in np.arange(1,n_bheels+1):
            pitelev = elevmat[pitsmat==bheel]
            h_wl = ht_wet[pitsmat==bheel]
            bheel_mask = np.full(np.shape(elevmat),0)
            bheel_mask[poldercatch==bheel] = 1 
            sed_thick = np.maximum(0.,np.minimum(pitelev + 2*trmsedrate,h_wl)-elevmat)*bheel_mask
            bheel_cells=np.sum(sed_thick>0.0)
            if bheel_cells > 0.0:
                stored_volume = (bheel_cells*cellsize*cellsize) * np.sum(sed_thick)/bheel_cells
            else:
                stored_volume = 0.0
                       
            waterlogged_sev_bheel= np.full(np.shape(elevmat),0)
            waterlogged_sev_bheel = waterlogged_sev_wet * bheel_mask

            if bheel_cells > 0.0:
                mean_watlog_bheel = np.sum(waterlogged_sev_bheel)/bheel_cells
            else:
                mean_watlog_bheel = 0.0
            
            if (stored_volume > max_stored_volume) and (mean_watlog_bheel>0.8):
                max_stored_volume=stored_volume
                p_id_max = p_id
                bheel_id_max = bheel

    if is_TRM_prev:
        elevmat = elevmat + trmlevelyear2
        trmlevelyear2=np.full(np.shape(elevmat), 0)
        is_TRM_prev = False

    if p_id_max > 0:
        is_TRM = True
        p_id=p_id_max
        bheel=bheel_id_max
        poldermask = polmat == p_id
        pcrpoldermask = pcr.numpy2pcr(pcr.Boolean, poldermask, -999.)
        polderldd = pcr.lddmask(pcrldd, pcrpoldermask)
 
        #Determine pit cells
        pcrpits = pcr.pit(polderldd)
        pitsmat = pcr.pcr2numpy(pcrpits,0)
               
        #Make a map of upstream area of each bheel
        pcrpoldercatch = pcr.subcatchment(polderldd, pcrpits)
        poldercatch = pcr.pcr2numpy(pcrpoldercatch,0)
        
        pitelev = elevmat[pitsmat==bheel]
        h_wl = ht_wet[pitsmat==bheel]
        bheel_mask = np.full(np.shape(elevmat),0)
        bheel_mask[poldercatch==bheel] = 1 
    
        trmlevelyear1 = np.maximum(0.0, np.minimum(pitelev + trmsedrate, h_wl) - elevmat) * bheel_mask
        trmlevelyear2 = np.maximum(0.0, np.minimum(pitelev + trmlevelyear1 + trmsedrate, h_wl) - elevmat) * bheel_mask
        plt.matshow(trmlevelyear1)
        plt.colorbar()
        plt.title('trmlevel_polder_' + str(p_id) + '_' +  str(year))
        plt.show()
        plt.close()
   
    if is_TRM:
        elevmat = elevmat + trmlevelyear1
        trmlevelyear1=np.full(np.shape(elevmat), 0)
        is_TRM_prev = True
    
    #PLOT
    #filename
    filename_waterlogging=r'p:\11208012-011-nabaripoma\Model\Python\results\real\waterlogging\waterlogging_' + str(year) + '.png'

    if plot:
        #plot
        plt.rcParams["figure.figsize"] = [22, 10]
        f, (ax1, ax2, ax3) = plt.subplots(1,3, sharex=True, sharey=True)
        f1=ax1.matshow(elevmat, vmin=-1, vmax = 10)
        ax1.set_title('elevation')
        plt.colorbar(f1,ax=ax1)
        f2=ax2.matshow(waterlogged_sev_dry, vmin=0, vmax = 1)
        ax2.set_title('waterlogged_sev_dry')
        plt.colorbar(f2,ax=ax2)
        f3=ax3.matshow(waterlogged_sev_wet, vmin=0, vmax = 1)
        ax3.set_title('waterlogged_sev_wet')
        plt.colorbar(f3,ax=ax3)
        f.suptitle(year, fontsize=16, x=0.5)
        plt.tight_layout()
        plt.savefig(filename_waterlogging, format='png', bbox_inches='tight', dpi=300)
        plt.show()
        plt.close()
    
    if raster:
        if year == startyear or year == endyear:            
            #Write raster file (waterlogging)
            with rasterio.open(r'p:\11208012-011-nabaripoma\Model\Python\results\real\waterlogging\geotif\waterlogging_dry_' + str(year) + '.tif', 'w', **ras_meta) as dst:
                dst.write(waterlogged_sev_dry, indexes=1) #Dry
                
            with rasterio.open(r'p:\11208012-011-nabaripoma\Model\Python\results\real\waterlogging\geotif\waterlogging_wet_' + str(year) + '.tif', 'w', **ras_meta) as dst:
                dst.write(waterlogged_sev_wet, indexes=1) #Wet   
            
            #Write raster file (elevation)   
            with rasterio.open(r'p:\11208012-011-nabaripoma\Model\Python\results\real\elevation\geotif\elevation_' + str(year) + '.tif', 'w', **ras_meta) as dst:
                dst.write(elevmat, indexes=1) #elevation
  
   
    #SOCIO-ECONOMICS
    
    #init arrays
    production_rice = np.zeros(np.shape(elevmat))
    production_fish = np.zeros(np.shape(elevmat))
    production_shrimp = np.zeros(np.shape(elevmat))
    pop_inc_below_pov = np.zeros(np.shape(elevmat))
    emp_perm = np.zeros(np.shape(elevmat))
    emp_seasonal = np.zeros(np.shape(elevmat))
    pop_food_insecure = np.zeros(np.shape(elevmat))
    pop_migration = np.zeros(np.shape(elevmat))
    
    # ind_name_list = ['production_rice', 'production_fish', 'production_shrimp', 'pop_inc_below_pov', 'emp_perm', 'emp_seasonal', 'pop_food_insecure', 'pop_migration']
    ind_name_list = [1, 2, 3, 4, 5, 6, 7, 8]
    ind_value_list= [production_rice, production_fish, production_shrimp, pop_inc_below_pov, emp_perm, emp_seasonal, pop_food_insecure, pop_migration]
    
    #i=0  rice_irrig_small                 #i=1  rice_irrig_med               #i=2  rice_irrig_large
    #i=3  rice_no_irrig_small              #i=4  rice_no_irrig_med            #i=5  rice_no_irrig_large
    #i=6  rice_irrig_landlease_small       #i=7  rice_irrig_landlease_med     #i=8  rice_irrig_landlease_large
    #i=9  rice_no_irrig_landlease_small    #i=10 rice_no_irrig_landlease_med  #i=11 rice_no_irrig_landlease_large
    #i=12 fish_landlease_small             #i=13 fish_landlease_med           #i=14 fish_landlease_large
    #i=15 fish_no_landlease_small          #i=16 fish_no_landlease_med        #i=17 fish_no_landlease_large
    #i=18 shrimp_landlease_small           #i=19 shrimp_landlease_med         #i=20 shrimp_landlease_large
    #i=21 shrimp_no_landlease_small        #i=22 shrimp_no_landlease_med      #i=23 shrimp_no_landlease_large
    #i=24 fish-rice_landlease_small        #i=25 fish-rice_landlease_med      #i=26 fish-rice_landlease_large
    #i=27 fish-rice_no_landlease_small     #i=28 fish-rice_no_landlease_med   #i=29 fish-rice_no_landlease_large
    
    
    #Calculate income, food security and migration with wet and dry season water logging severity as input
    for x in np.arange(0, np.shape(elevmat)[0]):
        for y in np.arange(0, np.shape(elevmat)[1]):
            population=np.sum(landowner_agents_xy[x,y,:]) + landless_agents_perm[x,y] + landless_agents_seas[x,y]
            #landless agents
            if landless_agents_perm[x,y] >= 1.0:
                for no_agent in np.arange(1, landless_agents_perm[x,y]+1):
                    (production, income_above_poverty, req_perm_farm_empl, req_seasonal_farm_empl, food_security, migration_family, landless_farmer) = agent_functions(waterlogged_sev_wet[x,y])            
                    
                    pop_inc_below_pov[x,y] = pop_inc_below_pov[x,y] + landless_farmer['income_above_poverty']['perm_empl']
                    pop_food_insecure[x,y] = pop_food_insecure[x,y] + landless_farmer['food_security']['perm_empl']

            if landless_agents_seas[x,y] >= 1.0:
                for no_agent in np.arange(1, landless_agents_seas[x,y]+1):
                    (production, income_above_poverty, req_perm_farm_empl, req_seasonal_farm_empl, food_security, migration_family, landless_farmer) = agent_functions(waterlogged_sev_wet[x,y])            
                    
                    pop_inc_below_pov[x,y] = pop_inc_below_pov[x,y] + landless_farmer['income_above_poverty']['seasonal_empl'] 
                    pop_food_insecure[x,y] = pop_food_insecure[x,y] + landless_farmer['food_security']['seasonal_empl']
                    
            else:
                pass
            
            #landowner agents
            for i in np.arange(0, 30):
                if landowner_agents_xy[x,y,i] >= 1.0:
                    for no_agent in np.arange(1, landowner_agents_xy[x,y,i]+1):
                        (production, income_above_poverty, req_perm_farm_empl, req_seasonal_farm_empl, food_security, migration_family, landless_farmer) = agent_functions(0.8) #agent_functions(waterlogged_sev_wet[x,y])
                      
                        #update indicators per agent
                        if i==0:
                            production_rice[x,y] = production_rice[x,y] + production['rice']['small']
                            pop_inc_below_pov[x,y] = pop_inc_below_pov[x,y] + income_above_poverty['rice_irrig']['small']  
                            emp_perm[x,y] = emp_perm[x,y] + req_perm_farm_empl['rice']['small']
                            emp_seasonal[x,y] = emp_seasonal[x,y] + req_seasonal_farm_empl['rice']['small']
                            pop_food_insecure[x,y] = pop_food_insecure[x,y] + food_security['rice_irrig']['small'] 
                            pop_migration[x,y] = pop_migration[x,y] + migration_family['rice_irrig']['small'] 
                        elif i==1:
                            production_rice[x,y] = production_rice[x,y] + production['rice']['med']
                            pop_inc_below_pov[x,y] = pop_inc_below_pov[x,y] + income_above_poverty['rice_irrig']['med'] 
                            emp_perm[x,y] = emp_perm[x,y] + req_perm_farm_empl['rice']['med']
                            emp_seasonal[x,y] = emp_seasonal[x,y] + req_seasonal_farm_empl['rice']['med']
                            pop_food_insecure[x,y] = pop_food_insecure[x,y] + food_security['rice_irrig']['med'] 
                            pop_migration[x,y] = pop_migration[x,y] + migration_family['rice_irrig']['med']
                        elif i==2:
                            production_rice[x,y] = production_rice[x,y] + production['rice']['large']
                            pop_inc_below_pov[x,y] = pop_inc_below_pov[x,y] + income_above_poverty['rice_irrig']['large'] 
                            emp_perm[x,y] = emp_perm[x,y] + req_perm_farm_empl['rice']['large']
                            emp_seasonal[x,y] = emp_seasonal[x,y] + req_seasonal_farm_empl['rice']['large']
                            pop_food_insecure[x,y] = pop_food_insecure[x,y] + food_security['rice_irrig']['large'] 
                            pop_migration[x,y] = pop_migration[x,y] + migration_family['rice_irrig']['large'] 
                        elif i==3:
                            production_rice[x,y] = production_rice[x,y] + production['rice']['small']                            
                            pop_inc_below_pov[x,y] = pop_inc_below_pov[x,y] + income_above_poverty['rice_no_irrig']['small']  
                            emp_perm[x,y] = emp_perm[x,y] + req_perm_farm_empl['rice']['small']
                            emp_seasonal[x,y] = emp_seasonal[x,y] + req_seasonal_farm_empl['rice']['small']
                            pop_food_insecure[x,y] = pop_food_insecure[x,y] + food_security['rice_no_irrig']['small'] 
                            pop_migration[x,y] = pop_migration[x,y] + migration_family['rice_no_irrig']['small'] 
                        elif i==4:
                            production_rice[x,y] = production_rice[x,y] + production['rice']['med']
                            pop_inc_below_pov[x,y] = pop_inc_below_pov[x,y] + income_above_poverty['rice_no_irrig']['med'] 
                            emp_perm[x,y] = emp_perm[x,y] + req_perm_farm_empl['rice']['med']
                            emp_seasonal[x,y] = emp_seasonal[x,y] + req_seasonal_farm_empl['rice']['med']
                            pop_food_insecure[x,y] = pop_food_insecure[x,y] + food_security['rice_no_irrig']['med'] 
                            pop_migration[x,y] = pop_migration[x,y] + migration_family['rice_no_irrig']['med'] 
                        elif i==5:
                            production_rice[x,y] = production_rice[x,y] + production['rice']['large']
                            pop_inc_below_pov[x,y] = pop_inc_below_pov[x,y] + income_above_poverty['rice_no_irrig']['large'] 
                            emp_perm[x,y] = emp_perm[x,y] + req_perm_farm_empl['rice']['large']
                            emp_seasonal[x,y] = emp_seasonal[x,y] + req_seasonal_farm_empl['rice']['large']
                            pop_food_insecure[x,y] = pop_food_insecure[x,y] + food_security['rice_no_irrig']['large'] 
                            pop_migration[x,y] = pop_migration[x,y] + migration_family['rice_no_irrig']['large']
                        elif i==6: 
                            production_rice[x,y] = production_rice[x,y] + production['rice']['small']
                            pop_inc_below_pov[x,y] = pop_inc_below_pov[x,y] + income_above_poverty['rice_irrig_landlease']['small']
                            emp_perm[x,y] = emp_perm[x,y] + req_perm_farm_empl['rice']['small']
                            emp_seasonal[x,y] = emp_seasonal[x,y] + req_seasonal_farm_empl['rice']['small']
                            pop_food_insecure[x,y] = pop_food_insecure[x,y] + food_security['rice_irrig_landlease']['small'] 
                            pop_migration[x,y] = pop_migration[x,y] + migration_family['rice_irrig_landlease']['small'] 
                        elif i==7:
                            production_rice[x,y] = production_rice[x,y] + production['rice']['med']
                            pop_inc_below_pov[x,y] = pop_inc_below_pov[x,y] + income_above_poverty['rice_irrig_landlease']['med']
                            emp_perm[x,y] = emp_perm[x,y] + req_perm_farm_empl['rice']['med']
                            emp_seasonal[x,y] = emp_seasonal[x,y] + req_seasonal_farm_empl['rice']['med']
                            pop_food_insecure[x,y] = pop_food_insecure[x,y] + food_security['rice_irrig_landlease']['med'] 
                            pop_migration[x,y] = pop_migration[x,y] + migration_family['rice_irrig_landlease']['med'] 

                        elif i==8:  
                            production_rice[x,y] = production_rice[x,y] + production['rice']['large']
                            pop_inc_below_pov[x,y] = pop_inc_below_pov[x,y] + income_above_poverty['rice_irrig_landlease']['large']
                            emp_perm[x,y] = emp_perm[x,y] + req_perm_farm_empl['rice']['large']
                            emp_seasonal[x,y] = emp_seasonal[x,y] + req_seasonal_farm_empl['rice']['large']
                            pop_food_insecure[x,y] = pop_food_insecure[x,y] + food_security['rice_irrig_landlease']['large'] 
                            pop_migration[x,y] = pop_migration[x,y] + migration_family['rice_irrig_landlease']['large'] 
                        
                        elif i==9:
                            production_rice[x,y] = production_rice[x,y] + production['rice']['small']
                            pop_inc_below_pov[x,y] = pop_inc_below_pov[x,y] + income_above_poverty['rice_no_irrig_landlease']['small']
                            emp_perm[x,y] = emp_perm[x,y] + req_perm_farm_empl['rice']['small']
                            emp_seasonal[x,y] = emp_seasonal[x,y] + req_seasonal_farm_empl['rice']['small']
                            pop_food_insecure[x,y] = pop_food_insecure[x,y] + food_security['rice_no_irrig_landlease']['small'] 
                            pop_migration[x,y] = pop_migration[x,y] + migration_family['rice_no_irrig_landlease']['small'] 

                        elif i==10:
                            production_rice[x,y] = production_rice[x,y] + production['rice']['med']
                            pop_inc_below_pov[x,y] = pop_inc_below_pov[x,y] + income_above_poverty['rice_no_irrig_landlease']['med']
                            emp_perm[x,y] = emp_perm[x,y] + req_perm_farm_empl['rice']['med']
                            emp_seasonal[x,y] = emp_seasonal[x,y] + req_seasonal_farm_empl['rice']['med']
                            pop_food_insecure[x,y] = pop_food_insecure[x,y] + food_security['rice_no_irrig_landlease']['med'] 
                            pop_migration[x,y] = pop_migration[x,y] + migration_family['rice_no_irrig_landlease']['med'] 

                        elif i==11:   
                            production_rice[x,y] = production_rice[x,y] + production['rice']['large']
                            pop_inc_below_pov[x,y] = pop_inc_below_pov[x,y] + income_above_poverty['rice_no_irrig_landlease']['large']
                            emp_perm[x,y] = emp_perm[x,y] + req_perm_farm_empl['rice']['large']
                            emp_seasonal[x,y] = emp_seasonal[x,y] + req_seasonal_farm_empl['rice']['large']
                            pop_food_insecure[x,y] = pop_food_insecure[x,y] + food_security['rice_no_irrig_landlease']['large'] 
                            pop_migration[x,y] = pop_migration[x,y] + migration_family['rice_no_irrig_landlease']['large'] 
                            
                        elif i==12:
                            production_rice[x,y] = production_rice[x,y] + production['fish']['small']
                            pop_inc_below_pov[x,y] = pop_inc_below_pov[x,y] + income_above_poverty['fish_landlease']['small']
                            emp_perm[x,y] = emp_perm[x,y] + req_perm_farm_empl['fish']['small']
                            emp_seasonal[x,y] = emp_seasonal[x,y] + req_seasonal_farm_empl['fish']['small']
                            pop_food_insecure[x,y] = pop_food_insecure[x,y] + food_security['fish_landlease']['small'] 
                            pop_migration[x,y] = pop_migration[x,y] + migration_family['fish_landlease']['small'] 
                            
                        elif i==13:  
                            production_rice[x,y] = production_rice[x,y] + production['fish']['med']
                            pop_inc_below_pov[x,y] = pop_inc_below_pov[x,y] + income_above_poverty['fish_landlease']['med']
                            emp_perm[x,y] = emp_perm[x,y] + req_perm_farm_empl['fish']['med']
                            emp_seasonal[x,y] = emp_seasonal[x,y] + req_seasonal_farm_empl['fish']['med']
                            pop_food_insecure[x,y] = pop_food_insecure[x,y] + food_security['fish_landlease']['med'] 
                            pop_migration[x,y] = pop_migration[x,y] + migration_family['fish_landlease']['med'] 
                        
                        elif i==14:
                            production_rice[x,y] = production_rice[x,y] + production['fish']['large']
                            pop_inc_below_pov[x,y] = pop_inc_below_pov[x,y] + income_above_poverty['fish_landlease']['large']
                            emp_perm[x,y] = emp_perm[x,y] + req_perm_farm_empl['fish']['large']
                            emp_seasonal[x,y] = emp_seasonal[x,y] + req_seasonal_farm_empl['fish']['large']
                            pop_food_insecure[x,y] = pop_food_insecure[x,y] + food_security['fish_landlease']['large'] 
                            pop_migration[x,y] = pop_migration[x,y] + migration_family['fish_landlease']['large'] 

                        elif i==15:
                            production_rice[x,y] = production_rice[x,y] + production['fish']['small']
                            pop_inc_below_pov[x,y] = pop_inc_below_pov[x,y] + income_above_poverty['fish_no_landlease']['small']
                            emp_perm[x,y] = emp_perm[x,y] + req_perm_farm_empl['fish']['small']
                            emp_seasonal[x,y] = emp_seasonal[x,y] + req_seasonal_farm_empl['fish']['small']
                            pop_food_insecure[x,y] = pop_food_insecure[x,y] + food_security['fish_no_landlease']['small'] 
                            pop_migration[x,y] = pop_migration[x,y] + migration_family['fish_no_landlease']['small'] 

                        elif i==16:  
                            production_rice[x,y] = production_rice[x,y] + production['fish']['med']
                            pop_inc_below_pov[x,y] = pop_inc_below_pov[x,y] + income_above_poverty['fish_no_landlease']['med']
                            emp_perm[x,y] = emp_perm[x,y] + req_perm_farm_empl['fish']['med']
                            emp_seasonal[x,y] = emp_seasonal[x,y] + req_seasonal_farm_empl['fish']['med']
                            pop_food_insecure[x,y] = pop_food_insecure[x,y] + food_security['fish_no_landlease']['med'] 
                            pop_migration[x,y] = pop_migration[x,y] + migration_family['fish_no_landlease']['med'] 

                        elif i==17:
                            production_rice[x,y] = production_rice[x,y] + production['fish']['large']
                            pop_inc_below_pov[x,y] = pop_inc_below_pov[x,y] + income_above_poverty['fish_no_landlease']['large']
                            emp_perm[x,y] = emp_perm[x,y] + req_perm_farm_empl['fish']['large']
                            emp_seasonal[x,y] = emp_seasonal[x,y] + req_seasonal_farm_empl['fish']['large']
                            pop_food_insecure[x,y] = pop_food_insecure[x,y] + food_security['fish_no_landlease']['large'] 
                            pop_migration[x,y] = pop_migration[x,y] + migration_family['fish_no_landlease']['large'] 

                        elif i==18:       
                            production_shrimp[x,y] = production_shrimp[x,y] + production['shrimp']['small']
                            pop_inc_below_pov[x,y] = pop_inc_below_pov[x,y] + income_above_poverty['shrimp_landlease']['small']
                            emp_perm[x,y] = emp_perm[x,y] + req_perm_farm_empl['shrimp']['small']
                            emp_seasonal[x,y] = emp_seasonal[x,y] + req_seasonal_farm_empl['shrimp']['small']
                            pop_food_insecure[x,y] = pop_food_insecure[x,y] + food_security['shrimp_landlease']['small'] 
                            pop_migration[x,y] = pop_migration[x,y] + migration_family['shrimp_landlease']['small'] 
                        
                        elif i==19:
                            production_shrimp[x,y] = production_shrimp[x,y] + production['shrimp']['med']
                            pop_inc_below_pov[x,y] = pop_inc_below_pov[x,y] + income_above_poverty['shrimp_landlease']['med']
                            emp_perm[x,y] = emp_perm[x,y] + req_perm_farm_empl['shrimp']['med']
                            emp_seasonal[x,y] = emp_seasonal[x,y] + req_seasonal_farm_empl['shrimp']['med']
                            pop_food_insecure[x,y] = pop_food_insecure[x,y] + food_security['shrimp_landlease']['med'] 
                            pop_migration[x,y] = pop_migration[x,y] + migration_family['shrimp_landlease']['med'] 

                        elif i==20:
                            production_shrimp[x,y] = production_shrimp[x,y] + production['shrimp']['large']
                            pop_inc_below_pov[x,y] = pop_inc_below_pov[x,y] + income_above_poverty['shrimp_landlease']['large']
                            emp_perm[x,y] = emp_perm[x,y] + req_perm_farm_empl['shrimp']['large']
                            emp_seasonal[x,y] = emp_seasonal[x,y] + req_seasonal_farm_empl['shrimp']['large']
                            pop_food_insecure[x,y] = pop_food_insecure[x,y] + food_security['shrimp_landlease']['large'] 
                            pop_migration[x,y] = pop_migration[x,y] + migration_family['shrimp_landlease']['large'] 

                        elif i==21:  
                            production_shrimp[x,y] = production_shrimp[x,y] + production['shrimp']['small']
                            pop_inc_below_pov[x,y] = pop_inc_below_pov[x,y] + income_above_poverty['shrimp_no_landlease']['small']
                            emp_perm[x,y] = emp_perm[x,y] + req_perm_farm_empl['shrimp']['small']
                            emp_seasonal[x,y] = emp_seasonal[x,y] + req_seasonal_farm_empl['shrimp']['small']
                            pop_food_insecure[x,y] = pop_food_insecure[x,y] + food_security['shrimp_no_landlease']['small'] 
                            pop_migration[x,y] = pop_migration[x,y] + migration_family['shrimp_no_landlease']['small'] 
                            
                        elif i==22:  
                            production_shrimp[x,y] = production_shrimp[x,y] + production['shrimp']['med']
                            pop_inc_below_pov[x,y] = pop_inc_below_pov[x,y] + income_above_poverty['shrimp_no_landlease']['med']
                            emp_perm[x,y] = emp_perm[x,y] + req_perm_farm_empl['shrimp']['med']
                            emp_seasonal[x,y] = emp_seasonal[x,y] + req_seasonal_farm_empl['shrimp']['med']
                            pop_food_insecure[x,y] = pop_food_insecure[x,y] + food_security['shrimp_no_landlease']['med'] 
                            pop_migration[x,y] = pop_migration[x,y] + migration_family['shrimp_no_landlease']['med'] 
                        
                        elif i==23:
                            production_shrimp[x,y] = production_shrimp[x,y] + production['shrimp']['large']
                            pop_inc_below_pov[x,y] = pop_inc_below_pov[x,y] + income_above_poverty['shrimp_no_landlease']['large']
                            emp_perm[x,y] = emp_perm[x,y] + req_perm_farm_empl['shrimp']['large']
                            emp_seasonal[x,y] = emp_seasonal[x,y] + req_seasonal_farm_empl['shrimp']['large']
                            pop_food_insecure[x,y] = pop_food_insecure[x,y] + food_security['shrimp_no_landlease']['large'] 
                            pop_migration[x,y] = pop_migration[x,y] + migration_family['shrimp_no_landlease']['large'] 

                        elif i==24:
                            production_rice[x,y] = production_rice[x,y] + production['fish-rice']['rice']['small']
                            production_fish[x,y] = production_fish[x,y] + production['fish-rice']['fish']['small']   
                            pop_inc_below_pov[x,y] = pop_inc_below_pov[x,y] + income_above_poverty['fish-rice_landlease']['small']
                            emp_perm[x,y] = emp_perm[x,y] + req_perm_farm_empl['fish-rice']['small']
                            emp_seasonal[x,y] = emp_seasonal[x,y] + req_seasonal_farm_empl['fish-rice']['small']
                            pop_food_insecure[x,y] = pop_food_insecure[x,y] + food_security['fish-rice_landlease']['small'] 
                            pop_migration[x,y] = pop_migration[x,y] + migration_family['fish-rice_landlease']['small'] 

                        elif i==25:                            
                            production_rice[x,y] = production_rice[x,y] + production['fish-rice']['rice']['med']
                            production_fish[x,y] = production_fish[x,y] + production['fish-rice']['fish']['med']
                            pop_inc_below_pov[x,y] = pop_inc_below_pov[x,y] + income_above_poverty['fish-rice_landlease']['med']
                            emp_perm[x,y] = emp_perm[x,y] + req_perm_farm_empl['fish-rice']['med']
                            emp_seasonal[x,y] = emp_seasonal[x,y] + req_seasonal_farm_empl['fish-rice']['med']
                            pop_food_insecure[x,y] = pop_food_insecure[x,y] + food_security['fish-rice_landlease']['med'] 
                            pop_migration[x,y] = pop_migration[x,y] + migration_family['fish-rice_landlease']['med'] 
                            
                            
                        elif i==26:
                            production_rice[x,y] = production_rice[x,y] + production['fish-rice']['rice']['large']
                            production_fish[x,y] = production_fish[x,y] + production['fish-rice']['fish']['large']
                            pop_inc_below_pov[x,y] = pop_inc_below_pov[x,y] + income_above_poverty['fish-rice_landlease']['large']
                            emp_perm[x,y] = emp_perm[x,y] + req_perm_farm_empl['fish-rice']['large']
                            emp_seasonal[x,y] = emp_seasonal[x,y] + req_seasonal_farm_empl['fish-rice']['large']
                            pop_food_insecure[x,y] = pop_food_insecure[x,y] + food_security['fish-rice_landlease']['large'] 
                            pop_migration[x,y] = pop_migration[x,y] + migration_family['fish-rice_landlease']['large'] 
                            
                        elif i==27:                            
                            production_rice[x,y] = production_rice[x,y] + production['fish-rice']['rice']['small']
                            production_fish[x,y] = production_fish[x,y] + production['fish-rice']['fish']['small']    
                            pop_inc_below_pov[x,y] = pop_inc_below_pov[x,y] + income_above_poverty['fish-rice_no_landlease']['small']
                            emp_perm[x,y] = emp_perm[x,y] + req_perm_farm_empl['fish-rice']['small']
                            emp_seasonal[x,y] = emp_seasonal[x,y] + req_seasonal_farm_empl['fish-rice']['small']
                            pop_food_insecure[x,y] = pop_food_insecure[x,y] + food_security['fish-rice_no_landlease']['small'] 
                            pop_migration[x,y] = pop_migration[x,y] + migration_family['fish-rice_no_landlease']['small'] 
                            
                        elif i==28:
                            production_rice[x,y] = production_rice[x,y] + production['fish-rice']['rice']['med']
                            production_fish[x,y] = production_fish[x,y] + production['fish-rice']['fish']['med']
                            pop_inc_below_pov[x,y] = pop_inc_below_pov[x,y] + income_above_poverty['fish-rice_no_landlease']['med']
                            emp_perm[x,y] = emp_perm[x,y] + req_perm_farm_empl['fish-rice']['med']
                            emp_seasonal[x,y] = emp_seasonal[x,y] + req_seasonal_farm_empl['fish-rice']['med']
                            pop_food_insecure[x,y] = pop_food_insecure[x,y] + food_security['fish-rice_no_landlease']['med'] 
                            pop_migration[x,y] = pop_migration[x,y] + migration_family['fish-rice_no_landlease']['med'] 
                            
                        elif i==29:
                            production_rice[x,y] = production_rice[x,y] + production['fish-rice']['rice']['large']
                            production_fish[x,y] = production_fish[x,y] + production['fish-rice']['fish']['large']
                            pop_inc_below_pov[x,y] = pop_inc_below_pov[x,y] + income_above_poverty['fish-rice_no_landlease']['large']
                            emp_perm[x,y] = emp_perm[x,y] + req_perm_farm_empl['fish-rice']['large']
                            emp_seasonal[x,y] = emp_seasonal[x,y] + req_seasonal_farm_empl['fish-rice']['large']
                            pop_food_insecure[x,y] = pop_food_insecure[x,y] + food_security['fish-rice_no_landlease']['large'] 
                            pop_migration[x,y] = pop_migration[x,y] + migration_family['fish-rice_no_landlease']['large'] 
                     
                else:
                    pass
            
            if population > 1.0:
                pop_inc_below_pov[x,y] = (1.0 - (pop_inc_below_pov[x,y] / population)) * 100.0  
                pop_food_insecure[x,y] = (1.0 - (pop_food_insecure[x,y] / population)) * 100.0   
            else:
                pass

            if landless_agents_perm[x,y] > 1.0:            
                if landless_agents_perm[x,y]*(42.0*5.0) < emp_perm[x,y] * 42.0 * 5.0 :
                    landless_migration_perm = 1.0
                else:
                    landless_migration_perm = (emp_perm[x,y] * 42.0 * 5.0) / (landless_agents_perm[x,y]*(42.0*5.0))
            else: 
                landless_migration_perm = 0.0 
                
            if landless_agents_seas[x,y] > 1.0:              
                if landless_agents_seas[x,y]*(42.0*5.0) < emp_seasonal[x,y] * 42.0 * 5.0 :
                    landless_migration_seas = 1.0
                else:
                    landless_migration_seas = (emp_seasonal[x,y] * 42.0 * 5.0) / (landless_agents_seas[x,y]*(42.0*5.0))
            else: 
                landless_migration_seas = 0.0 

            pop_migration[x,y] = np.nansum([pop_migration[x,y], (landless_migration_perm*landless_agents_perm[x,y] + landless_migration_seas*landless_agents_seas[x,y])*4.15])

    #Write and save .csv                 
    #update dataframe
    #df = pd.concat([df.copy(),pd.DataFrame([{'Year':year, 'Indicator':'gross_income_rice_small', 'Polder':0, 'Value':np.mean(farm_gross_income_rice_small[polmat!=0])}])])
    for ind in np.arange(0, len(ind_name_list)):
        for p in np.arange(1, no_polder+1):
            df = pd.concat( [df.copy(),pd.DataFrame([{'Year':year, 'Strategy':strategy, 'Indicator': ind_name_list[ind], 'Polder':p, 'Value': np.mean(ind_value_list[ind][polmat==p]) }])] )

#Save .csv
df.to_csv(r'p:\11208012-011-nabaripoma\Model\Python\results\real\csv\model_output.csv', index=False, float_format='%.2f')

