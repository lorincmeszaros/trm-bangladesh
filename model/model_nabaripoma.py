# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 15:44:47 2022

@author: lorinc
"""
import numpy as np
import math as math
import pcraster as pcr
from matplotlib import pyplot as plt 

model_params = {
    "height": 100,
    "width": 20,
    "slr2100": 1, #UserSettableParameter("slider", "Mean Sea Level Rise to 2100 (m)", 1.00, 0.00, 5.00, 0.01),
    "subsidence": 2, #UserSettableParameter("slider", "Soil subsidence rate (cm/year)", 2, 0, 10, 1),
    "sedrate": 3, #UserSettableParameter("slider", "Sedimentation Rate Outside Polders (cm/year)", 3, 0, 10, 1),
    "trmsedrate": 20, #UserSettableParameter("slider", "Sedimentation Rate TRM Polders (cm/year)", 20, 0, 100, 1),
}

#Main model
height = model_params['height']
width = model_params['width']
slr2100 = model_params['slr2100']
subsidence = model_params['subsidence']
sedrate = model_params['sedrate']
trmsedrate = model_params['trmsedrate']
showvar = model_params['showvar']
mslstart = 0.00
startyear = 2022
endyear = 2100
kslr = 0.02
mindraingrad = 0.1 / 1000. # 10cm per km minimum drainage gradient

year = startyear
msl = 0.00
   
# Read the elevation from elevation.asc
elevmat = np.loadtxt(r'p:\11208012-011-nabaripoma\Model\Mesa\NaBaRiPoMa01\elevation.asc', delimiter = ' ', skiprows = 6) * 0.01

# read the location of rivers from file rs.asc
rsmat = np.loadtxt(r'p:\11208012-011-nabaripoma\Model\Mesa\NaBaRiPoMa01\rs.asc', delimiter = ' ', skiprows = 6)

# read the location of polders from file polder.asc
polmat = np.loadtxt(r'p:\11208012-011-nabaripoma\Model\Mesa\NaBaRiPoMa01\Polders.asc', delimiter = ' ', skiprows = 6)

#initial pcraster calculation
#set clonemap
pcr.setclone(height, width, 1, 0, 0)
#set floodelevmat with polders 10m higher
floodelevmat = np.where(polmat > 0, elevmat + 10., elevmat)
#convert numpay arrays to PCR rasters
pcrelev = pcr.numpy2pcr(pcr.Scalar, elevmat, -999.)
pcr.report(pcrelev, 'elevation.map')
pcrfloodelev = pcr.numpy2pcr(pcr.Scalar, floodelevmat, -999.)
pcr.report(pcrfloodelev, 'elevation.map')
#create ldd
pcr.setglobaloption('lddin')
pcr.setglobaloption('unittrue')
pcrldd = pcr.lddcreate(pcrfloodelev,9999999,9999999,9999999,9999999)
pcr.report(pcrldd,'ldd.map')
#create river and sea array
rivmat = np.where(rsmat == 2, 1, 0)
seamat = np.where(rsmat == 1, 1, 0)
#convert river and sea array to map
pcrriv = pcr.numpy2pcr(pcr.Boolean, rivmat, -999.)
pcrsea = pcr.numpy2pcr(pcr.Boolean, seamat, -999.)
#calculate distance to river over ldd
pcrdist2riv = pcr.ldddist(pcrldd,pcrriv,1.)
dist2rivmat = pcr.pcr2numpy(pcrdist2riv,-999)
pcr.report(pcrdist2riv,r'p:\11208012-011-nabaripoma\Model\Mesa\NaBaRiPoMa01\dist2riv.map')
#calculate distance to sea over ldd
pcrdist2sea = pcr.ldddist(pcrldd,pcrsea,1.)
dist2seamat = pcr.pcr2numpy(pcrdist2sea,-999)
pcr.report(pcrdist2sea,r'p:\11208012-011-nabaripoma\Model\Mesa\NaBaRiPoMa01\dist2sea.map')


for year in np.arange(startyear, endyear+1,1):
    """
    Run one step of the model. 
    """
    print(year)
    # Sea Level Rise        
    msl =  mslstart + slr2100 / (math.exp(kslr * (endyear - startyear)) - math.exp(0)) * (math.exp( kslr * ( year - startyear)) - 1)
    #recalculate based on new elevation
    pcrelev = pcr.numpy2pcr(pcr.Scalar, elevmat, -999.)
    #create ldd
    pcrldd = pcr.lddcreate(pcrelev,9999999,9999999,9999999,9999999)
    #convert river and sea array to map
    pcrriv = pcr.numpy2pcr(pcr.Boolean, rivmat, -999.)
    pcrsea = pcr.numpy2pcr(pcr.Boolean, seamat, -999.)
    #calculate distance to river over ldd
    pcrdist2riv = pcr.ldddist(pcrldd,pcrriv,1.)
    dist2rivmat = pcr.pcr2numpy(pcrdist2riv,-999)
    #calculate distance to sea over ldd
    pcrdist2sea = pcr.ldddist(pcrldd,pcrsea,1.)
    dist2seamat = pcr.pcr2numpy(pcrdist2sea,-999)
    
    #Physics calculation
    productivity = 0.0
    is_waterlogged = np.full(np.shape(elevmat),False)
    is_trm = np.full(np.shape(elevmat),False)

    is_sea = rsmat==1
    is_river = rsmat == 2
    is_polder = polmat > 0
    is_nopolder = polmat == 0
    is_land = rsmat == 0

    
    # update topography
    # soil subsidence
    elevmat[is_land] = elevmat[is_land] - subsidence * 0.01
    #sedimentation on land outside polders
    elevmat[is_land & is_nopolder] = elevmat[is_land & is_nopolder] + sedrate * 0.01
    #sedimentation in trm areas
    elevmat[is_trm] = elevmat[is_trm] + trmsedrate * 0.01
    
    #tidal range - for the moment a linear decrease with 2cm per km from 2m at sea
    tidalrange = 2. - 0.02 * dist2seamat
    tidalrange[tidalrange < 0.]= 0.
    # #flood depth - high tide minus elevation
    flooddepth=np.zeros(np.shape(elevmat))
    flooddepth[((is_nopolder) | (is_trm) | (is_river))] = msl + tidalrange[((is_nopolder) | (is_trm) | (is_river))] * 0.5 - elevmat[((is_nopolder) | (is_trm) | (is_river))]
    flooddepth[flooddepth < 0.] = 0.
        
    # #upstream drainage area for each river cell as number of nopolder and trm cells with pycor > pycor of the patch itself    
    # if (is_river):    
    #     rivy = pos[1]
    #     usdraina = 0.
    #     prism = 0.
    #     y = cell[2]
    #     upagent = cell[0]
    #     if (y > rivy) and ((upagent.is_nopolder) or (upagent.is_trm) or (upagent.is_river)):
    #         usdraina += 1
    #         prism += upagent.flooddepth
    
        #upstream flow - perhaps later, for now zero
        
        #polder drainage - perhaps later, for now zero
        
    #water logging - patches with gradient less than drainhead to low tide
    gradient=np.full(np.shape(elevmat),-999)
    gradient[is_land] = (elevmat[is_land] - (msl - 0.5 * tidalrange[is_land] )) / dist2rivmat[is_land]
    is_waterlogged[(is_land) & (gradient < mindraingrad)] = True
    
    #plot
    plt.imshow(is_waterlogged)
    plt.title(year)
    plt.show()

    #river flow
    
    #river bed --> update elevation
    
    #river salt - later, for now fixed