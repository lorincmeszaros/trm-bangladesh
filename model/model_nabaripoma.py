# -*- coding: utf-8 -*-
"""
Model to assess the combined bio-physical and behavioural developments
along an imaginary river stretch in the coastal zone of Bangladesh.
This model has been implemented in Python.
This demontrsation has been developed as part of the 2022 research project: 
'Integrated Assessment Modelling of Tidal River Management in Bangladesh'.
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
    "sedrate": 10, #UserSettableParameter("slider", "Sedimentation Rate Outside Polders (cm/year)", 3, 0, 10, 1),
    "trmsedrate": 40, #UserSettableParameter("slider", "Sedimentation Rate TRM Polders (cm/year)", 20, 0, 100, 1),
}

#Main model
height = model_params['height']
width = model_params['width']
slr2100 = model_params['slr2100']
subsidence = model_params['subsidence']
sedrate = model_params['sedrate']
trmsedrate = model_params['trmsedrate']
mslstart = 0.00
startyear = 2022
endyear = 2100
kslr = 0.02
mindraingrad = 0.1 / 1000. # 10cm per km minimum drainage gradient

year = startyear
msl = 0.00
   
# Read the elevation from elevation.asc
elevmat = np.loadtxt(r'p:\11208012-011-nabaripoma\Model\Mesa\NaBaRiPoMa01\elevation.asc', delimiter = ' ', skiprows = 6) * 0.01
#plot
plt.matshow(elevmat)
plt.title('elevation')
plt.colorbar()
plt.show()

# read the location of rivers from file rs.asc
rsmat = np.loadtxt(r'p:\11208012-011-nabaripoma\Model\Mesa\NaBaRiPoMa01\rs.asc', delimiter = ' ', skiprows = 6)
#plot
plt.matshow(rsmat)
plt.title('location of rivers')
plt.colorbar()
plt.show()

# read the location of polders from file polder.asc
polmat = np.loadtxt(r'p:\11208012-011-nabaripoma\Model\Mesa\NaBaRiPoMa01\Polders.asc', delimiter = ' ', skiprows = 6)
#plot
plt.matshow(polmat)
plt.title('location of polders')
plt.colorbar()
plt.show()

#initial pcraster calculation
#set clonemap
pcr.setclone(height, width, 1, 0, 0)
#set floodelevmat with polders 10m higher
floodelevmat = np.where(polmat > 0, elevmat + 10., elevmat)
#plot
plt.matshow(floodelevmat)
plt.title('flood elevation')
plt.colorbar()
plt.show()

#convert numpay arrays to PCR rasters
pcrelev = pcr.numpy2pcr(pcr.Scalar, elevmat, -999.)
pcr.report(pcrelev, r'p:\11208012-011-nabaripoma\Model\Python\results\maps\elevation.map')
pcrfloodelev = pcr.numpy2pcr(pcr.Scalar, elevmat, -999.)
pcr.report(pcrfloodelev, r'p:\11208012-011-nabaripoma\Model\Python\results\maps\floodelevation.map')
#create ldd
pcr.setglobaloption('lddin')
pcr.setglobaloption('unittrue')
pcrldd = pcr.lddcreate(pcrelev,9999999,9999999,9999999,9999999)
pcr.report(pcrldd, r'p:\11208012-011-nabaripoma\Model\Python\results\maps\ldd.map')
lddmat = pcr.pcr2numpy(pcrldd,-999)
#plot
plt.matshow(lddmat)
plt.title('ldd')
plt.colorbar()
plt.show()

#create river and sea array
rivmat = np.where(rsmat == 2, 1, 0)
#plot
plt.matshow(rivmat)
plt.title('river')
plt.colorbar()
plt.show()

seamat = np.where(rsmat == 1, 1, 0)
#plot
plt.matshow(seamat)
plt.title('sea')
plt.colorbar()
plt.show()

#convert river and sea array to map
pcrriv = pcr.numpy2pcr(pcr.Boolean, rivmat, -999.)
pcrsea = pcr.numpy2pcr(pcr.Boolean, seamat, -999.)
#calculate distance to river over ldd
pcrdist2riv = pcr.ldddist(pcrldd,pcrriv,1.)
dist2rivmat = pcr.pcr2numpy(pcrdist2riv,-999)
#plot
plt.matshow(dist2rivmat, vmin=0, vmax=20)
plt.title('dist2riv_ldd')
plt.colorbar()
plt.show()

pcr.report(pcrdist2riv,r'p:\11208012-011-nabaripoma\Model\Python\results\maps\dist2riv.map')

#calculate distance to sea over ldd
pcrdist2sea = pcr.ldddist(pcrldd,pcrsea,1.)
dist2seamat = pcr.pcr2numpy(pcrdist2sea,-999)
#plot
plt.matshow(dist2seamat)
plt.title('dist2sea_ldd')
plt.colorbar()
plt.show()

pcr.report(pcrdist2sea,r'p:\11208012-011-nabaripoma\Model\Python\results\maps\dist2sea.map')

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

    #calculate distance to river over ldd
    pcrdist2riv = pcr.ldddist(pcrldd,pcrriv,1.)
    dist2rivmat = pcr.pcr2numpy(pcrdist2riv,-999)
       
    #calculate distance to sea over ldd
    pcrdist2sea = pcr.ldddist(pcrldd,pcrsea,1.)
    dist2seamat = pcr.pcr2numpy(pcrdist2sea,-999)
    
    #Physics calculation
    is_waterlogged_dry = np.full(np.shape(elevmat),False)    
    is_waterlogged_wet = np.full(np.shape(elevmat),False)  
    
    is_trm = np.full(np.shape(elevmat),False)
    # if year in [2031, 2032]:
    #     is_trm = polmat == 3

    is_sea = rsmat==1
    is_river = rsmat == 2
    is_polder = polmat > 0
    is_nopolder = polmat == 0
    is_land = rsmat == 0

    # update topography
    # soil subsidence
    elevmat[is_land] = elevmat[is_land] - subsidence * 0.01
    #sedimentation on land outside polders
    elevmat[(is_land & is_nopolder) | is_river] = elevmat[(is_land & is_nopolder) | is_river] + sedrate * 0.01
    #sedimentation in trm areas
    elevmat[is_trm] = elevmat[is_trm] + trmsedrate * 0.01
    
    #tidal range - for the moment a linear decrease with 2cm per km from 2m at sea
    #tidalrange = 2. - 0.02 * dist2seamat
    tidalrange = np.full(np.shape(elevmat),2.5) #fixed tidal range  
    tidalrange[tidalrange < 0.]= 0.
    # #plot
    # plt.matshow(tidalrange)
    # plt.title('tidalrange')
    # plt.colorbar()
    # plt.show()
    
    #Dry season water level
    wl_dry=0.1
    #wer season water level
    wl_wet=0.3

    # #flood depth - high tide minus elevation
    flooddepth=np.zeros(np.shape(elevmat))
    flooddepth[((is_nopolder) | (is_trm) | (is_river))] = msl + tidalrange[((is_nopolder) | (is_trm) | (is_river))] * 0.5 - elevmat[((is_nopolder) | (is_trm) | (is_river))]
    flooddepth[flooddepth < 0.] = 0.
    # #plot
    # plt.matshow(flooddepth)
    # plt.title('flooddepth')
    # plt.colorbar()
    # plt.show()
        
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
    gradient_dry=np.full(np.shape(elevmat),-999.0)
    gradient_wet=np.full(np.shape(elevmat),-999.0)
    gradient_dry[is_land] = (elevmat[is_land] - (msl + wl_dry - 0.5 * tidalrange[is_land] )) / dist2rivmat[is_land]
    gradient_wet[is_land] = (elevmat[is_land] - (msl + wl_wet - 0.5 * tidalrange[is_land] )) / dist2rivmat[is_land]
    
    is_waterlogged_dry[(is_land) & (gradient_dry < mindraingrad)] = True    
    is_waterlogged_wet[(is_land) & (gradient_wet < mindraingrad)] = True  
    
    waterlogged_sev_dry = 1 - (gradient_dry / mindraingrad)
    waterlogged_sev_dry[waterlogged_sev_dry < 0.] = 0.
    waterlogged_sev_dry[waterlogged_sev_dry > 1.] = 1.
    
    waterlogged_sev_wet = 1 - (gradient_wet / mindraingrad)
    waterlogged_sev_wet[waterlogged_sev_wet < 0.] = 0.
    waterlogged_sev_wet[waterlogged_sev_wet > 1.] = 1.


    #plot
    plt.rcParams["figure.figsize"] = [10, 10]
    f, (ax1, ax2, ax3) = plt.subplots(1,3, sharex=True, sharey=True)
    f1=ax1.matshow(elevmat, vmin=-5, vmax = 3)
    ax1.set_title('elevation')
    plt.colorbar(f1,ax=ax1)
    f2=ax2.matshow(waterlogged_sev_dry, vmin=0, vmax = 1)
    ax2.set_title('waterlogged_sev_dry')
    plt.colorbar(f2,ax=ax2)
    f3=ax3.matshow(waterlogged_sev_wet, vmin=0, vmax = 1)
    ax3.set_title('waterlogged_sev_wet')
    plt.colorbar(f3,ax=ax3)
    f.suptitle(year, fontsize=16, x=0.6)
    plt.tight_layout()
    plt.savefig(r'p:\11208012-011-nabaripoma\Model\Python\results\waterlogging\waterlogging_' + str(year) + '.png', format='png', bbox_inches='tight', dpi=300)
    plt.show()

    #river flow
    
    #river bed --> update elevation
    
    #river salt - later, for now fixed