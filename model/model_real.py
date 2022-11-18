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
import xarray as xr

model_params = {
    "slr2100": 1, #UserSettableParameter
    "subsidence": 2, #UserSettableParameter
    "sedrate": 10, #UserSettableParameter
    "trmsedrate": 40, #UserSettableParameter
}

#%%INITIALIZE
#Model parameters
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

#Read grid maps   
# Read the elevation (topography/bathymetry)
elev_raster = xr.open_rasterio(r'p:\11208012-011-nabaripoma\Data\elevation.tif')
elevmat = elev_raster.to_numpy().squeeze()*0.01
#plot
plt.matshow(elevmat)
plt.title('elevation')
plt.colorbar()
plt.show()

# read the location of rivers and sea
rs_raster = xr.open_rasterio(r'p:\11208012-011-nabaripoma\Data\rivers.tif')
rsmat = rs_raster.to_numpy().squeeze()
rsmat[-5:,:]=2
#plot
plt.matshow(rsmat)
plt.title('location of rivers and sea')
plt.colorbar()
plt.show()

# read the location of polders
pol_raster = xr.open_rasterio(r'p:\11208012-011-nabaripoma\Data\polders.tif')
polmat = pol_raster.to_numpy().squeeze()
#plot
plt.matshow(polmat)
plt.title('location of polders')
plt.colorbar()
plt.show()

# read households per ha (per gridcell)
hh_raster = xr.open_rasterio(r'p:\11208012-011-nabaripoma\Data\hh_perha.tif')
hhmat = hh_raster.to_numpy().squeeze()
#plot
plt.matshow(hhmat)
plt.title('households per ha')
plt.colorbar()
plt.show()


#initial pcraster calculation
#set clonemap
height = np.shape(elevmat)[0]
width = np.shape(elevmat)[1]
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
pcr.report(pcrelev, r'p:\11208012-011-nabaripoma\Model\Python\results\real\maps\elevation.map')
pcrfloodelev = pcr.numpy2pcr(pcr.Scalar, elevmat, -999.)
pcr.report(pcrfloodelev, r'p:\11208012-011-nabaripoma\Model\Python\results\real\maps\floodelevation.map')
#create ldd
pcr.setglobaloption('lddin')
pcr.setglobaloption('unittrue')
pcrldd = pcr.lddcreate(pcrelev,9999999,9999999,9999999,9999999)
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

seamat = np.where(rsmat == 2, 1, 0)
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
plt.matshow(dist2rivmat)
plt.title('dist2riv_ldd')
plt.colorbar()
plt.show()
pcr.report(pcrdist2riv,r'p:\11208012-011-nabaripoma\Model\Python\results\real\maps\dist2riv.map')

#calculate distance to sea over ldd
pcrdist2sea = pcr.ldddist(pcrldd,pcrsea,1.)
dist2seamat = pcr.pcr2numpy(pcrdist2sea,-999)
#plot
plt.matshow(dist2seamat)
plt.title('dist2sea_ldd')
plt.colorbar()
plt.show()
pcr.report(pcrdist2sea,r'p:\11208012-011-nabaripoma\Model\Python\results\real\maps\dist2sea.map')

#Masks
is_sea = rsmat==1
is_river = rsmat == 2
is_polder = polmat > 0
is_nopolder = polmat == 0
is_land = rsmat == 0

#%%Agents = rural households
#Create a list of households with attributes 

class hh_agents:
    """
    Agent to describe the rural households
    
    outputs:
        Total Production costs
        Farm income
        Total income
        Migration
        Food security

    """

    def __init__(self, wlog_sev):
        #Agent attributes
        self.farmsize_small = 0.51
        self.farmsize_med = 2.02
        self.farmsize_large = 6.07
        
        self.householdsize_small = 4.15
        self.householdsize_med = 4.15
        self.householdsize_large = 4.15
        
        self.leasedarea_small = 0.1
        self.leasedarea_med = 0.3
        self.leasedarea_large = 0.4        
        
        #Farm production
        self.farmprod_rice = 3.74 #ton/hectare 
        self.farmprod_fish = 1.96
        self.farmprod_shrimp = 0.33 
        
        #Farm employment
        self.farmempl_totperm_rice = 15.0 # #/hectare
        self.farmempl_totperm_rice_fish = 25.0
        self.farmempl_totperm_fish = 25.0
        self.farmempl_totperm_shrimp = 10.0
        
        self.farmempl_hirperm_rice_small = 0.05 # #/hectare
        self.farmempl_hirperm_rice_med = 1.25
        self.farmempl_hirperm_rice_large = 2.0
        self.farmempl_hirperm_fish_small = 0.15 
        self.farmempl_hirperm_fish_med = 2.0
        self.farmempl_hirperm_fish_large = 3.0
        self.farmempl_hirperm_shrimp_small = 0.045 
        self.farmempl_hirperm_shrimp_med = 1.0
        self.farmempl_hirperm_shrimp_large = 1.5
        
        #Others
        self.migr_income = 500. #BDT/day
        self.land_lease = 8090. #BDT/hectare/year
        self.var_prod_costs = 4357. #BDT/hectare
        self.human_lab = 6840. #BDT/hectare
        self.irrigation = 1523.
        self.rice_cons = 181. #kg/person/year2021
        self.fish_cons = 23.
        self.shrimp_cons = 23.

        #Prices
        self.price_freshw_fish = 130. #Taka/kg 2019 prices
        self.price_freshw_shrimp = 750. 
        self.price_saltw_shrimp = 675. 
        self.price_saltw_fish = 417.5
        self.price_HYV_Boro = 20.8 


        #Agent functions
        
        #Farm production
        #Rice
        self.farm_prod_rice = (1-wlog_sev)*self.farmprod_rice #ton/hectare
        #Fish
        if wlog_sev > 0.8:
            self.farm_prod_fish = (self.farmprod_fish*((1-wlog_sev)+0.6))
        else:
            self.farm_prod_fish = self.farmprod_fish
        #Shrimp
        if wlog_sev > 0.8:
            self.farm_prod_shrimp = (self.farmprod_shrimp*((1-wlog_sev)+0.6))
        else:
            self.farm_prod_shrimp = self.farmprod_shrimp
            
        #Farm production per household category
        self.farm_prod_rice_small = self.farm_prod_rice * self.farmsize_small
        self.farm_prod_rice_med = self.farm_prod_rice * self.farmsize_med
        self.farm_prod_rice_large = self.farm_prod_rice * self.farmsize_large
        
        self.farm_prod_fish_small = self.farm_prod_fish * self.farmsize_small
        self.farm_prod_fish_med = self.farm_prod_fish * self.farmsize_med
        self.farm_prod_fish_large = self.farm_prod_fish * self.farmsize_large
        
        self.farm_prod_shrimp_small = self.farm_prod_shrimp * self.farmsize_small
        self.farm_prod_shrimp_med = self.farm_prod_shrimp * self.farmsize_med
        self.farm_prod_shrimp_large = self.farm_prod_shrimp * self.farmsize_large

        #Subsistence consumption
        self.subs_comp_rice_small = self.householdsize_small * self.rice_cons
        self.subs_comp_rice_med = self.householdsize_med * self.rice_cons
        self.subs_comp_rice_large = self.householdsize_med * self.rice_cons
        
        self.subs_comp_fish_small = self.householdsize_small * self.fish_cons
        self.subs_comp_fish_med = self.householdsize_med * self.fish_cons
        self.subs_comp_fish_large = self.householdsize_large * self.fish_cons
        
        self.subs_comp_shrimp_small = self.householdsize_small * self.shrimp_cons
        self.subs_comp_shrimp_med = self.householdsize_med * self.shrimp_cons
        self.subs_comp_shrimp_large = self.householdsize_large * self.shrimp_cons
        
        #Farm production for market
        if self.farm_prod_rice_small - (self.subs_comp_rice_small/1000.0) < 0:
            self.farm_prod_market_rice_small = 0
        else:
            self.farm_prod_market_rice_small = self.farm_prod_rice_small - (self.subs_comp_rice_small/1000.0)
            
        self.farm_prod_market_rice_med = self.farm_prod_rice_med - (self.subs_comp_rice_med/1000.0)
        self.farm_prod_market_rice_large = self.farm_prod_rice_large - (self.subs_comp_rice_large/1000.0)
        
        self.farm_prod_market_fish_small = self.farm_prod_fish_small - (self.subs_comp_fish_small/1000.0)
        self.farm_prod_market_fish_med = self.farm_prod_fish_med - (self.subs_comp_fish_med/1000.0)
        self.farm_prod_market_fish_large = self.farm_prod_fish_large - (self.subs_comp_fish_large/1000.0)
        
        self.farm_prod_market_shrimp_small = self.farm_prod_shrimp_small - (self.subs_comp_shrimp_small/1000.0)
        self.farm_prod_market_shrimp_med = self.farm_prod_shrimp_med - (self.subs_comp_shrimp_med/1000.0)
        self.farm_prod_market_shrimp_large = self.farm_prod_shrimp_large - (self.subs_comp_shrimp_large/1000.0)     
        
        #Farm gross income
        self.farm_gross_income_rice_small = self.farm_prod_market_rice_small * 1000.0 * self.price_HYV_Boro 
        self.farm_gross_income_rice_med = self.farm_prod_market_rice_med * 1000.0 * self.price_HYV_Boro 
        self.farm_gross_income_rice_large = self.farm_prod_market_rice_large * 1000.0 * self.price_HYV_Boro 
        
        self.farm_gross_income_fish_small = self.farm_prod_market_fish_small * 1000.0 * self.price_freshw_fish 
        self.farm_gross_income_fish_med = self.farm_prod_market_fish_med * 1000.0 * self.price_freshw_fish
        self.farm_gross_income_fish_large = self.farm_prod_market_fish_large * 1000.0 * self.price_freshw_fish
        
        self.farm_gross_income_shrimp_small = self.farm_prod_market_shrimp_small * 1000.0 * self.price_saltw_shrimp 
        self.farm_gross_income_shrimp_med = self.farm_prod_market_shrimp_med * 1000.0 * self.price_saltw_shrimp
        self.farm_gross_income_shrimp_large = self.farm_prod_market_shrimp_large * 1000.0 * self.price_saltw_shrimp
        
        self.farm_gross_income_rice = {
            "small": self.farm_gross_income_rice_small, 
            "med": self.farm_gross_income_rice_med, 
            "large": self.farm_gross_income_rice_large,
        }       

        self.farm_gross_income_fish = {
            "small": self.farm_gross_income_fish_small, 
            "med": self.farm_gross_income_fish_med, 
            "large": self.farm_gross_income_fish_large,
        }  

        self.farm_gross_income_shrimp = {
            "small": self.farm_gross_income_shrimp_small, 
            "med": self.farm_gross_income_shrimp_med, 
            "large": self.farm_gross_income_shrimp_large,
        }  
        
        self.farm_gross_income = {
            "rice": self.farm_gross_income_rice, 
            "fish": self.farm_gross_income_fish, 
            "shrimp": self.farm_gross_income_shrimp,
        }
        
        #Farm employment (total permanent)
        self.farm_empl_tot_perm_rice_small = self.farmsize_small * self.farmempl_totperm_rice
        self.farm_empl_tot_perm_rice_med = self.farmsize_med * self.farmempl_totperm_rice
        self.farm_empl_tot_perm_rice_large = self.farmsize_large * self.farmempl_totperm_rice
        
        self.farm_empl_tot_perm_fish_small = self.farmsize_small * self.farmempl_totperm_fish
        self.farm_empl_tot_perm_fish_med = self.farmsize_med * self.farmempl_totperm_fish
        self.farm_empl_tot_perm_fish_large = self.farmsize_large * self.farmempl_totperm_fish
        
        self.farm_empl_tot_perm_shrimp_small = self.farmsize_small * self.farmempl_totperm_shrimp
        self. farm_empl_tot_perm_shrimp_med = self.farmsize_med * self.farmempl_totperm_shrimp
        self. farm_empl_tot_perm_shrimp_large = self.farmsize_large * self.farmempl_totperm_shrimp
        
        #Farm employment (hired permanent)
        self.farm_empl_hir_perm_rice_small = self.farmsize_small * self.farmempl_hirperm_rice_small
        self.farm_empl_hir_perm_rice_med = self.farmsize_med * self.farmempl_hirperm_rice_med
        self.farm_empl_hir_perm_rice_large = self.farmsize_large * self.farmempl_hirperm_rice_large
        
        self.farm_empl_hir_perm_fish_small = self.farmsize_small * self.farmempl_hirperm_fish_small
        self.farm_empl_hir_perm_fish_med = self.farmsize_med * self.farmempl_hirperm_fish_med
        self.farm_empl_hir_perm_fish_large = self.farmsize_large * self.farmempl_hirperm_fish_large
        
        self.farm_empl_hir_perm_shrimp_small = self.farmsize_small * self.farmempl_hirperm_shrimp_small
        self. farm_empl_hir_perm_shrimp_med = self.farmsize_med * self.farmempl_hirperm_shrimp_med
        self.farm_empl_hir_perm_shrimp_large = self.farmsize_large * self.farmempl_hirperm_shrimp_large      
        
        #Production cost
        self.rice_irr_small = self.farmsize_small * (self.var_prod_costs + self.human_lab + self.irrigation)
        self.rice_irr_med = self.farmsize_med * (self.var_prod_costs + self.human_lab + self.irrigation)        
        self.rice_irr_large = self.farmsize_large * (self.var_prod_costs + self.human_lab + self.irrigation)        
        
        self.rice_noirr_small = self.farmsize_small * (self.var_prod_costs + self.human_lab)
        self.rice_noirr_med = self.farmsize_med * (self.var_prod_costs + self.human_lab)
        self.rice_noirr_large = self.farmsize_large * (self.var_prod_costs + self.human_lab)
        
        self.rice_irr_landlease_small = self.farmsize_small * (self.var_prod_costs + self.human_lab + self.irrigation) + (self.leasedarea_small * self.land_lease) 
        self.rice_irr_landlease_med = self.farmsize_med * (self.var_prod_costs + self.human_lab + self.irrigation) + (self.leasedarea_med * self.land_lease) 
        self.rice_irr_landlease_large = self.farmsize_large * (self.var_prod_costs + self.human_lab + self.irrigation) + (self.leasedarea_large * self.land_lease) 
        
        self.rice_noirr_landlease_small = self.farmsize_small * (self.var_prod_costs + self.human_lab) + (self.leasedarea_small * self.land_lease) 
        self.rice_noirr_landlease_med = self.farmsize_med * (self.var_prod_costs + self.human_lab) + (self.leasedarea_med * self.land_lease) 
        self.rice_noirr_landlease_large = self.farmsize_large * (self.var_prod_costs + self.human_lab) + (self.leasedarea_large * self.land_lease)       


#%%RUN CALCULATION (Loop from 2022 to 2100)
for year in np.arange(startyear, endyear+1,1):
    """
    Run one iteration of the model. 
    """
    print(year)
    
    #BIOPHYSICAL
    
    # update topography
    # soil subsidence
    elevmat[is_land] = elevmat[is_land] - subsidence * 0.01
    #sedimentation on land outside polders
    elevmat[(is_land & is_nopolder) | is_river] = elevmat[(is_land & is_nopolder) | is_river] + sedrate * 0.01
    
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
    #lt_dry= max((msl + wl_dry - 0.5 * tidalrange[is_land]), elevmat[is_river])
    
    #Upstream flow - wet season water level
    wl_wet=0.3
    #Wet low tide level

    #TRM
    
    is_trm = np.full(np.shape(elevmat),False)
    # if year in [2031, 2032]:
    #     is_trm = polmat == 3

    #sedimentation in trm areas
    elevmat[is_trm] = elevmat[is_trm] + trmsedrate * 0.01

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
    #     usdraina = 0.6
    #     prism = 0.
    #     y = cell[2]
    #     upagent = cell[0]
    #     if (y > rivy) and ((upagent.is_nopolder) or (upagent.is_trm) or (upagent.is_river)):
    #         usdraina += 1
    #         prism += upagent.flooddepth
    
        
        #polder drainage - perhaps later, for now zero
        
    #water logging - patches with gradient less than drainhead to low tide
    gradient_dry=np.full(np.shape(elevmat),-999.0)
    gradient_wet=np.full(np.shape(elevmat),-999.0)
    gradient_dry[is_land] = (elevmat[is_land] - (msl + wl_dry - 0.5 * tidalrange[is_land] )) / dist2rivmat[is_land]
    gradient_wet[is_land] = (elevmat[is_land] - (msl + wl_wet - 0.5 * tidalrange[is_land] )) / dist2rivmat[is_land]
       
    #Waterlogging
    is_waterlogged_dry = np.full(np.shape(elevmat),False)    
    is_waterlogged_wet = np.full(np.shape(elevmat),False) 
    is_waterlogged_dry[(is_land) & (gradient_dry < mindraingrad)] = True    
    is_waterlogged_wet[(is_land) & (gradient_wet < mindraingrad)] = True  
    
    waterlogged_sev_dry = 1 - (gradient_dry / mindraingrad)
    waterlogged_sev_dry[waterlogged_sev_dry < 0.] = 0.
    waterlogged_sev_dry[waterlogged_sev_dry > 1.] = 1.
    
    waterlogged_sev_wet = 1 - (gradient_wet / mindraingrad)
    waterlogged_sev_wet[waterlogged_sev_wet < 0.] = 0.
    waterlogged_sev_wet[waterlogged_sev_wet > 1.] = 1.


    #plot
    plt.rcParams["figure.figsize"] = [20, 20]
    f, (ax1, ax2, ax3) = plt.subplots(1,3, sharex=True, sharey=True)
    f1=ax1.matshow(elevmat, vmin=-1, vmax = 1)
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
    plt.savefig(r'p:\11208012-011-nabaripoma\Model\Python\results\real\waterlogging\waterlogging_' + str(year) + '.png', format='png', bbox_inches='tight', dpi=300)
    plt.show()

    #river flow
    
    #river bed --> update elevation
    
    #river salt - later, for now fixed
    
    #init arrays
    farm_gross_income_rice_small = np.zeros(np.shape(elevmat))
    
    #SOCIO-ECONOMICS
    #Calculate income, food security and migration with wet and dry season water logging severity as input
    for x in np.arange(0, np.shape(elevmat)[0]):
        for y in np.arange(0, np.shape(elevmat)[1]):
                socio=hh_agents(waterlogged_sev_dry)
                farm_gross_income_rice_small[x,y]=socio.farm_gross_income['rice']['small']
                
    #plot
    plt.rcParams["figure.figsize"] = [20, 20]
    plt.matshow(farm_gross_income_rice_small)
    plt.title('Gross income for rice in small farms')
    plt.colorbar()
    plt.show()
    f.suptitle(year, fontsize=16, x=0.5)
    plt.tight_layout()
    plt.savefig(r'p:\11208012-011-nabaripoma\Model\Python\results\real\gross_income\gross_income_rice_' + str(year) + '.png', format='png', bbox_inches='tight', dpi=300)
    plt.show()