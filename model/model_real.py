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
import rasterio
import pandas as pd

model_params = {
    "slr2100": 1, #UserSettableParameter
    "subsidence": 2, #UserSettableParameter
    "sedrate": 10, #UserSettableParameter
    "trmsedrate": 40, #UserSettableParameter
}

#Options
plot = False
raster = True

#%%INITIALIZE
#Model parameters
slr2100 = model_params['slr2100']
subsidence = model_params['subsidence']
sedrate = model_params['sedrate']
trmsedrate = model_params['trmsedrate']
mslstart = 0.00
startyear = 2022
endyear = 2030
kslr = 0.02
mindraingrad = 0.1 / 1000. # 10cm per km minimum drainage gradient
year = startyear
msl = 0.00

#Read grid maps   
# Read the elevation (topography/bathymetry)
elev_raster = xr.open_rasterio(r'p:\11208012-011-nabaripoma\Data\elevation.tif')
elevmat = elev_raster.to_numpy().squeeze()
elevmat[elevmat<-52.0] = -52.0
elevmat[elevmat==0] = -57.0
elevmat = elevmat + 52.0
#plot
plt.matshow(elevmat)
plt.title('elevation')
plt.colorbar()
plt.show()

#Read rater metadata
with rasterio.open(r'p:\11208012-011-nabaripoma\Data\elevation.tif') as src:
    ras_meta = src.profile

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

#Numebr of polders
no_polder=np.max(polmat)

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
	Total Production costs		Distribution produciton costs per land size category
	Farm income		            Distribution farm income per land size category
	Total income		        Distribution total income per land size category
	Poverty		                % of households per group with income under poverty line
	Migration		            # migrated people
	Food security		        % of households per group with food security
	Total employment		    #permanent jobs
			                    #seasonal jobs


    """

    def __init__(self, wlog_sev):
        #Agent attributes
        self.farmsize = {
        "small": 0.51,
        "med": 2.02,
        "large": 6.07
        }
        
        self.tot_pop_agr = {
        "small": 47.0,
        "med": 8.,
        "large": 1.,
        "landless": 44.
        }
        
        self.householdsize = {
        "small": 4.15,
        "med": 4.15,
        "large": 4.15
        }
        
        self.leasedarea = {
        "small": 0.1,
        "med": 0.3,
        "large": 0.4
        }
             
        self.croppping_pattern = {
            {
            "rice": 
                {
                "small": 70.,
                "med": 50.,
                "large": 30.
                },
            "rice-fish":
                {
                "small": 20.,
                "med": 20.,
                "large": 20.,
                },
            "fish":
                {
                "small": 10.,
                "med": 25.,
                "large": 20.,
                },
            "shrimp":
                {
                "small": 0.,
                "med": 5.,
                "large": 30.
                }
            }
        }

            #Household income additional activities
        self.hh_income_additional = {
        "small": 95053.0,
        "med": 95053.0,
        "large": 95053.0
        }        

        #Migrated household members (initial set-up)            
        self.migrated_hh_members = {
        "small": 0.3,
        "med": 0.3,
        "large": 0.3
        }   
        
        #Farm production
        self.farmprod = {
        "rice": 3.74,
        "fish": 1.96,
        "shrimp": 0.33
        }
                
        #Farm employment
        self.farmempl = {
        "family_perm": 
            {
            "rice": 
                {
                "small": 1.2,
                "med": 1.8,
                "large": 2.2
                },
            "rice-fish": 
                {
                "small": 1.2,
                "med": 1.8,
                "large": 2.2
                },
            "fish":
                {
                "small": 1.2,
                "med": 1.8,
                "large": 2.2,
                },
            "shrimp":
                {
                "small": 1.5,
                "med": 2.8,
                "large": 4.9
                }
            },
        "hired_perm": 
            {
            "rice": 
                {
                "small": 0.1,
                "med": 0.47,
                "large": 2.01
                },
            "rice-fish": 
                {
                "small": 0.12,
                "med": 0.56,
                "large": 2.49
                },
            "fish":
                {
                "small": 0.1,
                "med": 0.47,
                "large": 2.01,
                },
            "shrimp":
                {
                "small": 0.25,
                "med": 0.97,
                "large": 2.91
                }
            },
        "hired_temp":
            {
            "rice": 
                {
                "small": 16.8,
                "med": 66.1,
                "large": 198.5
                },
            "rice-fish": 
                {
                "small": 17.8,
                "med": 69.9,
                "large": 210.0
                },
            "fish":
                {
                "small": 16.8,
                "med": 66.1,
                "large": 198.5,
                },
            "shrimp":
                {
                "small": 20.6,
                "med": 80.9,
                "large": 242.8
                }
            }
        }
    
        #Temporary employment
        self.farmprod = {
        "small": 9.0,
        "med": 9.0,
        "large": 9.0
        }
            
        #Migrant income
        self.migr_income = 500. #BDT/day
        
        #Land lease
        self.land_lease = 8090. #BDT/hectare/year
        
        #Irrigation % of farms
        self.irrigation_perc = {
        "small": 67.0,
        "med": 74.6,
        "large": 70.30
        }
        
        #Variable production costs (- irrigation and human labour) (BDT/hectare)   
        self.var_prod_costs = {  
        "rice": 
            {
            "small": 4357.0,
            "med": 7388.0,
            "large": 11971.0
            },
        "fish": 37920,
        "shrimp": 46808
        }
        
        #Human labour (BDT/hectare)
        self.human_lab = {
        "small": 6840.0,
        "med": 8196,
        "large": 12001
        }
        
        #Irrigation
        self.irrigation = {
        "small": 1523.0,
        "med": 2105,
        "large": 2346
        }
        
        #Rice consumption (kg/person/year2021)
        self.consumption = {
        "rice": 181.,
        "fish": 23.,
        "shrimp": 23.
        }

        #Crop intensity
        self.cropping_intensity = {
        "small": 180.0,
        "med": 166,
        "large": 155
        }        

        #Transition costs rice-fish
        self.cost_trans_rice_fish = 9480.
        
        #Migrant income send home
        self.migr_income_send_home = 15.

        #Poverty line
        self.poverty_line = 192.

        #Days seasonal employment landless
        self.days_seas_emp_landless = 54.
        
        #People working in landless housseholds
        self.peop_work_landless = 50.

        #Prices farm gate
        self.price_farm_gate = {
        "freshw_fish" : 130., #Taka/kg 2019 prices
        "freshw_shrimp" : 750. ,
        "saltw_shrimp" : 675. ,
        "saltw_fish" : 417.5,
        "HYV_Boro" : 20.8 
        }
        
        #Selling price
        self.selling_price = {
        "rice": 65.0,
        "fish": 450.0,
        "shrimp": 1050.
        }

        #OUTPUTS
        #Farm production
        self.farm_prod = {
        "rice": 0,
        "fish": 0,
        "shrimp": 0
        }
        
        #Farm production per household category
        self.farm_prod_per_hh = {
            "rice": 
                {
                "small": 0,
                "med": 0,
                "large": 0
                },
            "fish":
                {
                "small": 0,
                "med": 0,
                "large": 0,
                },
            "shrimp":
                {
                "small": 0,
                "med": 0,
                "large": 0
                }
            }
        
        #Subsistence food consumption needed
        self.subs_food_cons = {
            "rice": 
                {
                "small": 0,
                "med": 0,
                "large": 0
                },
            "fish":
                {
                "small": 0,
                "med": 0,
                "large": 0,
                },
            "shrimp":
                {
                "small": 0,
                "med": 0,
                "large": 0
                }
            }
        
        #Farm production for market
        self.farm_prod_market = {
            "rice": 
                {
                "small": 0,
                "med": 0,
                "large": 0
                },
            "fish":
                {
                "small": 0,
                "med": 0,
                "large": 0,
                },
            "shrimp":
                {
                "small": 0,
                "med": 0,
                "large": 0
                }
            }
        
        #Farm production for food
        self.farm_prod_food = {
            "rice": 
                {
                "small": 0,
                "med": 0,
                "large": 0
                },
            "fish":
                {
                "small": 0,
                "med": 0,
                "large": 0,
                },
            "shrimp":
                {
                "small": 0,
                "med": 0,
                "large": 0
                }
            }
            
        #Farm gross income
        self.farm_gross_income = {
            "rice": 
                {
                "small": 0,
                "med": 0,
                "large": 0
                },
            "fish":
                {
                "small": 0,
                "med": 0,
                "large": 0,
                },
            "shrimp":
                {
                "small": 0,
                "med": 0,
                "large": 0
                }
            }
            
        #Production cost
        self.prod_cost = {
            "rice_irrig": 
                {
                "small": 0,
                "med": 0,
                "large": 0
                },
            "rice_no_irrig": 
                {
                "small": 0,
                "med": 0,
                "large": 0
                },
            "rice_irrig_landlease": 
                {
                "small": 0,
                "med": 0,
                "large": 0
                },
            "rice_no_irrig_landlease": 
                {
                "small": 0,
                "med": 0,
                "large": 0
                },
            "fish_landlease":
                {
                "small": 0,
                "med": 0,
                "large": 0,
                },
            "fish_no_landlease":
                {
                "small": 0,
                "med": 0,
                "large": 0,
                },
            "shrimp_landlease":
                {
                "small": 0,
                "med": 0,
                "large": 0,
                },
            "shrimp_no_landlease":
                {
                "small": 0,
                "med": 0,
                "large": 0,
                }
            }
            
        #Farm net income
        self.farm_net_income = {
            "rice_irrig": 
                {
                "small": 0,
                "med": 0,
                "large": 0
                },
            "rice_no_irrig": 
                {
                "small": 0,
                "med": 0,
                "large": 0
                },
            "rice_irrig_landlease": 
                {
                "small": 0,
                "med": 0,
                "large": 0
                },
            "rice_no_irrig_landlease": 
                {
                "small": 0,
                "med": 0,
                "large": 0
                },
            "fish_landlease":
                {
                "small": 0,
                "med": 0,
                "large": 0,
                },
            "fish_no_landlease":
                {
                "small": 0,
                "med": 0,
                "large": 0,
                },
            "shrimp_landlease":
                {
                "small": 0,
                "med": 0,
                "large": 0,
                },
            "shrimp_no_landlease":
                {
                "small": 0,
                "med": 0,
                "large": 0,
                },
            "fish-rice_landlease":
                {
                "small": 0,
                "med": 0,
                "large": 0,
                },
            "fish-rice_no_landlease":
                {
                "small": 0,
                "med": 0,
                "large": 0,
                }
            }

        #Total household income
        self.tot_hh_income = {
            "rice_irrig": 
                {
                "small": 0,
                "med": 0,
                "large": 0
                },
            "rice_no_irrig": 
                {
                "small": 0,
                "med": 0,
                "large": 0
                },
            "rice_irrig_landlease": 
                {
                "small": 0,
                "med": 0,
                "large": 0
                },
            "rice_no_irrig_landlease": 
                {
                "small": 0,
                "med": 0,
                "large": 0
                },
            "fish_landlease":
                {
                "small": 0,
                "med": 0,
                "large": 0,
                },
            "fish_no_landlease":
                {
                "small": 0,
                "med": 0,
                "large": 0,
                },
            "shrimp_landlease":
                {
                "small": 0,
                "med": 0,
                "large": 0,
                },
            "shrimp_no_landlease":
                {
                "small": 0,
                "med": 0,
                "large": 0,
                },
            "fish-rice_landlease":
                {
                "small": 0,
                "med": 0,
                "large": 0,
                },
            "fish-rice_no_landlease":
                {
                "small": 0,
                "med": 0,
                "large": 0,
                }
            }    

        #Income above poverty line (corrected for own food consumption)
        self.income_above_poverty = {
            "rice_irrig": 
                {
                "small": 0,
                "med": 0,
                "large": 0
                },
            "rice_no_irrig": 
                {
                "small": 0,
                "med": 0,
                "large": 0
                },
            "rice_irrig_landlease": 
                {
                "small": 0,
                "med": 0,
                "large": 0
                },
            "rice_no_irrig_landlease": 
                {
                "small": 0,
                "med": 0,
                "large": 0
                },
            "fish_landlease":
                {
                "small": 0,
                "med": 0,
                "large": 0,
                },
            "fish_no_landlease":
                {
                "small": 0,
                "med": 0,
                "large": 0,
                },
            "shrimp_landlease":
                {
                "small": 0,
                "med": 0,
                "large": 0,
                },
            "shrimp_no_landlease":
                {
                "small": 0,
                "med": 0,
                "large": 0,
                },
            "fish-rice_landlease":
                {
                "small": 0,
                "med": 0,
                "large": 0,
                },
            "fish-rice_no_landlease":
                {
                "small": 0,
                "med": 0,
                "large": 0,
                }
            }          
            
        #Food security
        self.food_security = {
            "rice_irrig": 
                {
                "small": 0,
                "med": 0,
                "large": 0
                },
            "rice_no_irrig": 
                {
                "small": 0,
                "med": 0,
                "large": 0
                },
            "rice_irrig_landlease": 
                {
                "small": 0,
                "med": 0,
                "large": 0
                },
            "rice_no_irrig_landlease": 
                {
                "small": 0,
                "med": 0,
                "large": 0
                },
            "fish_landlease":
                {
                "small": 0,
                "med": 0,
                "large": 0,
                },
            "fish_no_landlease":
                {
                "small": 0,
                "med": 0,
                "large": 0,
                },
            "shrimp_landlease":
                {
                "small": 0,
                "med": 0,
                "large": 0,
                },
            "shrimp_no_landlease":
                {
                "small": 0,
                "med": 0,
                "large": 0,
                },
            "fish-rice_landlease":
                {
                "small": 0,
                "med": 0,
                "large": 0,
                },
            "fish-rice_no_landlease":
                {
                "small": 0,
                "med": 0,
                "large": 0,
                }
            }   
         
        #Required hired permanent farm employment
        self.req_perm_farm_empl = {
            "rice": 
                {
                "small": 0,
                "med": 0,
                "large": 0
                },
            "fish":
                {
                "small": 0,
                "med": 0,
                "large": 0,
                },
            "shrimp":
                {
                "small": 0,
                "med": 0,
                "large": 0
                }
            }

        #Required hired seasonal farm employment
        self.req_seasonal_farm_empl = {
            "rice": 
                {
                "small": 0,
                "med": 0,
                "large": 0
                },
            "fish":
                {
                "small": 0,
                "med": 0,
                "large": 0,
                },
            "shrimp":
                {
                "small": 0,
                "med": 0,
                "large": 0
                }
            }

        #Landless farmers
        self.landless_farmer = {
            "hh_income": 
                {
                "perm_empl": 0,
                "seasonal_empl": 0,
                },
            "income_above_poverty":
                {
                "perm_empl": 0,
                "seasonal_empl": 0,
                },
            "food_security":
                {
                "perm_empl": 0,
                "seasonal_empl": 0,
                },
            "migration_fixed":
                {
                "perm_empl": 0,
                "seasonal_empl": 0,
                },
            }

        #Migration - family member
        self.migration_family = {
            "rice_irrig": 
                {
                "small": 0,
                "med": 0,
                "large": 0
                },
            "rice_no_irrig": 
                {
                "small": 0,
                "med": 0,
                "large": 0
                },
            "rice_irrig_landlease": 
                {
                "small": 0,
                "med": 0,
                "large": 0
                },
            "rice_no_irrig_landlease": 
                {
                "small": 0,
                "med": 0,
                "large": 0
                },
            "fish_landlease":
                {
                "small": 0,
                "med": 0,
                "large": 0,
                },
            "fish_no_landlease":
                {
                "small": 0,
                "med": 0,
                "large": 0,
                },
            "shrimp_landlease":
                {
                "small": 0,
                "med": 0,
                "large": 0,
                },
            "shrimp_no_landlease":
                {
                "small": 0,
                "med": 0,
                "large": 0,
                },
            "fish-rice_landlease":
                {
                "small": 0,
                "med": 0,
                "large": 0,
                },
            "fish-rice_no_landlease":
                {
                "small": 0,
                "med": 0,
                "large": 0,
                }
            } 

        def agent_functions(self, wlog_sev):
            '''
            #Agent functions
            '''
            #Farm production
            #Rice
            self.farm_prod.rice = (1-wlog_sev)*self.farmprod.rice #ton/hectare
            #Fish
            if wlog_sev > 0.8:
                self.farm_prod.fish = (self.farmprod.fish*((1-wlog_sev)+0.6))
            else:
                self.farm_prod.fish = self.farmprod.fish
            #Shrimp
            if wlog_sev > 0.8:
                self.farm_prod.shrimp = (self.farmprod.shrimp*((1-wlog_sev)+0.6))
            else:
                self.farm_prod.shrimp = self.farmprod.shrimp
                
            #Farm production per household category
            for hh in ['small', 'med', 'large']:
                for crop in ['rice', 'fish', 'shrimp']:
                    self.farm_prod_per_hh[crop][hh] = self.farm_prod[crop] * self.farmsize[hh]
                
            #Subsistence consumption
            for hh in ['small', 'med', 'large']:
                for crop in ['rice', 'fish', 'shrimp']:
                    self.subs_food_cons[crop][hh] = self.householdsize[hh] * self.consumption[crop]            
                        
            #Farm production for market           
            for hh in ['small', 'med', 'large']:
                for crop in ['rice', 'fish', 'shrimp']:
                    if hh == 'small' and crop == 'rice':
                        if self.farm_prod_market[crop][hh] - (self.subs_food_cons[crop][hh]/1000.0) < 0:
                            self.farm_prod_market[crop][hh] = 0
                    else:
                        self.farm_prod_market[crop][hh] = self.farm_prod_per_hh[crop][hh] - (self.subs_food_cons[crop][hh]/1000.0)
  
            #Farm production for food
            for hh in ['small', 'med', 'large']:
                for crop in ['rice', 'fish', 'shrimp', 'fish-rice']:
                    if crop == 'rice' or crop == 'fish' or crop == 'shrimp':
                        if self.subs_food_cons[crop][hh]/1000.0 > self.farm_prod_per_hh[crop][hh]:
                            self.farm_prod_food[crop][hh] = self.farm_prod_per_hh[crop][hh]
                        else:
                            self.farm_prod_food[crop][hh] = self.subs_food_cons[crop][hh]/1000.0    
                    elif crop == 'fish-rice':
                        if self.farm_prod[crop] * self.farmsize[hh] > self.subs_food_cons[crop][hh]/1000.0  :
                            self.farm_prod_food[crop][hh] = self.subs_food_cons[crop][hh]
                        else:
                            self.farm_prod_food[crop][hh] = self.farm_prod[crop] * self.farmsize[hh]
                    
            #Farm gross income
            for hh in ['small', 'med', 'large']:
                for crop in ['rice', 'fish', 'shrimp']:
                    if crop == 'rice':
                        self.farm_gross_income[crop][hh] = self.farm_prod_market[crop][hh] * 1000.0 * self.price_farm_gate["HYV_Boro"]
                    elif crop == 'fish': 
                        self.farm_gross_income[crop][hh] = self.farm_prod_market[crop][hh] * 1000.0 * self.price_farm_gate["freshw_fish"]
                    elif crop == 'shrimp':
                        self.farm_gross_income[crop][hh] = self.farm_prod_market[crop][hh] * 1000.0 * self.price_farm_gate["saltw_shrimp"]
                        
            #Production cost
            for hh in ['small', 'med', 'large']:
                for crop in ["rice_irrig", "rice_no_irrig", "rice_irrig_landlease", "rice_no_irrig_landlease", "fish_landlease", "fish_no_landlease", "shrimp_landlease", "shrimp_no_landlease"]:
                    if crop == 'rice_irrig':
                        self.prod_cost[crop][hh] = self.farmsize[hh] * (self.var_prod_costs['rice'][hh] + self.human_lab + self.irrigation)        
                    elif crop == "rice_no_irrig":
                        self.prod_cost[crop][hh] = self.farmsize[hh] * (self.var_prod_costs['rice'][hh] + self.human_lab)   
                    elif crop == "rice_irrig_landlease":
                        self.prod_cost[crop][hh] = self.farmsize[hh] * (self.var_prod_costs['rice'][hh] + self.human_lab + self.irrigation) + (self.leasedarea[hh] * self.land_lease) 
                    elif crop == "rice_no_irrig_landlease":
                        self.prod_cost[crop][hh] = self.farmsize[hh] * (self.var_prod_costs['rice'][hh] + self.human_lab) + (self.leasedarea[hh] * self.land_lease) 
                    elif crop == "fish_landlease":
                        self.prod_cost[crop][hh] = (self.farmsize[hh] * self.var_prod_costs['fish'][hh]) + (self.farmsize[hh] * self.land_lease) 
                    elif crop == "fish_no_landlease":
                        self.prod_cost[crop][hh] = (self.farmsize[hh] * self.var_prod_costs['fish'][hh])
                    elif crop == "shrimp_landlease":
                        self.prod_cost[crop][hh] = (self.farmsize[hh] * self.var_prod_costs['shrimp'][hh]) + (self.farmsize[hh] * self.land_lease) 
                    elif crop == "shrimp_no_landlease":
                        self.prod_cost[crop][hh] = (self.farmsize[hh] * self.var_prod_costs['shrimp'][hh])                        
 
            #Farm net income
            for hh in ['small', 'med', 'large']:
                for crop in ["rice_irrig", "rice_no_irrig", "rice_irrig_landlease", "rice_no_irrig_landlease", "fish_landlease", "fish_no_landlease", "shrimp_landlease", "shrimp_no_landlease", "fish-rice_landlease", "fish-rice_no_landlease"]:            
                        if crop == "fish-rice_landlease":
                            self.farm_net_income[crop][hh] = (self.farm_net_income["rice_no_irrig_landlease"][hh] + self.farm_net_income["fish_landlease"][hh])/2.0 - (self.cost_trans_rice_fish * self.land_lease) 
                        elif crop == "fish-rice_no_landlease":
                            self.farm_net_income[crop][hh] = (self.farm_net_income["rice_no_irrig"][hh] + self.farm_net_income["fish_no_landlease"][hh])/2.0 - (self.cost_trans_rice_fish * self.land_lease) 
                        else:
                            self.farm_net_income[crop][hh] = self.farm_gross_income[crop][hh] - self.prod_cost[crop][hh]
                            
            #Total household income
            for hh in ['small', 'med', 'large']:
                for crop in ["rice_irrig", "rice_no_irrig", "rice_irrig_landlease", "rice_no_irrig_landlease", "fish_landlease", "fish_no_landlease", "shrimp_landlease", "shrimp_no_landlease", "fish-rice_landlease", "fish-rice_no_landlease"]:            
                   self.tot_hh_income[crop][hh] = self.farm_net_income[crop][hh] + self.hh_income_additional[hh] + (self.migrated_hh_members[hh] * 50 * 6 * self.migr_income * self.migr_income_send_home)

            #Income above poverty line (corrected for own food consumption)
            for hh in ['small', 'med', 'large']:
                for crop in ["rice_irrig", "rice_no_irrig", "rice_irrig_landlease", "rice_no_irrig_landlease", "fish_landlease", "fish_no_landlease", "shrimp_landlease", "shrimp_no_landlease", "fish-rice_landlease", "fish-rice_no_landlease"]:             
                    if crop == "fish-rice_landlease" or crop == "fish-rice_no_landlease":
                        self.income_above_poverty[crop][hh] = self.tot_hh_income[crop][hh] > ((self.poverty_line * 365 * self.householdsize[hh]) - ((self.farm_prod_food['fish'][hh] * 1000 * self.selling_price['rice']) + (self.farm_prod_food['rice'][hh] * 1000 * self.selling_price['rice']))/2.0 )
                    else:
                        self.income_above_poverty[crop][hh] = self.tot_hh_income[crop][hh] > ((self.poverty_line * 365 * self.householdsize[hh]) - (self.farm_prod_food[crop][hh] * 1000 * self.selling_price[crop]))

            #Food security
            for hh in ['small', 'med', 'large']:
                for crop in ["rice_irrig", "rice_no_irrig", "rice_irrig_landlease", "rice_no_irrig_landlease", "fish_landlease", "fish_no_landlease", "shrimp_landlease", "shrimp_no_landlease", "fish-rice_landlease", "fish-rice_no_landlease"]:             
                    if crop == "rice_irrig" or crop == "rice_no_irrig" or crop == "rice_irrig_landlease" or crop == "rice_no_irrig_landlease":
                        if (self.farm_prod_food['rice'][hh] < self.subs_food_cons['rice'][hh] / 1000.0) and (self.income_above_poverty[crop][hh] == False):
                            self.food_security[crop][hh] = False
                        else:
                            self.food_security[crop][hh] = True
                    elif crop == "fish_landlease" or crop == "fish_no_landlease" or crop == "shrimp_landlease" or crop == "shrimp_no_landlease":     
                        if self.income_above_poverty[crop][hh] == False:
                            self.food_security[crop][hh] = False
                        else:
                            self.food_security[crop][hh] = True   
                    else:
                        if (self.farm_prod_food['fish-rice'][hh] < self.subs_food_cons['rice'][hh] / 1000.0) and (self.income_above_poverty[crop][hh] == False):
                            self.food_security[crop][hh] = False
                        else:
                            self.food_security[crop][hh] = True
            

#%%RUN CALCULATION (Loop from 2022 to 2100)
#initialize arrays and lists
indicators_av=np.zeros((3,(endyear-startyear)+1,1))
indicators_pol=np.zeros((3,(endyear-startyear)+1,no_polder))
df = pd.DataFrame(columns=['Year', 'Indicator', 'Polder','Value'])

#loop over timesteps
i=0
print('******** Simulation starts ********')
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
    
    #dry
    waterlogged_sev_dry = 1 - (gradient_dry / mindraingrad)
    waterlogged_sev_dry[waterlogged_sev_dry < 0.] = 0.
    waterlogged_sev_dry[waterlogged_sev_dry > 1.] = 1.
    
    #wet
    waterlogged_sev_wet = 1 - (gradient_wet / mindraingrad)
    waterlogged_sev_wet[waterlogged_sev_wet < 0.] = 0.
    waterlogged_sev_wet[waterlogged_sev_wet > 1.] = 1.

    #filename
    filename_waterlogging=r'p:\11208012-011-nabaripoma\Model\Python\results\real\waterlogging\waterlogging_' + str(year) + '.png'

    if plot:
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

    #river flow
    
    #river bed --> update elevation
    
    #river salt - later, for now fixed
    
    #init arrays
    farm_gross_income_rice_small = np.zeros(np.shape(elevmat))
    
    #SOCIO-ECONOMICS
    #Calculate income, food security and migration with wet and dry season water logging severity as input
    for x in np.arange(0, np.shape(elevmat)[0]):
        for y in np.arange(0, np.shape(elevmat)[1]):
                socio=hh_agents(waterlogged_sev_wet[x,y])
                farm_gross_income_rice_small[x,y]=socio.farm_gross_income['rice']['small'] #ind_id=0
    #update dataframe
    df = pd.concat([df.copy(),pd.DataFrame([{'Year':year, 'Indicator':'gross_income_rice_small', 'Polder':0, 'Value':np.mean(farm_gross_income_rice_small[polmat!=0])}])])
    for p in np.arange(1, no_polder+1):
        df = pd.concat([df.copy(),pd.DataFrame([{'Year':year, 'Indicator':'gross_income_rice_small', 'Polder':p, 'Value':np.mean(farm_gross_income_rice_small[polmat==p])}])])

    #filename
    filename_gross_income=r'p:\11208012-011-nabaripoma\Model\Python\results\real\gross_income\gross_income_rice_' + str(year) + '.png'
        
    if plot:
        #plot
        plt.rcParams["figure.figsize"] = [20, 20]
        plt.matshow(farm_gross_income_rice_small)
        plt.title('Gross income for rice in small farms')
        plt.colorbar()
        plt.suptitle(year, fontsize=16, x=0.5)
        plt.tight_layout()
        plt.savefig(filename_gross_income, format='png', bbox_inches='tight', dpi=300)
        plt.show()
        plt.close()
    
    if raster:
        if year == startyear or year == endyear: 
            #Write raster file (socio-econnomics)
            with rasterio.open(r'p:\11208012-011-nabaripoma\Model\Python\results\real\gross_income\geotif\gross_income_rice_' + str(year) + '.tif', 'w', **ras_meta) as dst:
                dst.write(farm_gross_income_rice_small, indexes=1) #gross income  
           
    #update loop
    i=i+1

#Save .csv
df.to_csv(r'p:\11208012-011-nabaripoma\Model\Python\results\real\csv\model_output.csv', index=False, float_format='%.2f')
