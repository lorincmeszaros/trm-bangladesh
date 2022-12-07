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
    "subsidence": 0.005, #UserSettableParameter
    "sedrate": 0.1, #UserSettableParameter
    "trmsedrate": 0.4, #UserSettableParameter
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
cellsize = 100 #100m
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
#rsmat[-5:,:]=2
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

#%%Agents = rural households
#Create a list of households with attributes 

def agent_functions(wlog_sev):

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

    #INIT
    #Agent attributes
    farmsize = {
    "small": 0.51,
    "med": 2.02,
    "large": 6.07
    }
    
    tot_pop_agr = {
    "small": 47.0/100.,
    "med": 8./100.,
    "large": 1./100.,
    "landless": 44./100.
    }
    
    householdsize = {
    "small": 4.15,
    "med": 4.15,
    "large": 4.15
    }
    
    leasedarea = {
    "small": 0.1,
    "med": 0.3,
    "large": 0.4
    }
         
    croppping_pattern = {
        "rice": 
            {
            "small": 70./100.,
            "med": 50./100.,
            "large": 30./100.
            },
        "rice-fish":
            {
            "small": 20./100.,
            "med": 20./100.,
            "large": 20./100.,
            },
        "fish":
            {
            "small": 10./100.,
            "med": 25./100.,
            "large": 20./100.,
            },
        "shrimp":
            {
            "small": 0./100.,
            "med": 5./100.,
            "large": 30./100.
            }
    }

    #Household income additional activities
    hh_income_additional = {
    "small": 95053.0,
    "med": 95053.0,
    "large": 95053.0
    }        

    #Migrated household members (initial set-up)            
    migrated_hh_members = {
    "small": 0.3,
    "med": 0.3,
    "large": 0.3
    }   
    
    #Farm production
    farmprod = {
    "rice": 3.74,
    "fish": 1.96,
    "shrimp": 0.33
    }
            
    #Farm employment
    farmempl = {
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
    temp_empl = {
    "small": 9.0,
    "med": 9.0,
    "large": 9.0
    }
        
    #Migrant income
    migr_income = 500. #BDT/day
    
    #Land lease
    land_lease = 8090. #BDT/hectare/year
    
    #Irrigation % of farms
    irrigation_perc = {
    "small": 67.0/100.,
    "med": 74.6/100.,
    "large": 70.30/100.
    }
    
    #Variable production costs (- irrigation and human labour) (BDT/hectare)   
    var_prod_costs = {  
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
    human_lab = {
    "small": 6840.0,
    "med": 8196,
    "large": 12001
    }
    
    #Irrigation
    irrigation = {
    "small": 1523.0,
    "med": 2105,
    "large": 2346
    }
    
    #Rice consumption (kg/person/year2021)
    consumption = {
    "rice": 181.,
    "fish": 23.,
    "shrimp": 23.
    }

    #Crop intensity
    cropping_intensity = {
    "small": 180.0/100.,
    "med": 166/100.,
    "large": 155/100.
    }        

    #Transition costs rice-fish
    cost_trans_rice_fish = 9480.
    
    #Migrant income send home
    migr_income_send_home = 15./100.

    #Poverty line
    poverty_line = 192.

    #Days seasonal employment landless
    days_seas_emp_landless = 54.
    
    #People working in landless housseholds
    peop_work_landless = 50./100.

    #Prices farm gate
    price_farm_gate = {
    "freshw_fish" : 130., #Taka/kg 2019 prices
    "freshw_shrimp" : 750. ,
    "saltw_shrimp" : 675. ,
    "saltw_fish" : 417.5,
    "HYV_Boro" : 20.8 
    }
    
    #Selling price
    selling_price = {
    "rice": 65.0,
    "fish": 450.0,
    "shrimp": 1050.
    }

    #OUTPUTS
    #Farm production
    farm_prod = {
    "rice": 0,
    "fish": 0,
    "shrimp": 0
    }
    
    #Farm production per household category
    farm_prod_per_hh = {
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
    subs_food_cons = {
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
    farm_prod_market = {
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
    farm_prod_food = {
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
            },
        "fish-rice":
            {
            "small": 0,
            "med": 0,
            "large": 0
            }
        }
        
    #Farm gross income
    farm_gross_income = {
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
    prod_cost = {
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
    farm_net_income = {
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
    tot_hh_income = {
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
    income_above_poverty = {
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
    food_security = {
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
    req_perm_farm_empl = {
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
    req_seasonal_farm_empl = {
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
    landless_farmer = {
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
    migration_family = {
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
    
    
    #FUNCTIONS
    #Farm production
    #Rice
    farm_prod['rice'] = (1-wlog_sev)*farmprod['rice'] #ton/hectare
    #Fish
    if wlog_sev > 0.8:
        farm_prod['fish'] = (farmprod['fish']*((1-wlog_sev)+0.6))
    else:
        farm_prod['fish'] = farmprod['fish']
    #Shrimp
    if wlog_sev > 0.8:
        farm_prod['shrimp'] = (farmprod['shrimp']*((1-wlog_sev)+0.6))
    else:
        farm_prod['shrimp'] = farmprod['shrimp']
        
    #Farm production per household category
    for hh in ['small', 'med', 'large']:
        for crop in ['rice', 'fish', 'shrimp']:
            farm_prod_per_hh[crop][hh] = farm_prod[crop] * farmsize[hh]
        
    #Subsistence consumption
    for hh in ['small', 'med', 'large']:
        for crop in ['rice', 'fish', 'shrimp']:
            subs_food_cons[crop][hh] = householdsize[hh] * consumption[crop]            
                
    #Farm production for market           
    for hh in ['small', 'med', 'large']:
        for crop in ['rice', 'fish', 'shrimp']:
            if hh == 'small' and crop == 'rice':
                if farm_prod_market[crop][hh] - (subs_food_cons[crop][hh]/1000.0) < 0:
                    farm_prod_market[crop][hh] = 0
            else:
                farm_prod_market[crop][hh] = farm_prod_per_hh[crop][hh] - (subs_food_cons[crop][hh]/1000.0)
  
    #Farm production for food
    for hh in ['small', 'med', 'large']:
        for crop in ['rice', 'fish', 'shrimp', 'fish-rice']:
            if crop == 'rice' or crop == 'fish' or crop == 'shrimp':
                if subs_food_cons[crop][hh]/1000.0 > farm_prod_per_hh[crop][hh]:
                    farm_prod_food[crop][hh] = farm_prod_per_hh[crop][hh]
                else:
                    farm_prod_food[crop][hh] = subs_food_cons[crop][hh]/1000.0    
            elif crop == 'fish-rice':
                if farm_prod['rice'] * farmsize[hh] > subs_food_cons['rice'][hh]/1000.0  :
                    farm_prod_food[crop][hh] = subs_food_cons['rice'][hh]
                else:
                    farm_prod_food[crop][hh] = farm_prod['rice'] * farmsize[hh]
            
    #Farm gross income
    for hh in ['small', 'med', 'large']:
        for crop in ['rice', 'fish', 'shrimp']:
            if crop == 'rice':
                farm_gross_income[crop][hh] = farm_prod_market[crop][hh] * 1000.0 * price_farm_gate["HYV_Boro"]
            elif crop == 'fish': 
                farm_gross_income[crop][hh] = farm_prod_market[crop][hh] * 1000.0 * price_farm_gate["freshw_fish"]
            elif crop == 'shrimp':
                farm_gross_income[crop][hh] = farm_prod_market[crop][hh] * 1000.0 * price_farm_gate["saltw_shrimp"]
                
    #Production cost
    for hh in ['small', 'med', 'large']:
        for crop in ["rice_irrig", "rice_no_irrig", "rice_irrig_landlease", "rice_no_irrig_landlease", "fish_landlease", "fish_no_landlease", "shrimp_landlease", "shrimp_no_landlease"]:
            if crop == 'rice_irrig':
                prod_cost[crop][hh] = farmsize[hh] * (var_prod_costs['rice'][hh] + human_lab[hh] + irrigation[hh])        
            elif crop == "rice_no_irrig":
                prod_cost[crop][hh] = farmsize[hh] * (var_prod_costs['rice'][hh] + human_lab[hh])   
            elif crop == "rice_irrig_landlease":
                prod_cost[crop][hh] = farmsize[hh] * (var_prod_costs['rice'][hh] + human_lab[hh] + irrigation[hh]) + (leasedarea[hh] * land_lease) 
            elif crop == "rice_no_irrig_landlease":
                prod_cost[crop][hh] = farmsize[hh] * (var_prod_costs['rice'][hh] + human_lab[hh]) + (leasedarea[hh] * land_lease) 
            elif crop == "fish_landlease":
                prod_cost[crop][hh] = (farmsize[hh] * var_prod_costs['fish']) + (leasedarea[hh] * land_lease) 
            elif crop == "fish_no_landlease":
                prod_cost[crop][hh] = (farmsize[hh] * var_prod_costs['fish'])
            elif crop == "shrimp_landlease":
                prod_cost[crop][hh] = (farmsize[hh] * var_prod_costs['shrimp']) + (leasedarea[hh] * land_lease) 
            elif crop == "shrimp_no_landlease":
                prod_cost[crop][hh] = (farmsize[hh] * var_prod_costs['shrimp'])                        
 
    #Farm net income
    for hh in ['small', 'med', 'large']:
        for crop in ["rice_irrig", "rice_no_irrig", "rice_irrig_landlease", "rice_no_irrig_landlease", "fish_landlease", "fish_no_landlease", "shrimp_landlease", "shrimp_no_landlease", "fish-rice_landlease", "fish-rice_no_landlease"]:            
                if crop == "fish-rice_landlease":
                    farm_net_income[crop][hh] = (farm_net_income["rice_no_irrig_landlease"][hh] + farm_net_income["fish_landlease"][hh])/2.0 - (cost_trans_rice_fish * land_lease) 
                elif crop == "fish-rice_no_landlease":
                    farm_net_income[crop][hh] = (farm_net_income["rice_no_irrig"][hh] + farm_net_income["fish_no_landlease"][hh])/2.0 - (cost_trans_rice_fish * land_lease) 
                else:
                    farm_net_income[crop][hh] = farm_gross_income[crop.split('_')[0]][hh] - prod_cost[crop][hh]
                    
    #Total household income
    for hh in ['small', 'med', 'large']:
        for crop in ["rice_irrig", "rice_no_irrig", "rice_irrig_landlease", "rice_no_irrig_landlease", "fish_landlease", "fish_no_landlease", "shrimp_landlease", "shrimp_no_landlease", "fish-rice_landlease", "fish-rice_no_landlease"]:            
           tot_hh_income[crop][hh] = farm_net_income[crop][hh] + hh_income_additional[hh] + (migrated_hh_members[hh] * 50 * 6 * migr_income * migr_income_send_home)

    #Income above poverty line (corrected for own food consumption)
    for hh in ['small', 'med', 'large']:
        for crop in ["rice_irrig", "rice_no_irrig", "rice_irrig_landlease", "rice_no_irrig_landlease", "fish_landlease", "fish_no_landlease", "shrimp_landlease", "shrimp_no_landlease", "fish-rice_landlease", "fish-rice_no_landlease"]:             
            if crop == "fish-rice_landlease" or crop == "fish-rice_no_landlease":
                income_above_poverty[crop][hh] = tot_hh_income[crop][hh] > ((poverty_line * 365 * householdsize[hh]) - ((farm_prod_food['fish'][hh] * 1000 * selling_price['rice']) + (farm_prod_food['rice'][hh] * 1000 * selling_price['rice']))/2.0 )
            else:
                income_above_poverty[crop][hh] = tot_hh_income[crop][hh] > ((poverty_line * 365 * householdsize[hh]) - (farm_prod_food[crop.split('_')[0]][hh] * 1000 * selling_price[crop.split('_')[0]]))

    #Food security
    for hh in ['small', 'med', 'large']:
        for crop in ["rice_irrig", "rice_no_irrig", "rice_irrig_landlease", "rice_no_irrig_landlease", "fish_landlease", "fish_no_landlease", "shrimp_landlease", "shrimp_no_landlease", "fish-rice_landlease", "fish-rice_no_landlease"]:             
            if crop == "rice_irrig" or crop == "rice_no_irrig" or crop == "rice_irrig_landlease" or crop == "rice_no_irrig_landlease":
                if (farm_prod_food['rice'][hh] < subs_food_cons['rice'][hh] / 1000.0) and (income_above_poverty[crop][hh] == False):
                    food_security[crop][hh] = False
                else:
                    food_security[crop][hh] = True
            elif crop == "fish_landlease" or crop == "fish_no_landlease" or crop == "shrimp_landlease" or crop == "shrimp_no_landlease":     
                if income_above_poverty[crop][hh] == False:
                    food_security[crop][hh] = False
                else:
                    food_security[crop][hh] = True   
            else:
                if (farm_prod_food['fish-rice'][hh] < subs_food_cons['rice'][hh] / 1000.0) and (income_above_poverty[crop][hh] == False):
                    food_security[crop][hh] = False
                else:
                    food_security[crop][hh] = True
    
    return income_above_poverty, food_security 


#%%RUN CALCULATION (Loop from 2022 to 2100)
#initialize arrays and lists
indicators_av=np.zeros((3,(endyear-startyear)+1,1))
indicators_pol=np.zeros((3,(endyear-startyear)+1,no_polder))
df = pd.DataFrame(columns=['Year', 'Indicator', 'Polder','Value'])

#loop over timesteps
i=0
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
    pcrldd = pcr.lddcreate(pcrelev,2*trmsedrate,1.0e+12,1.0e+12,1.0e+12)

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
    submat = pcr.pcr2numpy(pcrsub,-999)

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
    wl_wet=0.3
    #Wet low tide level
    lt_wet= np.maximum((msl + wl_wet - 0.5 * tidalrange), polderbedlevelmat)
    
    #water logging - patches with gradient less than drainhead to low tide
    gradient_dry=np.full(np.shape(elevmat),-999.0)
    gradient_wet=np.full(np.shape(elevmat),-999.0)
    gradient_dry[is_land] = (elevmat[is_land] - lt_dry[is_land]) / dist2rivmat[is_land]
    gradient_wet[is_land] = (elevmat[is_land] - lt_wet[is_land]) / dist2rivmat[is_land]
       
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

    #TRM
    is_TRM=False
        
    ht_wet= msl + wl_wet + 0.5 * tidalrange #calculate for each cell the high tide level of the nearest cell in the wet season

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
        plt.title('trmlevel_' + str(p_id) + '_' +  str(year))
   
    if is_TRM:
        elevmat = elevmat + trmlevelyear1
        trmlevelyear1=np.full(np.shape(elevmat), 0)
        is_TRM_prev = True
         
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
                (income_above_poverty, food_security) = agent_functions(waterlogged_sev_wet[x,y])
                (income_above_poverty, food_security) = agent_functions(0.8)
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
