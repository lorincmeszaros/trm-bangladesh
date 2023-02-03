# -*- coding: utf-8 -*-
"""
Created on Wed Dec  7 10:19:01 2022

@author: lorinc
"""

def agent_functions(wlog_sev):

    import numpy as np
    
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
#%%
    #INIT
    #Agent attributes
    # farmsize = {
    # "small": np.random.normal(loc=0.51, scale=(1.01-0.51)/3.0),
    # "med": np.random.normal(loc=2.02, scale=(3.03-2.02)/3.0),
    # "large": np.random.normal(loc=6.07, scale=(6.07-3.04)/3.0)
    # }
    
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
    
    # householdsize = {
    # "small": np.random.normal(loc=4.15, scale=(4.15-1.0)/3.0),
    # "med": np.random.normal(loc=4.15, scale=(4.15-1.0)/3.0),
    # "large": np.random.normal(loc=4.15, scale=(4.15-1.0)/3.0),
    # "landless": np.random.normal(loc=4.15, scale=(4.15-1.0)/3.0)
    # }

    householdsize = {
    "small": 4.15,
    "med": 4.15,
    "large": 4.15,
    "landless": 4.15,
    }

    # fam_member_12 = {
    # "small": np.random.normal(loc=1.0375, scale=1.0375/3.0),
    # "med": np.random.normal(loc=1.0375, scale=1.0375/3.0),
    # "large": np.random.normal(loc=1.0375, scale=1.0375/3.0),
    # "landless": np.random.normal(loc=1.0375, scale=1.0375/3.0)
    # }
 
    fam_member_12 = {
    "small": 1.0375,
    "med": 1.0375,
    "large": 1.0375,
    "landless": 1.0375,
    }
    
    # leasedarea = {
    # "small": np.random.normal(loc=0.1, scale=(0.1-0)/3.0),
    # "med": np.random.normal(loc=0.3, scale=(0.3-0)/3.0),
    # "large": np.random.normal(loc=0.4, scale=(0.4-0)/3.0)
    # }

    leasedarea = {
    "small": 0.1,
    "med": 0.3,
    "large": 0.4,
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
    "large": 95053.0,
    "landless": 47527.0
    }        

    #Migrated household members (initial set-up)            
    migrated_hh_members = {
    "small": 0.3,
    "med": 0.3,
    "large": 0.3
    }   

    #Irrigation % of farms
    irrigation_perc = {
    "small": 11.3/100.,
    "med": 74.6/100.,
    "large": 70.30/100.
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
    # days_seas_emp_landless = np.random.normal(loc=54., scale=(0.3*54.))
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
        "fish-rice": {
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
                "large": 0
                }
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
            },
        "rice-fish":
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
            },
        "rice-fish":
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
            }
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

       
    #Aggregate indicators       
    #Production of rice, fish and shrimp
    production = {  
    "rice": 
        {
        "small": 0.0,
        "med": 0.0,
        "large": 0.0
        },
    "fish":
        {
        "small": 0.0,
        "med": 0.0,
        "large": 0.0
        },
    "shrimp":
        {
        "small": 0.0,
        "med": 0.0,
        "large": 0.0
        }
    }    
    
    #% of population income below poverty line
    #% of unemployed rural labour
    employed = {  
    "rice": 
        {
        "small": 0.0,
        "med": 0.0,
        "large": 0.0
        },
    "fish":
        {
        "small": 0.0,
        "med": 0.0,
        "large": 0.0
        },
    "shrimp":
        {
        "small": 0.0,
        "med": 0.0,
        "large": 0.0
        }
    }       
    
    #% of population food insecure
    #% of population likely to migrate

#%%    
    #FUNCTIONS
    #Farm production
    #Rice
    farm_prod['rice'] = (1-wlog_sev)*farmprod['rice'] #ton/hectare
    #Fish
    if wlog_sev > 0.6:
        farm_prod['fish'] = (farmprod['fish']*((1-wlog_sev)+0.6))
    else:
        farm_prod['fish'] = farmprod['fish']
    #Shrimp
    if wlog_sev > 0.6:
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
                if farm_prod_per_hh[crop][hh] - (subs_food_cons[crop][hh]/1000.0) < 0:
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
                #Rice
                if farm_prod['rice'] * farmsize[hh] > subs_food_cons['rice'][hh]/1000.0  :
                    farm_prod_food[crop]['rice'][hh] = subs_food_cons['rice'][hh]/1000.0
                else:
                    farm_prod_food[crop]['rice'][hh] = farm_prod['rice'] * farmsize[hh]
                #Fish
                if subs_food_cons['fish'][hh]/1000.0 > farm_prod_per_hh['fish'][hh]/2.0:
                    farm_prod_food[crop]['fish'][hh] = farm_prod_per_hh['fish'][hh]/2.0
                else:
                    farm_prod_food[crop]['fish'][hh] = subs_food_cons['fish'][hh]/1000.0
            
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
                if (farm_prod_food['fish-rice']['rice'][hh] < subs_food_cons['rice'][hh] / 1000.0) and (income_above_poverty[crop][hh] == False):
                    food_security[crop][hh] = False
                else:
                    food_security[crop][hh] = True

    #Required hired permanent farm employment
    for hh in ['small', 'med', 'large']:
        for crop in ['rice', 'fish', 'shrimp', 'rice-fish']:
            if (householdsize[hh] - fam_member_12[hh] - migrated_hh_members[hh]) > farmempl["family_perm"][crop][hh]:
                req_perm_farm_empl[crop][hh] = farmempl["hired_perm"][crop][hh]
            else:
                if (householdsize[hh] - fam_member_12[hh] - migrated_hh_members[hh]) > 0.0:
                    req_perm_farm_empl[crop][hh] = farmempl["hired_perm"][crop][hh] + (householdsize[hh] - fam_member_12[hh] - migrated_hh_members[hh])
                else:
                    req_perm_farm_empl[crop][hh] = farmempl["hired_perm"][crop][hh] + farmempl["family_perm"][crop][hh]
             
    #Required hired seasonal farm employment
    for hh in ['small', 'med', 'large']:
        for crop in ['rice', 'fish', 'shrimp', 'rice-fish']:
            req_seasonal_farm_empl[crop][hh] = farmempl["hired_temp"][crop][hh] / temp_empl[hh]


    #Migration - family member
    for hh in ['small', 'med', 'large']:
        for crop in ["rice_irrig", "rice_no_irrig", "rice_irrig_landlease", "rice_no_irrig_landlease", "fish_landlease", "fish_no_landlease", "shrimp_landlease", "shrimp_no_landlease", "fish-rice_landlease", "fish-rice_no_landlease"]:             
            if income_above_poverty[crop][hh] == False and food_security[crop][hh] == False and ((householdsize[hh] - fam_member_12[hh])>2):
                migration_family[crop][hh] = True
            else:
                migration_family[crop][hh] = False
   
    #Landless farmers
    for attr in ["hh_income", "income_above_poverty", "food_security"]:
        for emp in ["perm_empl", "seasonal_empl"]:
            if attr == "hh_income":
                if emp == "perm_empl":
                    landless_farmer[attr][emp] = (50.0 * 5.0 * migr_income) * householdsize["landless"] * peop_work_landless
                elif emp == "seasonal_empl":
                    landless_farmer[attr][emp] = ((days_seas_emp_landless * migr_income) + hh_income_additional["landless"]) * householdsize["landless"] * peop_work_landless
            elif attr == "income_above_poverty":            
                landless_farmer[attr][emp] = landless_farmer["hh_income"][emp] > (householdsize["landless"] * poverty_line * 365.0)
            elif attr == "food_security":            
                landless_farmer[attr][emp] = landless_farmer["income_above_poverty"][emp]     


#%%
    #Aggregate indicators
    
    #Production of rice, fish and shrimp
    for hh in ['small', 'med', 'large']:
        for crop in ['rice', 'fish', 'shrimp']:
            production[crop][hh] = farm_prod_per_hh[crop][hh]
    
    return production, income_above_poverty, req_perm_farm_empl, req_seasonal_farm_empl, food_security, migration_family, landless_farmer 