# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 13:26:41 2022

@author: lorinc
"""
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
