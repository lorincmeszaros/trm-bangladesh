# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 15:44:47 2022

@author: lorinc
"""
import numpy as np
import math as math
import pcraster as pcr

model_params = {
    "height": 100,
    "width": 20,
    "slr2100": 1, #UserSettableParameter("slider", "Mean Sea Level Rise to 2100 (m)", 1.00, 0.00, 5.00, 0.01),
    "subsidence": 2, #UserSettableParameter("slider", "Soil subsidence rate (cm/year)", 2, 0, 10, 1),
    "sedrate": 3, #UserSettableParameter("slider", "Sedimentation Rate Outside Polders (cm/year)", 3, 0, 10, 1),
    "trmsedrate": 20, #UserSettableParameter("slider", "Sedimentation Rate TRM Polders (cm/year)", 20, 0, 100, 1),
    "showvar": 'Polders'#UserSettableParameter('choice', 'Select variable to show on map', value='Polders', choices=['Elevation', 'Polders', 'Productivity'])
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
    self.pos = pos
    self.type = 1
    self.z = z
    self.rs = rs
    self.pol = pol
    self.model = model
    self.rspid = 100 * rs + pol
    self.dist2riv = dist2riv
    self.dist2sea = dist2sea
    self.productivity = 0.0
    self.is_waterlogged = False
    self.is_trm = False
    self.is_waterlogged = False
    if rs == 1:
        self.is_sea = True
        self.is_river=False
        self.is_polder = False
        self.is_nopolder = False
        self.is_land = False
    elif rs == 2:
        self.is_sea = False
        self.is_river= True
        self.is_polder = False
        self.is_nopolder = False
        self.is_land = False
    elif rs == 0:
        self.is_sea = False
        self.is_river= False
        self.is_land = True
        if pol == 0:
            self.is_polder = False
            self.is_nopolder = True
        else:
            self.is_polder = True
            self.is_nopolder = False
            self.polid = pol
    
    # Set up environment agents
    for cell in self.grid.coord_iter():
        x = cell[1]
        y = cell[2]
        z = elevmat[height - y - 1,x]
        rs = rsmat[height - y - 1,x]
        pol = polmat[height - y - 1,x]
        dist2riv = dist2rivmat[height - y - 1,x]
        dist2sea = dist2seamat[height - y - 1,x]
    
    # update topography
    if is_land: 
        # soil subsidence
        z = z - subsidence * 0.01
        #sedimentation on land outside polders
        if is_nopolder:
            z = z + sedrate * 0.01
        #sedimentation in trm areas
        if is_trm:
            z = z + trmsedrate * 0.01
    
        elevmat[height - y - 1,x] = z
    
    #tidal range - for the moment a linear decrease with 2cm per km from 2m at sea
    tidalrange = 2. - 0.02 * dist2sea
    if tidalrange < 0.: tidalrange = 0.
    #flood depth - high tide minus elevation
    if ((is_nopolder) or (is_trm) or (is_river)):
        flooddepth = msl + tidalrange * 0.5 - z
        if flooddepth < 0.: flooddepth = 0.
    else:
        flooddepth = 0.
        
    #upstream drainage area for each river cell as number of nopolder and trm cells with pycor > pycor of the patch itself    
    if (is_river):    
        rivy = pos[1]
        usdraina = 0.
        prism = 0.
        for cell in self.model.grid.coord_iter():
            y = cell[2]
            upagent = cell[0]
            if (y > rivy) and ((upagent.is_nopolder) or (upagent.is_trm) or (upagent.is_river)):
                usdraina += 1
                prism += upagent.flooddepth
    
    
        #upstream flow - perhaps later, for now zero
        
        #polder drainage - perhaps later, for now zero
        
    #water logging - patches with gradient less than drainhead to low tide
    if (is_land):
        gradient = (z - (msl - 0.5 * tidalrange )) / dist2riv
        if gradient < mindraingrad:
            is_waterlogged = True
    else:
        is_waterlogged = False
    
    #river flow
    
    #river bed --> update elevation
    
    #river salt - later, for now fixed

####
class EnvAgent():
    """
    Agent to describe the environment attributes like elevation, river, sea, polder
    """

    def __init__(self, pos, model, z, rs, pol, dist2riv, dist2sea):
        """
        Create a new elevation agent.

        Args:
           pos(x, y): Agent initial location.
           z: elevation (m)
           rs: river / sea code
           pol: polder code
        """
        super().__init__(pos, model)
        self.pos = pos
        self.type = 1
        self.z = z
        self.rs = rs
        self.pol = pol
        self.model = model
        self.rspid = 100 * rs + pol
        self.dist2riv = dist2riv
        self.dist2sea = dist2sea
        self.productivity = 0.0
        self.is_waterlogged = False
        self.is_trm = False
        self.is_waterlogged = False
        if rs == 1:
            self.is_sea = True
            self.is_river=False
            self.is_polder = False
            self.is_nopolder = False
            self.is_land = False
        elif rs == 2:
            self.is_sea = False
            self.is_river= True
            self.is_polder = False
            self.is_nopolder = False
            self.is_land = False
        elif rs == 0:
            self.is_sea = False
            self.is_river= False
            self.is_land = True
            if pol == 0:
                self.is_polder = False
                self.is_nopolder = True
            else:
                self.is_polder = True
                self.is_nopolder = False
                self.polid = pol
            

        def step1(self):
        """
        Run stage one of step for the environmental agent
        """
        # update topography
        if self.is_land: 
            # soil subsidence
            self.z = self.z - self.model.subsidence * 0.01
            #sedimentation on land outside polders
            if self.is_nopolder:
                self.z = self.z + self.model.sedrate * 0.01
            #sedimentation in trm areas
            if self.is_trm:
                self.z = self.z + self.model.trmsedrate * 0.01
        
            self.model.elevmat[self.model.height - self.y - 1,self.x] = self.z
        
        #tidal range - for the moment a linear decrease with 2cm per km from 2m at sea
        self.tidalrange = 2. - 0.02 * self.dist2sea
        if self.tidalrange < 0.: self.tidalrange = 0.
        #flood depth - high tide minus elevation
        if ((self.is_nopolder) or (self.is_trm) or (self.is_river)):
            self.flooddepth = self.model.msl + self.tidalrange * 0.5 - self.z
            if self.flooddepth < 0.: self.flooddepth = 0.
        else:
            self.flooddepth = 0.
            
        def step2(self):
        """
        Run stage two of step for the environmental agent
        """
        if (self.is_river):
            #upstream drainage area for each river cell as number of nopolder and trm cells with pycor > pycor of the patch itself
            rivy = self.pos[1]
            usdraina = 0.
            prism = 0.
            for cell in self.model.grid.coord_iter():
                y = cell[2]
                upagent = cell[0]
                if (y > rivy) and ((upagent.is_nopolder) or (upagent.is_trm) or (upagent.is_river)):
                    usdraina += 1
                    prism += upagent.flooddepth
        
        
            #upstream flow - perhaps later, for now zero
            
            #polder drainage - perhaps later, for now zero
            
        #water logging - patches with gradient less than drainhead to low tide
        if (self.is_land):
            gradient = (self.z - (self.model.msl - 0.5 * self.tidalrange )) / self.dist2riv
            if gradient < self.model.mindraingrad:
                self.is_waterlogged = True
        else:
            self.is_waterlogged = False
        
        #river flow
        
        #river bed --> update elevation
        
        #river salt - later, for now fixed