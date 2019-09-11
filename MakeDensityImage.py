#!/usr/bin/env python

import numpy as np
from DDFacet.ToolsDir.rad2hmsdms import rad2hmsdms
import astropy.io.fits as pyfits
import sys

def MakeDensityImage(CatName):

    Cat=pyfits.open(CatName)[1].data
    Cat=Cat.view(np.recarray)
    
    ind=np.where((Cat.FLAG_CLEAN == 1)&(Cat.i_fluxerr > 0)&
                 (Cat.ch2_swire_fluxerr > 0))[0]
    
    Cat=Cat[ind]
    indFLAG=ind
    Cat.RA[:]*=np.pi/180
    Cat.DEC[:]*=np.pi/180
    
    NPix=1000
    
    Image=pyfits.open("/data/tasse/DataDeepFields/EN1/optical_images/iband/EL_EN1_iband.fitsCatName")[1].data
    Image.fill(0)
    
    
if __name__=="__main__":
    MakeDensityImage(sys.argv[1])
