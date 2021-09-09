#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
os.environ["OMP_NUM_THREADS"] = "1"
from multiprocessing import Pool
#import emcee
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as pylab
from astropy.cosmology import WMAP9 as cosmo
from DDFacet.Other import logger
log = logger.getLogger("RunMCMC")
from DDFacet.Other.AsyncProcessPool import APP, WorkerProcessError
from DDFacet.Other import Multiprocessing
from DDFacet.Other import ModColor
from DDFacet.Other import ClassTimeIt
from DDFacet.Other.progressbar import ProgressBar
from DDFacet.Other import AsyncProcessPool
from DDFacet.Array import shared_dict
import ClassCatalogMachine
import ClassInitGammaCube
import ClassGammaMachine_Wave
import ClassDisplayRGB
import ClassSaveFITS
import GeneDist
from DDFacet.ToolsDir.rad2hmsdms import rad2hmsdms


def main():
    CM=ClassCatalogMachine.ClassCatalogMachine()
    CM.Init()
    CM.Recompute_nz_nzt()
    Box=15.
    rac_deg,decc_deg=241.20678,55.59485
    Box=15.
    rac_deg,decc_deg=244.1718,55.75713889

    rarad,decrad=rac_deg*np.pi/180,decc_deg*np.pi/180
    print(rad2hmsdms(rarad,Type="ra").replace(" ",":"),rad2hmsdms(decrad,Type="dec").replace(" ","."))
    pylab.close("all")
    fig=pylab.figure("Cluster",figsize=(20,10))
    pylab.clf()
    ax=pylab.subplot(1,2,1)
    DRGB=ClassDisplayRGB.ClassDisplayRGB()
    DRGB.setRGB_FITS(*CM.DicoDataNames["RGBNames"])
    DRGB.setRaDec(rac_deg,decc_deg)
    DRGB.setBoxArcMin(Box)
    DRGB.FitsToArray()
    DRGB.Display(figax=(fig,ax),SkipShow=True)#Scale="linear",vmin=,vmax=30)
    ax.imshow(DRGB.img,
              aspect='equal',
              extent=DRGB.extent)
    ax.axes.xaxis.set_ticklabels([])
    ax.axes.yaxis.set_ticklabels([])
    pylab.grid()
    DRGB0=DRGB
    ax=pylab.subplot(1,2,2,sharex=ax,sharey=ax)
    LN=["EN1.Gamma.median.fits","EN1.Gamma.median.fits","EN1.Gamma.median.fits"]
    LN=["Test.median.fits","Test.median.fits","Test.median.fits"]
    DRGB=ClassDisplayRGB.ClassDisplayRGB()
    DRGB.setRGB_FITS(*LN)
    DRGB.setRaDec(rac_deg,decc_deg)
    DRGB.setBoxArcMin(Box)
    DRGB.FitsToArray(Slice=10)
    DRGB.Display(figax=(fig,ax),Scale="linear",vmin=-1.,vmax=2.,SkipShow=True)
    # ax.cla()
    ax.imshow(DRGB.img[:,:,0],
              aspect='equal',
              extent=DRGB0.extent,
              cmap=pylab.cm.cividis)
    #ax.axes.xaxis.set_visible(False)
    #ax.axes.yaxis.set_visible(False)
    ax.axes.xaxis.set_ticklabels([])
    ax.axes.yaxis.set_ticklabels([])
    pylab.grid()
    pylab.tight_layout()
    pylab.draw()
    pylab.show(block=False)
    pylab.pause(0.1)

    # if NamePNG:
    #     log.print("Saving image as: %s"%NamePNG)
    #             fig.savefig(NamePNG)
                
    
    
    
  
