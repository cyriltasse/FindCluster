import CutImage
import numpy as np
import pylab
import os
import img_scale
from DDFacet.Other import logger
log = logger.getLogger("ClassDisplayRGB")

import ntpath
def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)

class ClassDisplayRGB():
    def __init__(self,RName,GNane,BName):
        self.Names=[RName,GNane,BName]

    def setRaDec(self,ra,dec):
        self.ra=ra
        self.dec=dec
        
    def setBoxArcMin(self,boxArcMin):
        self.boxArcMin=boxArcMin

    def CutFits(self):
        self.CutNames=["/tmp/%s.cut.fits"%path_leaf(ThisName) for ThisName in self.Names]
        self.ListImage=[]
        for InFile,OutFile in zip(self.Names,self.CutNames):
            self.ListImage.append(CutImage.CutFits(InFile=InFile,
                                                   OutFile=OutFile,
                                                   Ra=self.ra,
                                                   Dec=self.dec,
                                                   boxArcMin=self.boxArcMin,
                                                   Overwrite=True,
                                                   dPix=1))

    def Display(self,
                Scale="sqrt",
                vmax=500):
        
        R,G,B=self.ListImage
        img = np.zeros((R.shape[0], R.shape[1], 3), dtype=float)

        if Scale=="linear":
            ff=img_scale.linear
        elif Scale=="sqrt":
            ff=img_scale.sqrt
        elif Scale=="log":
            ff=img_scale.log

        img[:,:,0] = ff(R, scale_min=0, scale_max=vmax)
        img[:,:,1] = ff(G, scale_min=0, scale_max=vmax)
        img[:,:,2] = ff(B, scale_min=0, scale_max=vmax)

        fig=pylab.figure("RGB Composite",figsize=(10,10))
        pylab.clf()
        pylab.imshow(img[::-1,:], aspect='equal',extent=[self.ra-self.boxArcMin/2/60.,self.ra+self.boxArcMin/2/60.,self.dec-self.boxArcMin/2/60.,self.dec+self.boxArcMin/2/60.])
        pylab.draw()
        pylab.show(False)
        pylab.pause(0.1)
        #pylab.title('Blue = J, Green = H, Red = K')
        #pylab.savefig('my_rgb_image.png')
        

def test():
    Names=["/data/tasse/DataDeepFields/EN1/optical_images/sw2band/EL_EN1_sw2band.fits",
           "/data/tasse/DataDeepFields/EN1/optical_images/Kband/EL_EN1_Kband.fits",
           "/data/tasse/DataDeepFields/EN1/optical_images/iband/EL_EN1_iband.fits"]
    DRGB=ClassDisplayRGB(*Names)
    radec=[241.51386,55.424]
    DRGB.setRaDec(*radec)
    DRGB.setBoxArcMin(20.)
    DRGB.CutFits()
    DRGB.Display()
