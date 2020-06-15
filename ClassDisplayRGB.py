from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import  CutImage
import numpy as np
import pylab
import os
import img_scale
from DDFacet.Other import logger
log = logger.getLogger("ClassDisplayRGB")
from astropy.io import fits as pyfits
from astropy.wcs import WCS
from DDFacet.ToolsDir import ModCoord

import ntpath
def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)

class ClassDisplayRGB():
    def __init__(self):
        pass

    def loadFitsCube(self,CubeName,Sig=2):
        ngrid=pyfits.open(CubeName)[0].data
        nch,nx,ny=ngrid.shape
        #ngridConv=np.zeros_like(ngrid)
        zg=np.linspace(0,nch-1,nch)[::-1]
        self.img=np.zeros((nx,ny, 3), dtype=float)
        z0,z1=zg.min(),zg.max()
        zz=np.linspace(z0,z1,4)
        zm=(zz[1::]+zz[0:-1])/2.
        MapChan=[]
        for ich in range(nch):
            Dch=np.argmin(np.abs(zg[ich]-zm))
            k=np.exp(-(zg-zg[ich])**2/(2*Sig**2))
            k/=np.sum(k)
            Coefs=np.zeros((3,),float)
            for iCoef in range(3):
                zz0,zz1=zz[iCoef],zz[iCoef+1]
                ind=np.where((zg>=zz0)&(zg<zz1))[0]
                Coefs[iCoef]=np.sum(k[ind])
                self.img[:,:,iCoef]+=ngrid[ich]*Coefs[iCoef]

        self.img=self.img[::-1,:]
            
    def setRGB_FITS(self,RName,GNane,BName):
        self.Names=[RName,GNane,BName]
        
    def setRaDec(self,ra,dec):
        self.ra=ra
        self.dec=dec
        
    def setBoxArcMin(self,boxArcMin):
        self.boxArcMin=boxArcMin

    def FitsToArray(self,Slice=None):
        self.CutNames=["/data/tasse/tmp/%s.cut.fits"%path_leaf(ThisName) for ThisName in self.Names]
        self.ListImage=[]
        self.LWCS=[]
        self.Slice=Slice
        for InFile,OutFile in zip(self.Names,self.CutNames):
            
            self.ListImage.append(CutImage.CutFits(InFile=InFile,
                                                   OutFile=OutFile,
                                                   Ra=self.ra,
                                                   Dec=self.dec,
                                                   boxArcMin=self.boxArcMin,
                                                   Overwrite=True,
                                                   dPix=1,
                                                   Slice=Slice))
            hdu = pyfits.open(OutFile)[0]
            wcs=WCS(hdu.header)
            self.LWCS.append(wcs)

            
        R,G,B=self.ListImage
        if not Slice:
            img = np.zeros((R.shape[0], R.shape[1], 3), dtype=float)
            img[:,:,0]=R[:,:]
            img[:,:,1]=G[:,:]
            img[:,:,2]=B[:,:]
        else:
            img = np.zeros((R.shape[1], R.shape[2], 3), dtype=float)
            img[:,:,0]=R[Slice,:,:]
            img[:,:,1]=G[Slice,:,:]
            img[:,:,2]=B[Slice,:,:]
        self.img=img[::-1,:]
        

        Image=self.CutNames[0]
        Fits=pyfits.open(Image)[0]
        
        if "CDELT1" in Fits.header.keys():
            self.CDELT=abs(Fits.header["CDELT1"])
        else:
            self.CDELT=abs(Fits.header["CD1_1"])
            
        self.rac=Fits.header["CRVAL1"]*np.pi/180
        self.decc=Fits.header["CRVAL2"]*np.pi/180
        
        
    def Display(self,
                Scale="sqrt",
                vmin=0,
                vmax=500,
                NamePNG=None,
                figax=None,
                SkipShow=False):
        
        img=self.img
        
        if Scale=="linear":
            ff=img_scale.linear
        elif Scale=="sqrt":
            ff=img_scale.sqrt
        elif Scale=="log":
            ff=img_scale.log
        
        img[:,:,0] = ff(img[:,:,0], scale_min=vmin, scale_max=vmax)
        img[:,:,1] = ff(img[:,:,1], scale_min=vmin, scale_max=vmax)
        img[:,:,2] = ff(img[:,:,2], scale_min=vmin, scale_max=vmax)

        if figax is None:
            fig=pylab.figure("RGB Composite",figsize=(10,10))
            pylab.clf()
            ax=pylab.subplot(1,1,1)
        else:
            fig,ax=figax
            
        nx=img.shape[0]
        d=(nx//2+0.5)*self.CDELT
        wcs=self.LWCS[0]
        if not self.Slice:
            ra0,dec0=wcs.all_pix2world([0],[0],1)
            ra1,dec1=wcs.all_pix2world([nx-1],[nx-1],1)
        else:
            ra0,dec0,_=wcs.all_pix2world([0],[0],[self.Slice],1)
            ra1,dec1,_=wcs.all_pix2world([nx-1],[nx-1],[self.Slice],1)
            
        CoordMachine = ModCoord.ClassCoordConv(self.rac, self.decc)
        l0,m0=CoordMachine.radec2lm(ra0*np.pi/180,dec0*np.pi/180)
        l1,m1=CoordMachine.radec2lm(ra1*np.pi/180,dec1*np.pi/180)

        # l0=self.rac_cut-d
        # l1=self.rac_cut+d
        # m0=self.decc_cut-d
        # m1=self.decc_cut+d
        self.img=img
        self.extent=[l0[0],l1[0],m0[0],m1[0]]
        if not SkipShow:
            ax.imshow(img,
                      aspect='equal',
                      extent=self.extent)
            pylab.draw()
            if NamePNG:
                log.print("Saving image as: %s"%NamePNG)
                fig.savefig(NamePNG)
            
            pylab.show(block=False)
            pylab.pause(0.1)

        
      

def test():
    Names=["/data/tasse/DataDeepFields/EN1/optical_images/sw2band/EL_EN1_sw2band.fits",
           "/data/tasse/DataDeepFields/EN1/optical_images/Kband/EL_EN1_Kband.fits",
           "/data/tasse/DataDeepFields/EN1/optical_images/iband/EL_EN1_iband.fits"]
    DRGB=ClassDisplayRGB()
    DRGB.setRGB_FITS(*Names)

    radec=[241.51386,55.424]
    DRGB.setRaDec(*radec)
    DRGB.setBoxArcMin(20.)
    DRGB.FitsToArray()
    #F=pyfits.open("Test100kpc.fits")[0].data
    #DRGB.setCube(F,Sig=1)
    DRGB.Display(Scale="linear",vmin=3,vmax=30)

    
