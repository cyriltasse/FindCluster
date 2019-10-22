#!/usr/bin/env python
import numpy as np
import pylab
import tables
import astropy.io.fits as pyfits
from astropy.cosmology import WMAP9 as cosmo
import tables
import matplotlib.pyplot as pylab
from astropy.wcs import WCS
from astropy.io import fits as fitsio
from pyrap.images import image as CasaImage
from DDFacet.Other import logger
log = logger.getLogger("ClassCatalogMachine")
from DDFacet.Other import ModColor
from DDFacet.Other.progressbar import ProgressBar
from DDFacet.Other import AsyncProcessPool
import ClassDisplayRGB
import ClassSaveFITS
from DDFacet.ToolsDir import ModCoord

NamesRGB=["/data/tasse/DataDeepFields/EN1/optical_images/sw2band/EL_EN1_sw2band.fits",
          "/data/tasse/DataDeepFields/EN1/optical_images/Kband/EL_EN1_Kband.fits",
          "/data/tasse/DataDeepFields/EN1/optical_images/iband/EL_EN1_iband.fits"]

# # ##############################
# # Catch numpy warning
# np.seterr(all='raise')
# import warnings
# warnings.filterwarnings('error')
# #with warnings.catch_warnings():
# #    warnings.filterwarnings('error')
# # ##############################


def AngDist(ra0,dec0,ra1,dec1):
    AC=np.arccos
    C=np.cos
    S=np.sin
    D=S(dec0)*S(dec1)+C(dec0)*C(dec1)*C(ra0-ra1)
    if type(D).__name__=="ndarray":
        D[D>1.]=1.
        D[D<-1.]=-1.
    else:
        if D>1.: D=1.
        if D<-1.: D=-1.
    return AC(D)

class ClassCatalogMachine():
    def __init__(self,ra,dec,CellDeg=0.01,NPix=71,ScaleKpc=100,z=[0.01,2.,40],NCPU=0,CubeMode=True):
        self.rac_deg,self.dec_deg=ra,dec
        self.rac=ra*np.pi/180
        self.decc=dec*np.pi/180
        self.CellDeg=CellDeg
        self.CellRad=CellDeg*np.pi/180
        boxDeg=CellDeg*NPix
        self.boxDeg=boxDeg
        self.boxRad=boxDeg*np.pi/180
        self.NPix=NPix
        self.ScaleKpc=ScaleKpc
        self.zg=np.linspace(*z)
        self.CoordMachine = ModCoord.ClassCoordConv(self.rac, self.decc)
        nn=NPix//2
        lg,mg=np.mgrid[-nn*self.CellRad:nn*self.CellRad:1j*self.NPix,-nn*self.CellRad:nn*self.CellRad:1j*self.NPix]
        rag,decg=self.CoordMachine.lm2radec(lg.flatten(),mg.flatten())
        self.rag=rag.reshape(lg.shape)
        self.decg=decg.reshape(lg.shape)

        self.NCPU=NCPU
        self.MaskFits=None
        self.DicoDATA={}

    def showRGB(self):
        DRGB=ClassDisplayRGB.ClassDisplayRGB()
        DRGB.setRGB_FITS(*NamesRGB)
        radec=[self.rac_deg,self.dec_deg]
        DRGB.setRaDec(*radec)
        DRGB.setBoxArcMin(self.boxDeg*60)
        DRGB.FitsToArray()
        DRGB.Display(NamePNG="RGBOut.png",vmax=500)


    def setCat(self,CatName):
        print>>log,"Opening catalog fits file: %s"%CatName
        self.Cat=pyfits.open(CatName)[1].data
        self.Cat=self.Cat.view(np.recarray)
        
        self.CatRange=np.arange(self.Cat.FLAG_CLEAN.size)
            
        ind=np.where((self.Cat.FLAG_CLEAN == 1)&
                     (self.Cat.i_fluxerr > 0)&
                     (self.Cat.K_flux > 0)&
                     (self.Cat.ch2_swire_fluxerr > 0))[0]
        # ind=np.where((self.Cat.FLAG_CLEAN == 1)&
        #              (self.Cat.i_fluxerr > 0)&
        #              #(self.Cat.K_flux > 0)&
        #              (self.Cat.ch2_swire_fluxerr > 0))[0]
        
        
        self.Cat=self.Cat[ind]
        self.CatRange=self.CatRange[ind]
        # K=self.Cat.K_flux-self.Cat.K_flux.min()+1.
        # pylab.hist(np.log10(K),bins=100)
        # pylab.show()
        self.indFLAG=ind

        self.Cat.RA*=np.pi/180
        self.Cat.DEC*=np.pi/180

        RA=self.Cat.RA
        DEC=self.Cat.DEC

        # MASS.fill(1.)

        
        if self.MaskFits:
            print>>log, "Flagging in-mask sources..."
            FLAGMASK=np.bool8(np.array([self.GiveMaskFlag(RA[iS],DEC[iS]) for iS in range(self.Cat.RA.size)]))
            #FLAGMASK.fill(1)
            self.Cat=self.Cat[FLAGMASK]
            self.CatRange=self.CatRange[FLAGMASK]
            print>>log, "  done ..."

        
        self.DicoDATA["RA"]=self.Cat.RA[:]
        self.DicoDATA["DEC"]=self.Cat.DEC[:]
        self.DicoDATA["K"]=self.Cat.K_flux[:]
        self.RA_orig,self.DEC_orig=self.DicoDATA["RA"].copy(),self.DicoDATA["DEC"].copy()
    
    def RaDecToMaskPix(self,ra,dec):
        if abs(ra)>2*np.pi: stop
        if abs(dec)>2*np.pi: stop
        f,p=self.fp
        _,_,xc,yc=self.MaskCasaImage.topixel([f,p,dec,ra])
        xc,yc=int(xc),int(yc)
        return xc,yc
        
    def GiveMaskFlag(self,ra,dec):
        xc,yc=self.RaDecToMaskPix(ra,dec)
        FLAG=self.MaskArray[0,0,xc,yc]
        return 1-FLAG
    
    
    def setMask(self,MaskImage):
        print>>log,"Opening mask image: %s"%MaskImage
        self.MaskFits=pyfits.open(MaskImage)[0]
        self.MaskArray=self.MaskFits.data
        self.MaskCasaImage=CasaImage(MaskImage)
        f,p,_,_=self.MaskCasaImage.toworld([0,0,0,0])
        self.fp=f,p
        self.CDELT=abs(self.MaskFits.header["CDELT1"])

    def giveFracMasked(self,ra,dec,R):
        xc,yc=self.RaDecToMaskPix(ra,dec)
        Rpix=int(R/self.CDELT)
        ThisMask=self.MaskArray[0,0,xc-Rpix:xc+Rpix+1,yc-Rpix:yc+Rpix+1]
        x,y=np.mgrid[-Rpix:Rpix+1,-Rpix:Rpix+1]
        r=np.sqrt(x**2+y**2)
        ThisMask[r>Rpix]=-1
        UsedArea_pix=(np.where(ThisMask==0)[0]).size
        Area_pix=(np.where(ThisMask!=-1)[0]).size
        if Area_pix!=0:
            frac=UsedArea_pix/float(Area_pix)
        else:
            frac=1.

        if frac==0: frac=-1.
        if frac<0.5: frac=-1
        # pylab.clf()
        # pylab.imshow(ThisMask,interpolation="nearest")
        # pylab.draw()
        # pylab.show(False)
        # pylab.pause(0.1)
        # stop
        # print frac
        return frac

        
    def setPz(self,PzFile):
        print>>log,"Opening p-z hdf5 file: %s"%PzFile
        H=tables.open_file(PzFile)
        self.DicoDATA["zgrid_pz"]=H.root.zgrid[:]
        self.DicoDATA["pz"]=H.root.Pz[:][self.CatRange].copy()
        
        
        H.close()

    def SaveFITS(self,Name="Test.fits"):
        G=self.DicoDATA["ngrid"]
        GG=G.reshape((self.nch,self.NPix,self.NPix))
        im=ClassSaveFITS.ClassSaveFITS(Name,
                                       GG.shape,
                                       self.CellDeg,
                                       (self.rac,self.decc))#, Freqs=np.linspace(1400e6,1500e6,20))


        Gd=np.zeros_like(G)
        for ich in range(self.nch):
            Gd[ich]=G[ich].T[:,::-1]
        im.setdata(Gd.astype(np.float32))#,CorrT=True)
        im.ToFits()
        im.close()
        DRGB=ClassDisplayRGB.ClassDisplayRGB()
        DRGB.setRGB_FITS(*NamesRGB)
        radec=[self.rac_deg,self.dec_deg]
        DRGB.setRaDec(*radec)
        DRGB.setBoxArcMin(self.boxDeg*60)
        DRGB.loadFitsCube(Name,Sig=1)
        DRGB.Display(NamePNG="%s.png"%Name,Scale="linear",vmin=5,vmax=30)

   
def test(Show=True,NameOut="Test100kpc.fits"):

    Cat="/data/tasse/DataDeepFields/EN1/EN1_opt_spitzer_merged_vac_opt3as_irac4as_all_hpx_public.fits"
    Pz="/data/tasse/DataDeepFields/EN1/EN1_opt_spitzer_merged_vac_opt3as_irac4as_all_public_pz.hdf"
    MaskImage="/data/tasse/DataDeepFields/EN1/optical_images/iband/EL_EN1_iband.fits.mask.fits"
    rac,decc=241.20678,55.59485 # cluster
    #rac,decc=240.36069,55.522467 # star
    #rac,decc=radec
    COM=ClassOverdensityMap(rac,decc,
                            CellDeg=0.001,
                            NPix=501,
                            ScaleKpc=500)
    if Show:
        COM.showRGB()
    COM.setMask(MaskImage)
    COM.setCat(Cat)
    COM.setPz(Pz)
    COM.finaliseInit()
    # COM.giveDensityAtRaDec(rac,decc)
    COM.giveDensityGrid(Randomize=True)
    COM.giveDensityGrid(Randomize=False)
    COM.normaliseCube()
    COM.killWorkers()
    COM.SaveFITS(Name=NameOut)

if __name__=="__main__":
    try:
        test()
    except Exception,e:
        print>>log, "Got an exception... : %s"%str(e)    
        print>>log, "killing workers..."
        APP.terminate()
        APP.shutdown()
        Multiprocessing.cleanupShm()
      
