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
import RecArrayOps
from DDFacet.Other import MyPickle

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
    def __init__(self):
        self.MaskFits=None
        self.DicoDATA={}
        self.PhysCat=None
        self.OmegaTotal=None

    def setPhysCatalog(self,CatName):
        self.PhysCatName=CatName
        print>>log,"Opening mass/sfr catalog fits file: %s"%CatName
        self.PhysCat=pyfits.open(CatName)[1].data
        self.PhysCat=self.PhysCat.view(np.recarray)

    def setCat(self,CatName):
        print>>log,"Opening catalog fits file: %s"%CatName
        self.PhotoCatName=CatName
        self.Cat=pyfits.open(CatName)[1].data
        self.Cat=self.Cat.view(np.recarray)
        
        if self.PhysCat is not  None:
            print>>log,"  Append fields..."
            self.Cat=RecArrayOps.AppendField(self.Cat,("Mass",np.float32))
            self.Cat=RecArrayOps.AppendField(self.Cat,("SFR",np.float32))
            self.Cat=RecArrayOps.AppendField(self.Cat,("z",np.float32))
            self.Cat.Mass.fill(-1)
            self.Cat.SFR.fill(-1)
            self.Cat.z.fill(-1)
            dID=self.Cat.id[1::]-self.Cat.id[0:-1]
            if np.max(dID)!=1: stop
            print>>log,"  Append physical information to photometric catalog..."
            self.Cat.Mass[self.PhysCat.ID]=self.PhysCat.Mass_median[:]  
            self.Cat.SFR[self.PhysCat.ID]=self.PhysCat.SFR_best[:]
            self.Cat.z[self.PhysCat.ID]=self.PhysCat.z[:]
            
          
        self.CatRange=np.arange(self.Cat.FLAG_CLEAN.size)
            
        print>>log,"Remove spurious objects..."
        ind=np.where((self.Cat.FLAG_CLEAN == 1)&
                     (self.Cat.i_fluxerr > 0)&
                     (self.Cat.K_flux > 0)&
                     (self.Cat.FLAG_OVERLAP==7)&
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


        self.DicoDATA["ID"]=np.int32(self.Cat.id[:])
        self.DicoDATA["RA"]=self.Cat.RA[:]
        self.DicoDATA["DEC"]=self.Cat.DEC[:]
        self.DicoDATA["K"]=self.Cat.K_flux[:]
        self.DicoDATA["Mass"]=self.Cat.Mass[:]
        self.DicoDATA["SFR"]=self.Cat.SFR[:]
        self.DicoDATA["z"]=self.Cat.z[:]
        self.RA_orig,self.DEC_orig=self.DicoDATA["RA"].copy(),self.DicoDATA["DEC"].copy()

        
    def PickleSave(self,FileName):
        print>>log, "Saving catalog as: %s"%FileName
        FileNames={"MaskFitsName":self.MaskFitsName,
                  "PhotoCatName":self.PhotoCatName,
                  "PhysCatName":self.PhysCatName}
        self.DicoDATA["FileNames"]=FileNames
        self.DicoDATA["OmegaTotal"]=self.OmegaTotal
        MyPickle.DicoNPToFile(self.DicoDATA,FileName)
        
    def PickleLoad(self,FileName):
        print>>log, "Loading catalog from: %s"%FileName
        self.DicoDATA=MyPickle.FileToDicoNP(FileName)
        self.OmegaTotal=self.DicoDATA["OmegaTotal"]
        self.setMask(self.DicoDATA["FileNames"]["MaskFitsName"])
    
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
        self.MaskFitsName=MaskImage
        self.MaskFits=pyfits.open(MaskImage)[0]
        self.MaskArray=self.MaskFits.data
        self.MaskCasaImage=CasaImage(MaskImage)
        f,p,_,_=self.MaskCasaImage.toworld([0,0,0,0])
        self.fp=f,p
        self.CDELT=abs(self.MaskFits.header["CDELT1"])
        if self.OmegaTotal is None:
            NPixZero=self.MaskArray.size-np.count_nonzero(self.MaskArray)
            self.OmegaTotal=NPixZero*(self.CDELT*np.pi/180)**2

        
    def setPz(self,PzFile):
        print>>log,"Opening p-z hdf5 file: %s"%PzFile
        H=tables.open_file(PzFile)
        self.DicoDATA["zgrid_pz"]=H.root.zgrid[:]
        self.DicoDATA["pz"]=H.root.Pz[:][self.CatRange].copy()
        H.close()
        
   
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
      
