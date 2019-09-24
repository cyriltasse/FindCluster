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
log = logger.getLogger("ClassOverdensityMap")
from DDFacet.Array import shared_dict
from DDFacet.Other.AsyncProcessPool import APP, WorkerProcessError
from DDFacet.Other import Multiprocessing
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


class ClassOverdensityMap():
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
        self.DicoDATA = shared_dict.create("DATA")

    def showRGB(self):
        DRGB=ClassDisplayRGB.ClassDisplayRGB()
        DRGB.setRGB_FITS(*NamesRGB)
        radec=[self.rac_deg,self.dec_deg]
        DRGB.setRaDec(*radec)
        DRGB.setBoxArcMin(self.boxDeg*60)
        DRGB.FitsToArray()
        DRGB.Display(NamePNG="RGBOut.png",vmax=500)


    def finaliseInit(self):
        APP.registerJobHandlers(self)
        AsyncProcessPool.init(ncpu=self.NCPU,
                              affinity=0)
        APP.startWorkers()

    def killWorkers(self):
        print>>log, "Killing workers"
        APP.terminate()
        APP.shutdown()
        Multiprocessing.cleanupShm()

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
        #print self.DicoDATA["pz"].shape,self.Cat.shape
        

    #def _giveFractionMasked(self,ra,dec,rad):
    #def _giveLum(self,K_flux,z):

    def _randomiseCatalog(self):
        RA,DEC=self.DicoDATA["RA"],self.DicoDATA["DEC"]
        self.RA_orig,self.DEC_orig=self.DicoDATA["RA"].copy(),self.DicoDATA["DEC"].copy()
        ra0,ra1=RA.min(),RA.max()
        dec0,dec1=DEC.min(),DEC.max()
        ram=RA
        decm=DEC
        ras=np.array([],np.float32)
        decs=np.array([],np.float32)
        print>>log,"Generating randomised catalog..."
        while True:
            ra=ra0+np.random.rand(ram.size)*(ra1-ra0)
            dec=dec0+np.random.rand(decm.size)*(dec1-dec0)
            #print ram.size
            FLAGMASK=np.bool8(np.array([self.GiveMaskFlag(ra[iS],dec[iS]) for iS in range(ra.size)]))
            #print "f"
            if FLAGMASK.any():
                ras=np.concatenate([ras,ra[FLAGMASK]])
                decs=np.concatenate([decs,dec[FLAGMASK]])
            if (FLAGMASK==0).any():
                ram=ra[FLAGMASK==0]
                decm=dec[FLAGMASK==0]
            else:
                break
            
        print>>log,"  done..."
            
        self.DicoDATA["RA"][:]=ras.flat[:]
        self.DicoDATA["DEC"][:]=decs.flat[:]

    def _giveDensityAtRaDec(self,ipix,GridName):
        self.DicoDATA.reload()
        ra,dec=self.rag.flat[ipix],self.decg.flat[ipix]

        RA,DEC=self.DicoDATA["RA"],self.DicoDATA["DEC"]
        K_flux=self.DicoDATA["K"]
        D=AngDist(ra,dec,RA,DEC)
        pz=self.DicoDATA["pz"]
        n=0
        zgrid_pz=self.DicoDATA["zgrid_pz"]
        for iz in range(self.zg.size-1):
            z0,z1=self.zg[iz],self.zg[iz+1]
            zm=(z0+z1)/2.
            indz=np.where((zgrid_pz>z0)&(zgrid_pz<z1))[0]
            f_kpc2arcsec=cosmo.arcsec_per_kpc_comoving(zm).to_value()
            
            R=f_kpc2arcsec*self.ScaleKpc/3600.*np.pi/180
            ind=np.where(D<3*R)[0]
            #ind=np.where(D<R)[0]
            if ind.size==0:
                self.DicoDATA["%s_mask"%GridName][iz].flat[ipix]=1
                continue
            if indz.size==0: continue
            
            # pylab.clf()
            # pylab.scatter(RA,DEC,s=1,c="black")
            # pylab.scatter(RA[ind],DEC[ind],s=10,c="red")
            # pylab.scatter([ra],[dec],s=30,c="blue",marker="+")
            # pylab.scatter(self.rag.flat[:],self.decg.flat[:],s=30,c="blue")
            # pylab.show()

            Dkpc=D[ind]*(180/np.pi*3600)/f_kpc2arcsec
            w=np.exp(-Dkpc**2/(2.*self.ScaleKpc**2))
            #print w
            #w.fill(1.)
            #M=MASS[ind]
            Dl=cosmo.luminosity_distance(zm).to_value()
            Fint=K_flux[ind]*4.*np.pi*Dl**2
            if (Fint<=0).any(): stop
            M=np.log10(Fint)
            # w.fill(1.)
            M.fill(1.)
            
            PP=(w.reshape((-1,1))*M.reshape((-1,1))*pz[ind][:,indz]).flatten()
            # PP=(w.reshape((-1,1))*pz[ind][:,indz]).flatten()
            # PP=pz[ind][:,indz].flatten()
            indnan=np.logical_not(np.isnan(PP))
            frac=self.giveFracMasked(ra,dec,R)
            if frac>0.6:
                #n+=np.sum(PP[indnan])/frac
                self.DicoDATA["%s"%GridName][iz].flat[ipix]=np.sum(PP[indnan])/frac
            else:
                self.DicoDATA["%s_mask"%GridName][iz].flat[ipix]=1

            self.DicoDATA["%s_mask"%GridName][iz].flat[ipix]=1-self.GiveMaskFlag(ra,dec)
            
            if np.isnan(n):
                print "cata"
                stop

            
        #self.DicoDATA["ngrid"].flat[ipix]=n

    def giveDensityGrid(self,Randomize=False):
        nx,ny=self.rag.shape
        nch=self.zg.size-1
        self.nch=nch
        if not Randomize:
            print>>log,"Will be working on original catalogs"
            self.DicoDATA["RA"][:]=self.RA_orig
            self.DicoDATA["DEC"][:]=self.DEC_orig
            GridName="ngrid"
        else:
            print>>log,"Will be working on randomized catalogs"
            self._randomiseCatalog()
            GridName="ngrid_rand"
           
        
        self.DicoDATA["%s"%GridName]=np.zeros((nch,nx,ny),np.float32)
        self.DicoDATA["%s_mask"%GridName]=np.zeros((nch,nx,ny),np.float32)
       
        print>>log,"Compute overdensity grid..."

        for ipix in np.arange(self.rag.size):
            #print ipix
            #self.DicoDATA["ngrid"].flat[ipix]=self._giveDensityAtRaDec(ipix)
            

            APP.runJob("giveDensityAtRaDec:%i"%(ipix), 
                       self._giveDensityAtRaDec,
                       args=(ipix,GridName))#,serial=True)
        APP.awaitJobResults("giveDensityAtRaDec:*", progress="Compute")

    def normaliseCube(self):
        print>>log,"Nomalising cube..."
        nch,_,_=self.DicoDATA["ngrid"].shape
        G=self.DicoDATA["ngrid"]
        M=self.DicoDATA["ngrid_mask"]

        Gr=self.DicoDATA["ngrid_rand"]
        Mr=self.DicoDATA["ngrid_rand_mask"]
        Gr[Mr==1]=0

        for ich in range(nch):
            Mean=np.mean(Gr[ich][Mr[ich]==0])
            Std=np.std(Gr[ich][Mr[ich]==0])
            if Mean==0:
                print>>log,"Channel %i (z=[%3.1f,%3.1f]) is empty"%(ich,self.zg[ich],self.zg[ich+1])
                continue
            G[ich]-=Mean
            G[ich]/=Std

            # Mean=np.mean(G[ich][M[ich]==0])
            # Std=np.std(G[ich][M[ich]==0])
            # if Mean==0:
            #     print>>log,"Channel %i (z=[%3.1f,%3.1f]) is empty"%(ich,self.zg[ich],self.zg[ich+1])
            #     continue
            # G[ich]-=Mean
            # G[ich]/=Mean
            

            
        G[M==1]=0
        
        for ich in range(nch):
            fig=pylab.figure("Overdensity",figsize=(10,10))
            pylab.clf()
            pylab.imshow(G[ich].T[::-1,::-1],interpolation="nearest")
            pylab.draw()
            pylab.show(False)
            pylab.pause(0.1)

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
      
