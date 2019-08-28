import numpy as np
import pylab
import tables
import astropy.io.fits as pyfits
from astropy.cosmology import WMAP9 as cosmo
import tables
import matplotlib.pyplot as pylab

from DDFacet.Other import MyLogger
log=MyLogger.getLogger("DynSpecMS")
from DDFacet.Array import shared_dict
from DDFacet.Other.AsyncProcessPool import APP, WorkerProcessError
from DDFacet.Other import Multiprocessing
from DDFacet.Other import ModColor
from DDFacet.Other.progressbar import ProgressBar
from DDFacet.Other import AsyncProcessPool

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
    def __init__(self,ra,dec,boxDeg,NPix=31,ScaleKpc=50,z=[0.01,2.,20],NCPU=0):
        self.rac=ra
        self.decc=dec
        self.boxDeg=boxDeg
        self.NPix=NPix
        self.ScaleKpc=ScaleKpc
        self.zg=np.linspace(*z)
        self.rag,self.decg=np.mgrid[self.rac-self.boxDeg:self.rac+self.boxDeg:1j*NPix,
                                    self.decc-self.boxDeg:self.decc+self.boxDeg:1j*NPix]

        self.NCPU=NCPU


        self.DicoDATA = shared_dict.create("DATA")

    def finaliseInit(self):
        APP.registerJobHandlers(self)
        AsyncProcessPool.init(ncpu=self.NCPU,
                              affinity=0)
        APP.startWorkers()

    def setCat(self,CatName):
        self.Cat=pyfits.open(CatName)[1].data
        self.Cat=self.Cat.view(np.recarray)
        
        ind=np.where((self.Cat.FLAG_CLEAN == 1)&(self.Cat.i_fluxerr > 0)&
                     (self.Cat.ch2_swire_fluxerr > 0))[0]

        # pylab.clf()
        # pylab.scatter(self.Cat.RA[::111],self.Cat.DEC[::111],s=3,c="black")

        self.Cat=self.Cat[ind]
        
        # pylab.scatter(self.Cat.RA[::111],self.Cat.DEC[::111],s=10,c="red")
        # pylab.show()
        

        
        self.DicoDATA["RA"]=self.Cat.RA[:]
        self.DicoDATA["DEC"]=self.Cat.DEC[:]

    def setPz(self,PzFile):
        H=tables.open_file(PzFile)
        self.DicoDATA["zgrid_pz"]=H.root.zgrid[:]
        self.DicoDATA["pz"]=H.root.Pz[:]
        H.close()
        
    def _giveDensityAtRaDec(self,ipix):
        self.DicoDATA.reload()
        ra,dec=self.rag.flat[ipix],self.decg.flat[ipix]
        print ra,dec
        RA,DEC=self.DicoDATA["RA"],self.DicoDATA["DEC"]
        D=AngDist(ra,dec,RA,DEC)
        pz=self.DicoDATA["pz"]
        n=0
        zgrid_pz=self.DicoDATA["zgrid_pz"]
        for iz in range(self.zg.size-1):
            z0,z1=self.zg[iz],self.zg[iz+1]
            zm=(z0+z1)/2.
            indz=np.where((zgrid_pz>z0)&(zgrid_pz<z1))[0]
            R=cosmo.arcsec_per_kpc_comoving(zm).to_value()*self.ScaleKpc/3600.
            ind=np.where(D<R)[0]
            #if ind.size==0: continue
            #if indz.size==0: continue
            PP=pz[ind][:,indz].flatten()
            indnan=np.logical_not(np.isnan(PP))
            
            n+=np.sum(PP[indnan])
            if np.isnan(n): stop
            #print n
            
        self.DicoDATA["ngrid"].flat[ipix]=n

    def giveDensityGrid(self):

        self.DicoDATA["ngrid"]=np.zeros(self.rag.shape,np.float32)
        print self.DicoDATA["ngrid"]
        for ipix in np.arange(self.rag.size):
            #print ipix
            #self.DicoDATA["ngrid"].flat[ipix]=self._giveDensityAtRaDec(ipix)
            

            APP.runJob("giveDensityAtRaDec:%i"%(ipix), 
                       self._giveDensityAtRaDec,
                       args=(ipix,))#,serial=True)
        APP.awaitJobResults("giveDensityAtRaDec:*", progress="Compute")

        pylab.clf()
        pylab.imshow(self.DicoDATA["ngrid"],interpolation="nearest")
        pylab.show()
        
        
def test():

    Cat="/data/tasse/DataDeepFields/EN1/EN1_opt_spitzer_merged_vac_opt3as_irac4as_all_hpx_public.fits"
    Pz="/data/tasse/DataDeepFields/EN1/EN1_opt_spitzer_merged_vac_opt3as_irac4as_all_public_pz.hdf"
    rac,decc=241.25047,55.624223
    COM=ClassOverdensityMap(rac,decc,2.)
    COM.setCat(Cat)
    COM.setPz(Pz)
    COM.finaliseInit()
    #COM.giveDensityAtRaDec(rac,decc)
    COM.giveDensityGrid()
