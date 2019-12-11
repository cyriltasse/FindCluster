from astropy.cosmology import WMAP9 as cosmo
import numpy as np
import scipy.signal
import ClassFFT
import matplotlib.pyplot as pylab
from DDFacet.ToolsDir import ModCoord
from DDFacet.Other import ClassTimeIt
from DDFacet.Other import logger
log = logger.getLogger("ClassInitGammaCube")

from DDFacet.Other.AsyncProcessPool import APP, WorkerProcessError
from DDFacet.Other import AsyncProcessPool
from DDFacet.Array import shared_dict



class ClassInitGammaCube():
    def __init__(self,LikeMachine):
        self.LM=LikeMachine
        self.ScaleKpc=LikeMachine.ScaleKpc[0]
        self.CM=self.LM.CM
        self.zParms=self.CM.zg_Pars
        self.z_g=np.linspace(*self.zParms)
        self.NCPU=56
        self.logMParms=self.CM.logM_Pars
        self.logM_g=np.linspace(*self.logMParms)
        self.GM=self.LM.MassFunction.GammaMachine
        self.lg,self.mg=self.GM.lg,self.GM.mg
        self.Mm=10**((self.logM_g[0:-1]+self.logM_g[1:])/2.)
        self.dlogM=self.logM_g[1::]-self.logM_g[0:-1]

        self.DicoCube = shared_dict.create("DicoCube")
        nz,nx=self.GM.NSlice,self.GM.NPix
        self.DicoCube["GammaCube0"]=np.zeros((nz,nx,nx),np.float32)
        self.finaliseInit()
        
    def finaliseInit(self):
        APP.registerJobHandlers(self)
        AsyncProcessPool.init(ncpu=self.NCPU,
                              affinity=0)
        APP.startWorkers()

    def InitGammaCube(self):
        print>>log,"Initialise the Gamma cube..."
        GammaCube0=self.DicoCube["GammaCube0"]
        nz,nx,_=GammaCube0.shape

        for iz in range(nz):
            for i in range(nx):
                for j in range(nx):
                    APP.runJob("_estimateGammaAt:%i:%i:%i"%(iz,i,j),
                               self._estimateGammaAt,
                               args=(iz,i,j))
                    #self.GammaCube0[iz,i,j]=self.estimateGammaAt(iz,i,j)

        APP.awaitJobResults("_estimateGammaAt:*", progress="Init Gamma Cube")
        return GammaCube0
    
    def _estimateGammaAt(self,iz,i,j):
        l0,m0=self.lg[i,j],self.mg[i,j]
        l=self.CM.Cat.l
        m=self.CM.Cat.m
        z=self.CM.Cat.z
        z0,z1=self.z_g[iz],self.z_g[iz+1]
        zm=(z0+z1)/2.
        RadiusRad=self.ScaleKpc*cosmo.arcsec_per_kpc_comoving(zm).to_value()/(3600.)*np.pi/180

        d=np.sqrt((l-l0)**2+(m-m0)**2)
        
        ind=np.where(d<3.*RadiusRad)[0]
        #ind=np.where((d<3.*RadiusRad)&(z>z0)&(z<z1))[0]
        
        
        if ind.size==0: return 0.
        C=self.CM.Cat[ind]
        ds=d[ind]
        w=np.exp(-ds**2/(2.*RadiusRad**2))
        #print C.Pzm.shape,np.sum(np.sum(C.Pzm,axis=-1),axis=-1)
        Mt=np.sum(w.reshape(-1,1)*C.Pzm[:,iz,:])#*self.dlogM.reshape((1,-1)))
        
        #Mt=ind.size
        #print self.CM.DicoDATA["DicoSelFunc"]["n_z"][iz]
        Theta=2*np.pi*(1.-np.cos(RadiusRad))
        Mt_th=self.CM.DicoDATA["DicoSelFunc"]["n_z"][iz]*Theta

        self.DicoCube["GammaCube0"][iz,i,j]=Mt/Mt_th

