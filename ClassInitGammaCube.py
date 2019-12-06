from astropy.cosmology import WMAP9 as cosmo
import numpy as np
import scipy.signal
import ClassFFT
import matplotlib.pyplot as pylab
from DDFacet.ToolsDir import ModCoord
from DDFacet.Other import ClassTimeIt
from DDFacet.Other import logger
log = logger.getLogger("ClassInitGammaCube")




class ClassInitGammaCube():
    def __init__(self,LikeMachine):
        self.LM=LikeMachine
        self.ScaleKpc=LikeMachine.ScaleKpc[0]
        self.CM=self.LM.CM
        self.zParms=self.CM.zg_Pars
        self.z_g=np.linspace(*self.zParms)

        self.logMParms=self.CM.logM_Pars
        self.logM_g=np.linspace(*self.logMParms)
        self.GM=self.LM.MassFunction.GammaMachine
        self.lg,self.mg=self.GM.lg,self.GM.mg
        self.Mm=10**((self.logM_g[0:-1]+self.logM_g[1:])/2.)

    def InitGammaCube(self):
        
        nz,nx=self.GM.NSlice,self.GM.NPix
        self.GammaCube0=np.zeros((nz,nx,nx),np.float32)
        for iz in range(nz):
            for i in range(nx):
                for j in range(nx):
                    self.GammaCube0[iz,i,j]=self.estimateGammaAt(iz,i,j)
        return self.GammaCube0
    
    def estimateGammaAt(self,iz,i,j):
        l0,m0=self.lg[i,j],self.mg[i,j]
        l=self.CM.Cat.l
        m=self.CM.Cat.m
        z0,z1=self.z_g[iz],self.z_g[iz+1]
        zm=(z0+z1)/2.
        RadiusRad=self.ScaleKpc*cosmo.arcsec_per_kpc_comoving(zm).to_value()/(3600.)*np.pi/180

        d=np.sqrt((l-l0)**2+(m-m0)**2)
        
        ind=np.where(d<RadiusRad)[0]
        if ind.size==0: return 0.
        C=self.CM.Cat[ind]
        
        Mt=np.sum(C.Pzm[:,iz,:]*self.Mm.reshape((1,-1)))
        Mt_th=self.CM.DicoDATA["DicoSelFunc"]["n_z"][iz]*RadiusRad**2
        return Mt/Mt_th
