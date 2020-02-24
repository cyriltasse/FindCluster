from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
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
    def __init__(self,CM,GM,ScaleKpc=None):
        self.ScaleKpc=ScaleKpc
        if self.ScaleKpc is None:
            self.ScaleKpc=LikeMachine.ScaleKpc
        self.NScales=len(self.ScaleKpc)
        self.CM=CM
        self.zParms=self.CM.zg_Pars
        self.z_g=np.linspace(*self.zParms)
        self.NCPU=56
        self.logMParms=self.CM.logM_Pars
        self.logM_g=np.linspace(*self.logMParms)
        self.GM=GM
        self.lg,self.mg=self.GM.lg,self.GM.mg
        self.rag,self.decg=self.GM.rag,self.GM.decg
        self.Mm=10**((self.logM_g[0:-1]+self.logM_g[1:])/2.)
        self.dlogM=self.logM_g[1::]-self.logM_g[0:-1]

        self.DicoCube = shared_dict.create("DicoCube")
        nz,nx=self.GM.NSlice,self.GM.NPix
        self.DicoCube["GammaCube0"]=np.zeros((self.NScales,nz,nx,nx),np.float32)
        self.ComputeTheta()
        
        APP.registerJobHandlers(self)
        

    def InitGammaCube(self):
        log.print("Initialise the Gamma cube...")
        GammaCube0=self.DicoCube["GammaCube0"]
        _,nz,nx,_=GammaCube0.shape
        for iScale in range(self.NScales):
            for i in range(nx):
                for j in range(nx):
                    APP.runJob("_estimateGammaAt:%i:%i:%i"%(iScale,i,j),
                               self._estimateGammaAt,
                               args=(iScale,i,j))#,serial=True)
                    #self.GammaCube0[iz,i,j]=self.estimateGammaAt(iz,i,j)

        APP.awaitJobResults("_estimateGammaAt:*", progress="Init Gamma Cube")
        return np.mean(GammaCube0,axis=0)

    def ComputeTheta(self):
        self.Theta=np.zeros((self.NScales,self.z_g.size-1,),np.float32)
        for iScale in range(self.NScales):
            for iz in range(self.z_g.size-1):
                z0,z1=self.z_g[iz],self.z_g[iz+1]
                zm=(z0+z1)/2.
                RadiusRad=self.ScaleKpc[iScale]*cosmo.arcsec_per_kpc_comoving(zm).to_value()/(3600.)*np.pi/180
                dTheta=RadiusRad/100.
                dx,dy=np.mgrid[-3.*RadiusRad:3.*RadiusRad:dTheta,-3.*RadiusRad:3.*RadiusRad:dTheta]
                r=np.sqrt(dx**2+dy**2)
                wt=np.exp(-r**2/(2.*RadiusRad**2))
                wt=wt[r<3.*RadiusRad].flatten()
                self.Theta[iScale,iz]=np.sum(wt)*dTheta**2

    def computeEffectiveOmega(self):
        _,_,nx,ny=self.DicoCube["GammaCube0"].shape
        self.DicoCube["EffectiveOmega"]=np.zeros((self.NScales,self.GM.NSlice,nx,ny),np.float32)
        _,nz,nx,ny=self.DicoCube["EffectiveOmega"].shape
        for iScale in range(self.NScales):
            for i in range(nx):
                for j in range(nx):
                    APP.runJob("_computeEffectiveOmegaAt:%i:%i:%i"%(iScale,i,j),
                               self._computeEffectiveOmegaAt,
                               args=(iScale,i,j))#,serial=True)
                    #self.GammaCube0[iz,i,j]=self.estimateGammaAt(iz,i,j)

        APP.awaitJobResults("_computeEffectiveOmegaAt:*", progress="Init Omega Cube")

    
    def _computeEffectiveOmegaAt(self,iScale,i,j):
        self.DicoCube.reload()
        
        ra,dec=self.rag[i,j],self.decg[i,j]
        xc,yc=self.CM.RaDecToMaskPix(ra,dec)
        if self.CM.MaskArray[0,0,xc,yc]: return
        
        for iz in range(self.z_g.size-1):
            z0,z1=self.z_g[iz],self.z_g[iz+1]
            zm=(z0+z1)/2.
            RadiusRad=self.ScaleKpc[iScale]*cosmo.arcsec_per_kpc_comoving(zm).to_value()/(3600.)*np.pi/180
            self.DicoCube["EffectiveOmega"][iScale,iz,i,j]=self.CM.giveEffectiveOmega(self.rag[i,j],self.decg[i,j],RadiusRad)

                
    def _estimateGammaAt(self,iScale,i,j):
        ra,dec=self.rag[i,j],self.decg[i,j]
        xc,yc=self.CM.RaDecToMaskPix(ra,dec)
        if self.CM.MaskArray[0,0,xc,yc]: return
        for iz in range(self.z_g.size-1):
            self._estimateGammaAtZ(iScale,iz,i,j)
            
    def _estimateGammaAtZ(self,iScale,iz,i,j):
        self.DicoCube.reload()

        l0,m0=self.lg[i,j],self.mg[i,j]
        Cat=self.CM.Cat_s
        Cat=self.CM.Cat
        
        # l=Cat.l
        # m=Cat.m
        
        self.CM.DicoDATA_Shared.reload()
        l=self.CM.DicoDATA_Shared["l"]
        m=self.CM.DicoDATA_Shared["m"]
        
        z=Cat.z
        z0,z1=self.z_g[iz],self.z_g[iz+1]
        zm=(z0+z1)/2.
        
        RadiusRad=self.ScaleKpc[iScale]*cosmo.arcsec_per_kpc_comoving(zm).to_value()/(3600.)*np.pi/180
        
        d=np.sqrt((l-l0)**2+(m-m0)**2)
        
        ind=np.where(d<3.*RadiusRad)[0]
        #ind=np.where((d<3.*RadiusRad)&(z>z0)&(z<z1))[0]
        
        
        if ind.size==0: return 0.
        C=Cat[ind]
        ds=d[ind]
        w=np.exp(-ds**2/(2.*RadiusRad**2))
        #print C.Pzm.shape,np.sum(np.sum(C.Pzm,axis=-1),axis=-1)
        Mt=np.sum(w.reshape(-1,1)*C.Pzm[:,iz,:])#*self.dlogM.reshape((1,-1)))
        
        #Mt=ind.size
        #print self.CM.DicoDATA["DicoSelFunc"]["n_z"][iz]
        Theta=2*np.pi*(1.-np.cos(RadiusRad))

        
        Omega0=self.Theta[iScale,iz]
        # Omega1=self.CM.giveEffectiveOmega(self.rag[i,j],self.decg[i,j],RadiusRad)
        Omega1=self.DicoCube["EffectiveOmega"][iScale,iz,i,j]
        #print Omega1/Omega0
        if Omega1/Omega0>0.3:
            Mt_th=self.CM.DicoDATA["DicoSelFunc"]["n_z"][iz]*Omega1

            # RDeg=RadiusRad*180/np.pi
            # ff=self.CM.giveFracMasked(self.rag[i,j],self.decg[i,j],RDeg)


        
            #print Mt,Mt_th
            self.DicoCube["GammaCube0"][iScale,iz,i,j]=(Mt/Mt_th)#/ff
        else:
            self.DicoCube["GammaCube0"][iScale,iz,i,j]=0.
            
