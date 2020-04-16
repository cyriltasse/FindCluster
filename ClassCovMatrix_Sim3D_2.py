from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from astropy.cosmology import WMAP9 as cosmo
import numpy as np
import scipy.signal
import matplotlib.pyplot as pylab
from DDFacet.ToolsDir import ModCoord
from DDFacet.Other import ClassTimeIt
from DDFacet.Other import logger
log = logger.getLogger("ClassCovMatrix_Sim")
import scipy.sparse
import scipy.sparse.linalg
import scipy.signal

import scipy.stats
import pydtfe

from DDFacet.Other.AsyncProcessPool import APP, WorkerProcessError
from DDFacet.Other import AsyncProcessPool
from DDFacet.Array import shared_dict
import plotly.graph_objects as go
from scipy import interpolate
import scipy.stats


def Sigmoid(x,a=None,MaxVal=None):
    return MaxVal*1./(1.+np.exp(-a*x))

def logit(x0,a=None,MaxVal=None):
    x=x0/MaxVal
    return 1./a*np.log(x/(1.-x))

def measureCovMat():
    CCM=ClassCovMatrix(ReComputeFromSim=True)
    CCM.buildFromCatalog()
    CCM.MeasureCovInMap()
    # CellSizeRad=0.001*np.pi/180
    # zz=np.linspace(0.01,1.5,10)
    # for z in zz:
    #     CCM.giveCovMat(CellSizeRad=CellSizeRad,NPix=50,zm=z)


def PlotCovMat():
    CCM=ClassCovMatrix()
    # CCM.buildFromCatalog()
    # CCM.MeasureCovInMap()
    CellSizeRad=0.001*np.pi/180
    zz=np.linspace(0.01,1.5,10)
    for z in zz:
        CCM.giveCovMat(CellSizeRad=CellSizeRad,NPix=50,zm=z)
    
class ClassCovMatrix():
    def __init__(self,
                 ReComputeFromSim=False):
        self.NCPU=0
        if ReComputeFromSim:
            self.DicoSim = shared_dict.create("DicoSim")
            APP.registerJobHandlers(self)
            self.finaliseInit()
        
    def finaliseInit(self):
        APP.registerJobHandlers(self)
        AsyncProcessPool.init(ncpu=self.NCPU,
                              affinity=0)
        APP.startWorkers()
        
    def buildFromCatalog(self,dz=0.1):
        File="/data/tasse/DataDeepFields/Millenium_z_0.5_0.63.txt"
        self.CatFile=File
        log.print("Reading %s"%File)
        C=np.genfromtxt(File,dtype=[("galaxyID",np.float32),("x",np.float32),("y",np.float32),("z",np.float32),("redshift",np.float32),("snapnum",np.float32),("stellarMass",np.float32)],delimiter=",")
        C=C.view(np.recarray)
        h=0.7
        C.x*=h
        C.y*=h
        C.z*=h

        C=C[np.where(np.logical_not(np.isnan(C.x)))]
        zz=np.unique(C.redshift)
        z0=zz[0]
        log.print("  Selecting sources at z = %f"%z0)
        C=C[C.redshift==z0]
        z1=z0+dz

        log.print("  Depth in the line of sight %.4f Mpc"%(C.z.max()-C.z.min()))
        a=cosmo.comoving_distance([0.5, 0.6])
        dStack=a.value[1]-a.value[0]

        NStack=int(dStack/(C.z.max()-C.z.min()))
        NStack=1
        log.print("  Making %i slices for dz=%.2f"%(NStack,dz))

        X,Y,Z=C.x, C.y, C.z

        
        Nn=500
        dx=0.05#025 # Mpc
        x0,x1=X.min(),X.max()
        y0,y1=Y.min(),Y.max()
        z0,z1=Z.min(),Z.max()
        
        DD=np.max([x1-x0,y1-y0,z1-z0])
        Nn=int(DD/dx)
        xc=np.mean(X) # np.random.rand(1)[0]*(x1-x0)+x0
        yc=np.mean(Y) # np.random.rand(1)[0]*(y1-y0)+y0
        zc=np.mean(Z) # np.random.rand(1)[0]*(z1-z0)+z0
        Dx=Nn*dx
        # Cx=(X>xc-Dx/2.)&(X<xc+Dx/2.)
        # Cy=(Y>yc-Dx/2.)&(Y<yc+Dx/2.)
        # Cz=(Z>zc-Dx/2.)&(Z<zc+Dx/2.)
        Cx=(X>x0)&(X<x0+Dx)
        Cy=(Y>y0)&(Y<y0+Dx)
        Cz=(Z>z0)&(Z<z0+Dx)
        
        C=Cx&Cy&Cz
        Xp=X[C].copy()
        Yp=Y[C].copy()
        Zp=Z[C].copy()
        log.print("  Making Delaunay map...")
        #Xp-=xc
        #Yp-=yc
        #Zp-=zc

        #self.xg_1d = R = np.linspace(-Dx/2.,Dx/2., Nn)
        self.xg_1d = R = np.linspace(0,Dx, Nn)
        x_m, y_m, z_m = np.meshgrid(R, R, R)
        
        Im=pydtfe.map_dtfe3d(Xp,Yp,Zp,x_m, y_m, z_m)
        log.print("    done!")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        Im[np.isnan(Im)]=0.

        MinNonZero=(Im[Im>0]).min()
        Im[Im<=0]=MinNonZero
        

            
        #Dx=300
        #Im=Im[Dx:-Dx,Dx:-Dx]

        Im/=np.mean(Im)


        SigMAD=scipy.stats.median_absolute_deviation
        Sig=SigMAD(Im.flat[:])

        # self.a_Sigmoid=1./Sig
        # self.MaxValue=Im.max()*1.1
        # self.ScaleCov="Sigmoid"
        # self.DicoSim["Gamma"]=logit(Im,MaxVal=self.MaxValue,a=self.a_Sigmoid)
        
        self.ScaleCov="log"
        self.MaxValue=None
        self.a_Sigmoid=None
        self.DicoSim["Gamma"]=np.log(Im)

        
        self.DicoSim["G"]=self.DicoSim["Gamma"]-np.mean(self.DicoSim["Gamma"])
        self.Gamma=self.DicoSim["Gamma"]
        
        for i in range(Im.shape[-1])[::10]:
            pylab.clf()
            pylab.imshow(self.Gamma[:,:,i])#,vmin=0.,vmax=10.)
            pylab.colorbar()
            pylab.draw()
            pylab.show(False)
            pylab.pause(0.3)
        

    def MeasureCovInMap(self):
        log.print("Measuring simulation covariance matrix...")
        # dx,dy,dz=self.dx,self.dy,self.dz
        Nn=self.Gamma.shape[0]
        self.DicoSim["R"]=self.xg_1d-self.xg_1d.min()
        NRad=self.DicoSim["R"].size

        Np=10000
        iJob=0
        for iWorker in range(APP.ncpu):
            LJob=[]
            for iP in range(Np):
                i,j=int(np.random.rand(1)[0]*Nn),int(np.random.rand(1)[0]*Nn)
                Ax=int(np.random.rand(1)[0]*3)
                LJob.append((Ax,i,j))
#            self._computeCovNAt(LJob)
            APP.runJob("_computeCovNAt:%i"%(iWorker),
                       self._computeCovNAt,
                       args=(LJob,))#,serial=True)
            iJob+=1
        results = APP.awaitJobResults("_computeCovNAt:*", progress="Meas. Cov. Sample")
        
        G=self.DicoSim["G"]
        nx,_,_=G.shape
        Cov_Sum=np.zeros((APP.ncpu,nx,nx),np.float32)
        N=0
        iJob=0
        for Ci,Ni in results:
            Cov_Sum[iJob]=Ci
            N+=Ni
            iJob+=1
            
        Cov=np.sum(Cov_Sum,axis=0)/N

        Cov[Cov<=0.]=1e-6

        
        Cov1d=np.zeros((nx,),np.float32)
        for i in range(nx):
            c=Cov.flat[i::nx+1][0:nx-i]
            Cov1d[i]=np.mean(c)


        R=self.DicoSim["R"]
        
        pylab.clf()
        pylab.subplot(1,2,1)
        pylab.imshow(Cov)
        pylab.colorbar()
        pylab.subplot(1,2,2)
        pylab.plot(R,Cov1d)
        pylab.draw()
        pylab.show(False)
        pylab.pause(0.1)

        FileOut="%s.RadialCov.npz"%self.CatFile
        log.print("Saving radial Cov in: %s"%FileOut)
        np.savez(FileOut,
                 R=R,
                 Cov1d=Cov1d,
                 MaxValue=self.MaxValue,
                 a_Sigmoid=self.a_Sigmoid,
                 ScaleCov=self.ScaleCov)
        

    def _computeCovNAt(self,LJobs):
        T=ClassTimeIt.ClassTimeIt()
        T.disable()
        self.DicoSim.reload()
        
        G=self.DicoSim["G"]
        nx,_,_=G.shape
        Cov_Sum=np.zeros((nx,nx),np.float32)
        N=0
        for Ax,i,j in LJobs:
            if Ax==0:
                x=G[:,i,j]
            elif Ax==1:
                x=G[i,:,j]
            elif Ax==2:
                x=G[i,j,:]
            Cov_Sum+=x.reshape((-1,1))*x.reshape((1,-1))
            N+=1
        return Cov_Sum,N

        
    def giveCovMat(self,CellSizeRad=None,NPix=None,zm=None):
        
        S=np.load("/data/tasse/DataDeepFields/Millenium_z_0.5_0.63.txt.RadialCov.npz",allow_pickle=True)
        R=S["R"]
        self.MaxValueSigmoid=S["MaxValue"][()]
        self.a_Sigmoid=S["a_Sigmoid"][()]
        self.ScaleCov=S["ScaleCov"][()]
        
        Cov1d=S["Cov1d"]
        # R=R[1::]
        # Cov1d=Cov1d[1::]

        #R+=1e-2
        
        #R=np.log10(R)
        #LogCov=Cov1d#np.log10(Cov1d)
        RCut=R[np.where(Cov1d==Cov1d.min())[0]]
        # ind=np.where(R<RCut)[0]
        # R_s=R[ind]
        # Cov_s=Cov[ind]

        Cov1d[R>RCut]=0.
        
        # c=np.polyfit(R_s, Cov_s, 3)
        # p = np.poly1d(c)
        # y=p(R_s)


        Rkpc=R*1e3
        Rrad=Rkpc*cosmo.arcsec_per_kpc_comoving(zm).to_value()/(3600.)*np.pi/180
        xg,yg=np.mgrid[0:NPix,0:NPix]*CellSizeRad
        f = interpolate.interp1d(Rrad, Cov1d)



        
        d=np.sqrt( (xg.reshape((-1,1))-xg.reshape((1,-1)))**2 + (yg.reshape((-1,1))-yg.reshape((1,-1)))**2 )
        Cov = f(d)   # use interpolation function returned by `interp1d`

        # # Plot 1d Cov
        # d1d=np.sqrt( (xg.reshape((-1,1))-xg.reshape((1,-1)))**2)
        # Cov1d = f(d1d)   # use interpolation function returned by `interp1d`
        # print(zm)
        # pylab.clf()
        # pylab.imshow(Cov1d)
        # pylab.title(zm)
        # pylab.colorbar()
        # #pylab.plot(d1d,Cov1d)
        # pylab.draw()
        # pylab.show(False)
        # pylab.pause(0.1)
        
        
        return Cov
