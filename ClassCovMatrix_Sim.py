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
from DDFacet.Other import Multiprocessing
from DDFacet.Other import AsyncProcessPool
from DDFacet.Array import shared_dict

def test():
    CCM=ClassCovMatrix(NPix=11,SigmaPix=2.)
    CCM.buildFromCatalog()
    CCM.MeasureCovInMap()
    
class ClassCovMatrix():
    def __init__(self,NPix=11,SigmaPix=2.,Type="Normal"):
        self.NPix=NPix
        self.SigmaPix=SigmaPix
        self.NCPU=0
        self.Type=Type
        self.DicoSim = shared_dict.create("DicoSim")
        
        APP.registerJobHandlers(self)
        self.finaliseInit()
    def finaliseInit(self):
        APP.registerJobHandlers(self)
        AsyncProcessPool.init(ncpu=self.NCPU,
                              affinity=0)
        APP.startWorkers()
        
    def toSparse(self,C,Th=1e-2):
        Ca=np.abs(C)
        indi,indj=np.where(Ca>Th*Ca.max())
        data=C[indi,indj]
        M=C.shape[0]
        Cs=scipy.sparse.coo_matrix((data, (indi, indj)), shape=(M, M))
        S=Cs.size/M**2
        return Cs,S

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

        def give_XYZ():
            r=int(np.random.rand(1)[0]*3)
            if r==0:
                X,Y=C.x, C.y
            elif r==1:
                X,Y=C.y, C.z
            elif r==2:
                X,Y=C.z, C.x

            D=X.max()-X.min()
            X,Y=np.mod(X+np.random.rand(1)[0]*D,D), np.mod(Y +np.random.rand(1)[0]*D,D)
            if int(np.random.rand(1)[0])==0:
                X,Y=Y.copy(),X.copy()
            return X,Y

        LX=[]
        LY=[]
        for iStack in range(NStack):
            X,Y=give_XYZ()
            LX.append(X)
            LY.append(Y)
            
            
        X=np.concatenate(LX)
        Y=np.concatenate(LY)
        
        Nn=10000
        log.print("  Making Delaunay map...")
        Im,self.dx,self.dy=pydtfe.map_dtfe2d(X,Y,xsize=Nn,ysize=Nn)

        Dx=300
        Im=Im[Dx:-Dx,Dx:-Dx]
        self.DicoSim["Gamma"]=Im/np.mean(Im)
        self.Gamma=self.DicoSim["Gamma"]
       
        pylab.clf()
        pylab.imshow(np.log10(self.Gamma))#,vmin=0,vmax=10000)
        pylab.colorbar()
        pylab.draw()
        pylab.show(False)
        pylab.pause(0.1)

    def MeasureCovInMap(self,MaxScaleMpc=2.,Np=1000):
        log.print("Measuring simulation covariance matrix...")
        dx,dy=self.dx,self.dy
        Nn=self.Gamma.shape[0]
        Fact=3
        self.DicoSim["Fact"]=Fact

        log.print("    over max scale of %.2f Mpc..."%MaxScaleMpc)
        log.print("    and NSamples = %i"%Np)
 
        self.DicoSim["R"]=np.arange(0,MaxScaleMpc,dx*Fact)
        NRad=self.DicoSim["R"].size
        xg,yg=np.mgrid[0:Nn,0:Nn]
        self.DicoSim["xg"]=xg*dx
        self.DicoSim["yg"]=yg*dy
        self.DicoSim["G"]=self.Gamma-1.

        

        self.DicoSim["Cov_Sum"]=np.zeros((Np,NRad),np.float32)
        self.DicoSim["Cov_N"]=np.zeros((Np,NRad),np.float32)

        
        
        self.DicoSim["dxdy"]=(dx,dy)
                    # APP.runJob("_estimateGammaAt:%i:%i:%i"%(iScale,i,j),
                    #            self._estimateGammaAt,                    #            args=(iScale,i,j))#,serial=True)


        for iP in range(Np):
            # self._computeCovNAt(iP)
            i,j=int(np.random.rand(1)[0]*Nn),int(np.random.rand(1)[0]*Nn)
            APP.runJob("_computeCovNAt:%i"%(iP),
                       self._computeCovNAt,
                       args=(iP,i,j))#,serial=True)

        APP.awaitJobResults("_computeCovNAt:*", progress="Meas. Cov. Sample")


        
        #Cov_Sum=np.sum(self.DicoSim["Cov_Sum"],axis=0)
        #Cov_N=np.sum(self.DicoSim["Cov_N"],axis=0)
        Cov_Sum=self.DicoSim["Cov_Sum"]
        Cov_N=self.DicoSim["Cov_N"]
        Cov=np.median(Cov_Sum/Cov_N,axis=0)
        R=self.DicoSim["R"]

        FileOut="%s.RadialCov.npz"%self.CatFile
        log.print("Saving radial Cov in: %s"%FileOut)
        np.savez(FileOut,
                 R=R,
                 Cov=Cov,
                 Cov_Sum=self.DicoSim["Cov_Sum"],
                 Cov_N=self.DicoSim["Cov_N"])
        
        pylab.clf()
        pylab.plot(R,Cov)
        pylab.draw()
        pylab.show(False)
        pylab.pause(0.1)

    def _computeCovNAt(self,iP,i,j):
        T=ClassTimeIt.ClassTimeIt()
        T.disable()
        self.DicoSim.reload()
        (dx,dy)=self.DicoSim["dxdy"]
        Fact=self.DicoSim["Fact"]
        R=self.DicoSim["R"]
        self.Gamma=self.DicoSim["Gamma"]
        G=self.DicoSim["G"]
        xg=self.DicoSim["xg"]
        yg=self.DicoSim["yg"]
        Nn=self.Gamma.shape[0]
        
        Dx=int(R.max()/dx)
        Dy=int(R.max()/dy)
        Nn=self.Gamma.shape[0]
        i0=np.max([0,i-Dx])
        i1=np.min([Nn,i+Dx])
        j0=np.max([0,j-Dy])
        j1=np.min([Nn,j+Dy])
        xgs=xg[i0:i1,j0:j1]
        ygs=yg[i0:i1,j0:j1]
        d=np.sqrt((i*dx-xgs)**2+(j*dy-ygs)**2)
        Gs=G[i0:i1,j0:j1]
        T.timeit("Compute d")
        for iR in range(R.size):
            # print(iP,iR)
            rm=R[iR]
            r0=rm-dx/2.*Fact
            r1=rm+dx/2.*Fact
            indx,indy=np.where((d>=r0) & (d<r1))
            Nsx,Nsy=Gs.shape
            Cx=((indx>=0)&(indx<Nsx))
            Cy=((indy>=0)&(indy<Nsy))
            Cxy=(Cx&Cy)
            indx=indx[Cxy]
            indy=indy[Cxy]
            T.timeit("  Where ")
            self.DicoSim["Cov_Sum"][iP,iR]+=np.sum(G[i,j]*Gs[indx,indy])
            self.DicoSim["Cov_N"][iP,iR]+=indx.size
            T.timeit("  Rest ")


        
                                   
