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
from DDFacet.Other import MyPickle
from DDFacet.Array import shared_dict
import plotly.graph_objects as go
from scipy import interpolate
import scipy.stats
import ClassCatalogMachine
import os
from DDFacet.Array import ModLinAlg

def Sigmoid(x,a=None,MaxVal=None):
    return MaxVal*1./(1.+np.exp(-a*x))

def logit(x0,a=None,MaxVal=None):
    x=x0/MaxVal
    return 1./a*np.log(x/(1.-x))

def computeAngularCovMat(FOV=0.05,CellDeg=0.002):
    
    NPix=int(FOV/CellDeg)
    if (NPix%2)!=0:
        NPix+=1
    log.print("Choosing NPix=%i"%NPix)
    CM=ClassCatalogMachine.ClassCatalogMachine()
    CM.Init()
    
    CACM=ClassAngularCovMat(CellDeg,NPix,CM.zg_Pars)
    CACM.initCovMatrices()

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

# ############################################
# ############################################
# ############################################

class ClassAngularCovMat():
    def __init__(self,
                 CellDeg,
                 NPix,
                 zParms):
        self.CellDeg=CellDeg
        self.NPix=NPix
        self.CellRad=CellDeg*np.pi/180
        self.zg=np.linspace(*zParms)
        self.NSlice=zParms[-1]-1
        self.Mode="Sim3D"
        self.DicoCovSVD = shared_dict.create("DicoCovSVD")

    def initCovMatrices(self,
                        ScaleFWHMkpc=100.):
        
        FileOut="Cov_%s_%ikpc.DicoCov"%(self.Mode,ScaleFWHMkpc)
        
        Parms={"CellDeg":self.CellDeg,
               "NPix":self.NPix,
               "CellRad":self.CellRad,
               "zg":self.zg.tolist()}
        
        if os.path.isfile(FileOut):
            log.print("Loading %s..."%FileOut)
            DicoSave=MyPickle.Load(FileOut)
            if DicoSave["Parms"]==Parms:
                log.print("   all parameters are the same... ")
                self.MaxValueSigmoid=DicoSave["MaxValueSigmoid"]
                self.a_Sigmoid=DicoSave["a_Sigmoid"]
                self.ScaleCov=DicoSave["ScaleCov"]
                self.L_SqrtCov=DicoSave["L_SqrtCov"]
                self.L_NParms=DicoSave["L_NParms"]
                self.L_Hinv=DicoSave["L_Hinv"]
                self.L_ssqs=DicoSave["L_ssqs"]
                self.NParms=np.sum(self.L_NParms)
                log.print("Number of free parameters %i"%self.NParms)
                log.print("  with ScaleCov=%s"%self.ScaleCov)
                self.DicoCov=DicoSave
                return
            
        APP.registerJobHandlers(self)
        AsyncProcessPool.init(ncpu=0,#self.NCPU,
                              affinity=0)
        APP.startWorkers()

        f=2.*np.sqrt(2.*np.log(2.))
        Sig_kpc=ScaleFWHMkpc/f
        self.L_SqrtCov=[]
        self.L_Hinv=[]
        self.L_ssqs=[]
        self.L_NParms=[]
        
        for iz in range(self.NSlice):
            APP.runJob("_computeSVD:%i"%(iz),
                       self._computeSVD,
                       args=(iz,Sig_kpc))#,serial=True)

        APP.awaitJobResults("_computeSVD:*", progress="Compute SqrtCov")
        self.DicoCovSVD.reload()
        for iz in range(self.NSlice):
            sqrtCs=(self.DicoCovSVD["sqrtCs_%i"%iz].copy())
            self.L_Hinv.append(self.DicoCovSVD["Hinv_%i"%iz].copy())
            self.L_ssqs.append(self.DicoCovSVD["ssqs_%i"%iz].copy())
            self.L_SqrtCov.append(sqrtCs)
            self.L_NParms.append(sqrtCs.shape[1])
        self.NParms=np.sum(self.L_NParms)
        log.print("Number of free parameters %i"%self.NParms)
        
        DicoSave={"Parms":Parms,
                  "ScaleCov":self.DicoCovSVD["ScaleCov_%i"%iz],
                  "a_Sigmoid":self.DicoCovSVD["a_Sigmoid_%i"%iz],
                  "MaxValueSigmoid":self.DicoCovSVD["MaxValueSigmoid_%i"%iz],
                  "L_SqrtCov":self.L_SqrtCov,
                  "L_ssqs":self.L_ssqs,
                  "L_NParms":self.L_NParms,
                  "L_Hinv":self.L_Hinv}
        self.ScaleCov=DicoSave["ScaleCov"]
        self.a_Sigmoid=DicoSave["a_Sigmoid"]
        self.MaxValueSigmoid=DicoSave["MaxValueSigmoid"]
        log.print("Saving sqrt(Cov) in %s"%FileOut)
        MyPickle.Save(DicoSave,FileOut)
        self.DicoCov=DicoSave
        
    def toSparse(self,C,Th=1e-2):
        Ca=np.abs(C)
        indi,indj=np.where(Ca>Th*Ca.max())
        data=C[indi,indj]
        M=C.shape[0]
        Cs=scipy.sparse.coo_matrix((data, (indi, indj)), shape=(M, M))
        S=Cs.size/M**2
        return Cs,S

    def _computeSVD(self,iz,Sig_kpc):
        z0,z1=self.zg[iz],self.zg[iz+1]
        zm=(z0+z1)/2.
        SigmaRad=Sig_kpc*cosmo.arcsec_per_kpc_comoving(zm).to_value()/(3600.)*np.pi/180
        self.Scale_Cov_X=None
        if self.Mode=="Gauss":
            stop
            CCM=ClassCovMatrix_Gauss.ClassCovMatrix()
            C=CCM.giveCovMat(CellSizeRad=self.CellRad,
                             NPix=self.NPix,
                             zm=zm,
                             SigmaPix=SigmaRad/self.CellRad)
            
        elif self.Mode=="Sim3D":
            CCM=ClassCovMatrix()
            C=CCM.giveCovMat(CellSizeRad=self.CellRad,
                             NPix=self.NPix,
                             zm=zm)
            
            #self.Scale_Cov_X="log"C.Scale_Cov_X

        #print(":!::")
        #C=np.eye(self.NPix**2,self.NPix**2)
        Cs,Sparsity=self.toSparse(C)
        M=C.shape[0]
        log.print("  Non-zeros are %.1f%% of the matrix size [%ix%i]"%(Sparsity*100,M,M))
        self.Cs=Cs
        CellSize=1.
        A=(CellSize*M)**2
        # k=np.max([A/Sig**2,1.])
        # k=int(k*3)
        k=M-1#int(np.min([M-1,k]))
        log.print("Choosing k=%i [M=%i]"%(k,M))
        self.k=k
        #print(":!::")
        #Us,ss,Vs=scipy.sparse.linalg.svds(Cs,k=k)
        Us,ss,Vs=np.linalg.svd(C)

        sss=ss[ss>0]
        log.print("  log Singular value Max/Min: %5.2f"%(np.log10(sss.max()/sss.min())))
        ssqs=np.sqrt(ss)
        
        ind=np.where(ssqs>0)[0]
        ind=np.where(ssqs>1e-2*ssqs.max())[0]
        S0=Us.shape

        Us=Us[:,ind]
        ssqs=ssqs.flat[ind]
        A= sqrtCs =Us*ssqs.reshape(1,ssqs.size)
        # print(S0,Us.shape,ssqs.shape)
        
        Hinv=ModLinAlg.invSVD((A.T).dot(A))

        self.DicoCovSVD["sqrtCs_%i"%iz]=A
        self.DicoCovSVD["ssqs_%i"%iz]=ssqs
        self.DicoCovSVD["Hinv_%i"%iz]=Hinv
        self.DicoCovSVD["MaxValueSigmoid_%i"%iz]=CCM.MaxValueSigmoid
        self.DicoCovSVD["ScaleCov_%i"%iz]=CCM.ScaleCov
        self.DicoCovSVD["a_Sigmoid_%i"%iz]=CCM.a_Sigmoid
            
    def CubeToVec(self,Cube):
        ii=0
        Cube=self.TypeCube(Cube.copy())
        XOut=np.zeros((self.NParms,),np.float32)
        for iSlice in range(self.NSlice):
            N=self.L_NParms[iSlice]
            if N==0: continue
            Slice=Cube[iSlice].copy()
            Slice[Slice==0]=1e-6
            #y=Slice.reshape((-1,1))-1.
            if self.ScaleCov=="log":
                y=np.log(Slice.reshape((-1,1)))
            elif self.ScaleCov=="Sigmoid":
                y=ClassCovMatrix_Sim3D_2.Logit(x,a_Sigmoid=self.a_Sigmoid,MaxVal=self.MaxValueSigmoid)
            A=self.L_SqrtCov[iSlice]
            x0=(A.T).dot(y)
            Hinv=self.L_Hinv[iSlice]#ModLinAlg.invSVD((A.T).dot(A))
            x=np.dot(Hinv,x0)
            # print(np.mean(np.abs(x.imag)) / np.mean(np.abs(x.real)))
            XOut[ii:ii+N]=(x.real.ravel())[:]
            ii+=N
        return XOut



class ClassCovMatrix():
    def __init__(self,
                 ReComputeFromSim=False):
        self.NCPU=0
        if ReComputeFromSim:
            self.DicoSim = shared_dict.create("DicoSim")
            APP.registerJobHandlers(self)
            self.finaliseInit()
        
    def finaliseInit(self):
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

        fig=pylab.figure("Gamma")
        for i in range(Im.shape[-1])[::10]:
            pylab.clf()
            pylab.imshow(self.Gamma[:,:,i])#,vmin=0.,vmax=10.)
            pylab.colorbar()
            pylab.draw()
            pylab.show(block=False)
            pylab.pause(0.1)
            fig.savefig("Gamma_%i.png"%i)
        

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
