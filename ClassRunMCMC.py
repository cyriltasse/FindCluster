import os
os.environ["OMP_NUM_THREADS"] = "1"
from multiprocessing import Pool
#import emcee
import numpy as np
import matplotlib.pyplot as plt
import GeneDist
import ClassSimulCatalog
import matplotlib.pyplot as pylab
import ClassMassFunction
from astropy.cosmology import WMAP9 as cosmo
from DDFacet.Other import logger
log = logger.getLogger("RunMCMC")

from DDFacet.Other.AsyncProcessPool import APP, WorkerProcessError
from DDFacet.Other import Multiprocessing
from DDFacet.Other import ModColor
from DDFacet.Other import ClassTimeIt
from DDFacet.Other.progressbar import ProgressBar
from DDFacet.Other import AsyncProcessPool
from DDFacet.Array import shared_dict
import ClassCatalogMachine
import ClassCatalogMachine2

def g_z(z):
    a=4.
    g=np.zeros_like(z)
    ind=np.where((z>(1./a))&(z<a))[0]
    g[ind]=1./z[ind]
    return g


class ClassRunMCMC():
    def __init__(self):
        rac,decc=241.20678,55.59485 # cluster
        self.CellDeg=0.001
        self.CellRad=self.CellDeg*np.pi/180
        self.NPix=101
        self.ScaleKpc=500
        self.NCPU=56
        self.MoveType="Stretch"
        self.MoveType="Walk"
        self.SWalk=3
        np.random.seed(6)
        #self.XSimul=np.random.randn(9)*2
        #self.XSimul=np.random.randn(1)
        #self.XSimul.fill(0)
        #self.XSimul[0]=10.
        self.rac_deg,self.decc_deg=rac,decc
        self.rac,self.decc=rac*np.pi/180,decc*np.pi/180
        self.zParms=[0.6,0.7,2]
        self.logMParms=[10,10.5,2]
        self.logM_g=np.linspace(*self.logMParms)

        # #################################################################
        # # Simulate Catalog
        # CSC=ClassSimulCatalog.ClassSimulCatalog(rac,decc,
        #                                         # z=[0.01,2.,40],
        #                                         z=self.zParms,
        #                                         ScaleKpc=self.ScaleKpc,
        #                                         CellDeg=self.CellDeg,
        #                                         NPix=self.NPix,
        #                                         #XSimul=self.XSimul,
        #                                         logMParms=self.logMParms)
        # CSC.doSimul()
        # self.CSC=CSC
        # self.Cat=CSC.Cat
        # #################################################################

        self.LoadData()

        #self.XSimul
        #CSC.MassFunc.CGM.NParms

        self.MassFuncLogProb=None


        

        
        self.DistMachine=GeneDist.ClassDistMachine()
        z=np.linspace(-10,10,1000)
        g=g_z(z)
        G=np.cumsum(g)
        G/=G[-1]
        self.DistMachine.setCumulDist(z,G)
        self.z0z1=CSC.zg[0],CSC.zg[1]

        self.NDim = self.CSC.MassFunc.CGM.NParms
        self.NChain = 8*self.NDim
        self.reinitDicoChains()
        
        self.DE_F=2.
        self.DE_CR=0.5
        
        self.finaliseInit()
        
    def LoadData(self,Show=False,DicoDataNames=DicoDataNames_EN1,ForceLoad=False):
        
        # CM=ClassCatalogMachine.ClassCatalogMachine(self.rac,self.decc,
        #                                            CellDeg=self.CellDeg,
        #                                            NPix=self.NPix,
        #                                            ScaleKpc=self.ScaleKpc)

        CM=ClassCatalogMachine2.ClassCatalogMachine()

        # if Show:
        #     CM.showRGB()
            
        if os.path.isfile(DicoDataNames["PickleSave"]) and not ForceLoad:
            CM.PickleLoad(DicoDataNames["PickleSave"])
        else:
            CM.setMask(DicoDataNames["MaskImage"])
            CM.setPhysCatalog(DicoDataNames["PhysCat"])
            CM.setCat(DicoDataNames["PhotoCat"])
            CM.setPz(DicoDataNames["PzCat"])
            CM.PickleSave(DicoDataNames["PickleSave"])
            
        self.CM=CM
        # CM.SaveFITS(Name=NameOut)

    def reinitDicoChains(self,XInit=None):
        self.DicoChains = shared_dict.create("DicoChains")
        if XInit is None:
            XInit=np.random.randn(self.NChain,self.NDim)*1e-1#np.mean(np.abs(self.CSC.XSimul))
            XInit[:,0]=10
        self.DicoChains["X"]=XInit
        # self.DicoChains["X"]=self.CSC.XSimul+np.random.randn(self.NChain,self.NDim)*1.e-2
        # self.X=self.CSC.XSimul+np.random.randn(self.NChain,self.NDim)*1.
        
        self.DicoChains["X1"]=self.DicoChains["X"].copy()
        self.DicoChains["L"]=np.array([self.log_prob(x) for x in self.DicoChains["X"]])
        self.DicoChains["L1"]=self.DicoChains["L"].copy()
        self.DicoChains["Accepted"]=np.zeros((self.NChain,),int)

        self.X=self.DicoChains["X"]
        self.X1=self.DicoChains["X1"]
        self.L=self.DicoChains["L"]
        self.L1=self.DicoChains["L1"]
        self.Accepted=self.DicoChains["Accepted"]

        

        
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

    # def log_prob(self, x):
    #     z0,z1=self.z0z1
    #     X=self.Cat.x
    #     Y=self.Cat.y
    #     ni=self.Cat.n_i
    #     iz=self.Cat.iz
    #     GammaSlice=self.CSC.CGM.SliceFunction(x,z0,z1)
    #     nx,ny=GammaSlice.shape
    #     ind=nx*ny*iz+ny*X+Y
    #     G=GammaSlice.flat[ind]
    #     n=G*ni
    #     L=self.CellRad**2*(np.sum(GammaSlice))+np.sum(np.log(n))
    #     return L

    def log_prob(self, x):
        z0,z1=self.z0z1
        X=self.Cat.x
        Y=self.Cat.y
        
        ni=self.Cat.n_i
        iz=self.Cat.iz
        T=ClassTimeIt.ClassTimeIt()
        T.disable()
        # ######################################
        # Init Mass Function for that one X
        if self.MassFuncLogProb is None:
            self.MassFuncLogProb=ClassMassFunction.ClassMassFunction()
            T.timeit("MassFunc")
            self.MassFuncLogProb.setGammaFunction((self.rac,self.decc),
                                                  self.CellDeg,
                                                  self.NPix,
                                                  z=self.zParms,
                                                  ScaleKpc=self.ScaleKpc)
        MassFunc=self.MassFuncLogProb
        T.timeit("SetGammaFunc")
        LX=[x]
        MassFunc.CGM.computeGammaCube(LX)
        GammaSlice=MassFunc.CGM.GammaCube[0]
        T.timeit("ComputeGammaFunc")
        # ######################################

        OmegaSr=((1./3600)*np.pi/180)**2
        L=0
        for iLogM in range(self.logM_g.size-1):
            logM0,logM1=self.logM_g[iLogM],self.logM_g[iLogM+1]
            logMm=(logM0+logM1)/2.
            zm=(z0+z1)/2.
            dz=z1-z0
            dlogM=logM1-logM0
            dV_dz=cosmo.differential_comoving_volume(zm).to_value()
            T.timeit("Cosmo")
            V=dz*dV_dz*self.CellRad**2
            
            n0=MassFunc.givePhiM(zm,logMm)*dlogM*V
            T.timeit("n0")
            L+=-np.sum(GammaSlice)*n0
            for iS in range(self.Cat.ra.size):
                n=MassFunc.give_N((self.Cat.ra[iS],self.Cat.dec[iS]),
                                  (z0,z1),
                                  (logM0,logM1),
                                  OmegaSr)
                L+=np.log(n)
                #L+=np.log(OmegaSr)
            T.timeit("log(n)")
        return L

    # ##########################################################
    # Global Minima solver
    
    def _evolveDifferentialEvolution(self,iChain):
        kArray=-1*np.ones((3,),np.int32)
        
        for iS in range(3):
            while True:
                k=int(np.random.rand(1)[0]*self.NChain)
                if (k!=iChain)&(k not in kArray):
                    kArray[iS]=k
                    break

        R=int(np.random.rand(1)[0]*self.NDim)
        r=np.random.rand(self.NDim)
        a=self.X[kArray[0]]
        b=self.X[kArray[1]]
        c=self.X[kArray[2]]
        
        y=self.X[iChain].copy()
        ye=a+self.DE_F*(b-c)
        y[R]=ye[R]
        y[r<self.DE_CR]=ye[r<self.DE_CR]
        L1=self.log_prob(y)
        L0=self.L[iChain]
        if L1>=L0:
            self.X[iChain][:]=y[:]
            self.L[iChain]=L1
            
    def runDifferentialEvolution(self):
        
        GammaCube=self.CSC.MassFunc.CGM.GammaCube
        ra0,ra1=self.CSC.rag.min(),self.CSC.rag.max()
        dec0,dec1=self.CSC.decg.min(),self.CSC.decg.max()
        self.extent=ra0,ra1,dec0,dec1
        
        #pylab.ion()
        vmin=0
        vmax=np.max(GammaCube[0])
        self.vminmax=(vmin,vmax)
        pylab.figure("Simul")
        pylab.clf()
        pylab.imshow(GammaCube[0].T[::-1,:],extent=(ra0,ra1,dec0,dec1),cmap="cubehelix",vmin=vmin,vmax=vmax)
        #s=self.Cat.logM
        #s0,s1=s.min(),s.max()
        s=10.#(s-s0)/(s1-s0)*20+5.
        pylab.scatter(self.CSC.Cat.ra,self.CSC.Cat.dec,s=s,linewidths=0)
        #pylab.colorbar()
        pylab.draw()
        pylab.show(False)
        pylab.pause(0.1)

        
        
        self.ListX=[]
        self.ListL=[]
        iDone=0
        
        while True:
            for iChain in range(self.NChain):
                APP.runJob("_evolveAgent:%i"%(iChain), 
                           self._evolveDifferentialEvolution,
                           args=(iChain,))#,serial=True)
            APP.awaitJobResults("_evolveAgent:*", progress="Compute step %i"%iDone)
            #self.X[:]=self.X1[:]
            #self.L[:]=self.L1[:]
            if iDone%10==0: self.PlotChains(OutName="Test%4.4i.png"%iDone)
            iDone+=1

    # ##########################################################

    def _evolveChainWalk(self,iChain,Sigma):
        kArray=-1*np.ones((self.SWalk,),np.int32)
        
        for iS in range(self.SWalk):
            while True:
                k=int(np.random.rand(1)[0]*self.NChain)
                H=self.NChain//2
                if iChain>=H:
                    Cond1=(k<H)
                else:
                    Cond1=(k>=H)
                if (k!=iChain)&(Cond1)&(k not in kArray):
                    kArray[iS]=k
                    break

        Xs=self.X[kArray]
        meanXs=np.mean(Xs,axis=0)
        Xs0=Xs-meanXs.reshape((1,-1))
        covXs=np.dot(Xs0.T,Xs)/self.SWalk

        X0=self.X[iChain]
        try:
            X2=X0+Sigma*np.random.multivariate_normal(np.zeros((X0.size,),np.float32),covXs,1)[0]
        except:
            X2=X0
            
        L1=self.log_prob(X2)
        L0=self.L[iChain]
        #
        dd=L1-L0
        if dd>10.:
            pp=1.
        else:
            pp=np.exp(L1-L0)
        p=np.min([1,pp])
        #print "   %f, %f --> %f"%(L1,L0,p)
        r=np.random.rand(1)[0]
        if r<p:
            self.X1[iChain]=X2[:]
            self.L1[iChain]=L1
            self.Accepted[iChain]=1
        else:
            self.X1[iChain]=X0[:]
            self.L1[iChain]=L0
            self.Accepted[iChain]=0

    def _evolveChainStretch(self,iChain,Sigma):
        while True:
            k=int(np.random.rand(1)[0]*self.NChain)
            H=self.NChain//2
            if iChain>=H:
                Cond1=(k<H)
            else:
                Cond1=(k>=H)
            if (k!=iChain)&(Cond1): break
                
        z=self.DistMachine.GiveSample(1)[0]
        X0=self.X[iChain]
        X1=self.X[k]
        
        X2=X0+z*(X1-X0)
            
        L1=self.log_prob(X2)
        L0=self.L[iChain]
        #
        dd=L1-L0+(self.NDim-1)*np.log(z)
        if dd>10.:
            pp=1.
        else:
            pp=np.exp(L1-L0)
        p=np.min([1,pp])
        #print "   %f, %f --> %f"%(L1,L0,p)
        r=np.random.rand(1)[0]
        if r<p:
            self.X1[iChain]=X2[:]
            self.L1[iChain]=L1
            self.Accepted[iChain]=1
        else:
            self.X1[iChain]=X0[:]
            self.L1[iChain]=L0
            self.Accepted[iChain]=0





    def runMCMC(self):
        
        GammaCube=self.CSC.MassFunc.CGM.GammaCube
        ra0,ra1=self.CSC.rag.min(),self.CSC.rag.max()
        dec0,dec1=self.CSC.decg.min(),self.CSC.decg.max()
        self.extent=ra0,ra1,dec0,dec1
        
        #pylab.ion()
        vmin=0
        vmax=np.max(GammaCube[0])
        self.vminmax=(vmin,vmax)
        pylab.figure("Simul")
        pylab.clf()
        pylab.imshow(GammaCube[0].T[::-1,:],extent=(ra0,ra1,dec0,dec1),cmap="cubehelix",vmin=vmin,vmax=vmax)
        #s=self.Cat.logM
        #s0,s1=s.min(),s.max()
        s=10.#(s-s0)/(s1-s0)*20+5.
        pylab.scatter(self.CSC.Cat.ra,self.CSC.Cat.dec,s=s,linewidths=0)
        #pylab.colorbar()
        pylab.draw()
        pylab.show(False)
        pylab.pause(0.1)

        
        
        self.ListX=[]
        self.ListL=[]
        iDone=0
        if self.MoveType=="Stretch":
            EvolveFunc=self._evolveChainStretch
        elif self.MoveType=="Walk":
            EvolveFunc=self._evolveChainWalk
        Sigma=1.
        LAccepted=[]
        while True:
            for iChain in range(self.NChain):
                APP.runJob("_evolveChain:%i"%(iChain), 
                           EvolveFunc,
                           args=(iChain,Sigma))#,serial=True)
            APP.awaitJobResults("_evolveChain:*", progress="Compute step %i"%iDone)
            ff=np.count_nonzero(self.Accepted)/float(self.Accepted.size)
            LAccepted.append(ff)
            print>>log,"Accepted fraction %.3f"%ff
            if (len(LAccepted)>10) and (len(LAccepted)%10==0):
                FAcc=np.mean(LAccepted[-10::])
                print>>log,"  Mean accepted fraction %.3f"%FAcc
                if FAcc>0.5:
                    Sigma*=1.05
                    print>>log,"    Accelerating proposal distribution (Sigma=%f)"%Sigma
                elif FAcc<0.2:
                    Sigma*=0.95
                    print>>log,"    Decelerating proposal distribution (Sigma=%f)"%Sigma
                    
            
            self.X[:]=self.X1[:]
            self.L[:]=self.L1[:]
            if iDone%10==0: self.PlotChains2(OutName="Test%4.4i.png"%iDone)
            iDone+=1

    def PlotChains(self,OutName="Test"):
        (vmin,vmax)=self.vminmax
        ra0,ra1,dec0,dec1=self.extent
        self.ListX.append(self.X.copy())
        self.ListL.append(self.L.copy())
        pylab.figure("Fit")
        pylab.clf()
        #pylab.plot(self.X.T)
        pylab.subplot(2,1,1)
        pylab.plot(np.array(self.ListX)[:,:,0])
        pylab.subplot(2,1,2)
        pylab.plot(np.array(self.ListL)[:,:,0])
        pylab.draw()
        pylab.show(False)
        pylab.pause(0.1)
            
        z0,z1=self.z0z1
        #X=np.mean(self.X,axis=0)
        fig=pylab.figure("Fit imshow",figsize=(10,5))
        pylab.clf()
        
        MassFunc=ClassMassFunction.ClassMassFunction()
        MassFunc.setGammaFunction((self.rac,self.decc),
                                  self.CellDeg,
                                  self.NPix,
                                  z=self.zParms,
                                  ScaleKpc=self.ScaleKpc)
        
        LSlice=[]
        for x in self.X:
            LX=[x]
            MassFunc.CGM.computeGammaCube(LX)
            LSlice.append(MassFunc.CGM.GammaCube[0])
        GammaSlice=np.mean(np.array(LSlice),axis=0)
        stdGammaSlice=np.std(np.array(LSlice),axis=0)
        pylab.subplot(1,2,1)
        pylab.imshow(GammaSlice.T[::-1,:],vmin=vmin,vmax=vmax,extent=(ra0,ra1,dec0,dec1),cmap="cubehelix")
        pylab.subplot(1,2,2)
        pylab.imshow(stdGammaSlice.T[::-1,:],vmin=vmin,vmax=vmax,extent=(ra0,ra1,dec0,dec1),cmap="cubehelix")
        pylab.draw()
        pylab.show(False)
        pylab.pause(0.1)
        fig.savefig(OutName)

    def PlotChains2(self,OutName="Test"):
        GammaCube=self.CSC.MassFunc.CGM.GammaCube
        ra0,ra1=self.CSC.rag.min(),self.CSC.rag.max()
        dec0,dec1=self.CSC.decg.min(),self.CSC.decg.max()
        self.extent=ra0,ra1,dec0,dec1
        aspect="auto"
        #pylab.ion()
        vmin=0
        vmax=np.max(GammaCube[0])
        self.vminmax=(vmin,vmax)
        fig=pylab.figure("MCMCOverview",figsize=(10,9))
        pylab.clf()
        pylab.subplot(2,2,1)
        pylab.imshow(GammaCube[0].T[::-1,:],extent=(ra0,ra1,dec0,dec1),cmap="cubehelix",vmin=vmin,vmax=vmax,aspect=aspect)
        #s=self.Cat.logM
        #s0,s1=s.min(),s.max()
        s=10.#(s-s0)/(s1-s0)*20+5.
        pylab.scatter(self.CSC.Cat.ra,self.CSC.Cat.dec,s=s,linewidths=0)
        pylab.title("Truth")
        
        (vmin,vmax)=self.vminmax
        ra0,ra1,dec0,dec1=self.extent
        self.ListX.append(self.X.copy())
        self.ListL.append(self.L.copy())

        pylab.subplot(2,2,2)
        pylab.plot(np.array(self.ListL)[:,:,0])
        pylab.title("Likelihood")
            
        z0,z1=self.z0z1
        
        MassFunc=ClassMassFunction.ClassMassFunction()
        MassFunc.setGammaFunction((self.rac,self.decc),
                                  self.CellDeg,
                                  self.NPix,
                                  z=self.zParms,
                                  ScaleKpc=self.ScaleKpc)
        
        LSlice=[]
        for x in self.X:
            LX=[x]
            MassFunc.CGM.computeGammaCube(LX)
            LSlice.append(MassFunc.CGM.GammaCube[0])
        GammaSlice=np.mean(np.array(LSlice),axis=0)
        stdGammaSlice=np.std(np.array(LSlice),axis=0)
        pylab.subplot(2,2,3)
        pylab.imshow(GammaSlice.T[::-1,:],vmin=vmin,vmax=vmax,extent=(ra0,ra1,dec0,dec1),cmap="cubehelix",aspect=aspect)
        pylab.title("Mean posterior")
        pylab.subplot(2,2,4)
        pylab.imshow(stdGammaSlice.T[::-1,:],vmin=vmin,vmax=vmax,extent=(ra0,ra1,dec0,dec1),cmap="cubehelix",aspect=aspect)
        pylab.title("std posterior")
        pylab.tight_layout()
        pylab.draw()
        pylab.show(False)
        pylab.pause(0.1)
        fig.savefig(OutName)
        
def test():
    C_MCMC=ClassRunMCMC()
    # C_MCMC.runDifferentialEvolution()
    C_MCMC.runMCMC()
    C_MCMC.killWorkers()
