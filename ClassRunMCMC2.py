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
import ClassLikelihoodMachine
import ClassInitGammaCube
import ClassDisplayRGB

def g_z(z):
    a=4.
    g=np.zeros_like(z)
    ind=np.where((z>(1./a))&(z<a))[0]
    g[ind]=1./z[ind]
    return g

def test(ComputeInitCube=False):
    rac_deg,decc_deg=241.20678,55.59485 # cluster
    FOV=0.15
    SubField={"rac_deg":rac_deg,
              "decc_deg":decc_deg,
              "FOV":FOV,
              "CellDeg":0.001}
    MCMCMachine=ClassRunMCMC(SubField,ComputeInitCube=ComputeInitCube)
    
    MCMCMachine.runMCMC()
    
class ClassRunMCMC():
    def __init__(self,SubField,NCPU=56,SWalk=3,MoveType="Stretch",ComputeInitCube=True):
        self.SWalk=SWalk
        self.NCPU=NCPU
        self.MoveType=MoveType
        self.rac_deg,self.decc_deg=SubField["rac_deg"],SubField["decc_deg"]
        self.rac,self.decc=self.rac_deg*np.pi/180,self.decc_deg*np.pi/180
        self.FOV=SubField["FOV"]
        self.CellDeg=SubField["CellDeg"]
        self.CellRad=self.CellDeg*np.pi/180
        # Type="bior1.3"
        # Type="db10"
        # Type="coif10"
        # Type="sym2"
        # Type="rbio1.3"
        # Type="haar"
        self.WaveDict={"Type":"haar",
                       "Level":5}
        # self.WaveDict={"Type":"db5",
        #                "Level":3}

        #self.NPix=int(self.FOV/self.CellDeg)
        #self.NPix, _ = EstimateNpix(float(self.NPix), Padding=1)
        
        self.NPix=int(self.FOV/self.CellDeg)
        if (self.NPix%2)!=0:
            self.NPix+=1
            
        print>>log,"Choosing NPix=%i"%self.NPix

        self.CM=ClassCatalogMachine.ClassCatalogMachine()
        self.CM.Init()
        self.CM.cutCat(self.rac,self.decc,self.NPix,self.CellRad)
        self.zParms=self.CM.zg_Pars
        self.NSlice=self.zParms[-1]-1
        self.logMParms=self.CM.logM_Pars
        self.logM_g=np.linspace(*self.logMParms)

        self.DistMachine=GeneDist.ClassDistMachine()
        z=np.linspace(-10,10,1000)
        g=g_z(z)
        G=np.cumsum(g)
        G/=G[-1]
        self.DistMachine.setCumulDist(z,G)

        self.CLM=ClassLikelihoodMachine.ClassLikelihoodMachine(self.CM)
        self.CLM.MassFunction.setGammaGrid((self.CM.rac_main,self.CM.decc_main),
                                           (self.rac,self.decc),
                                           self.CellDeg,
                                           self.NPix,
                                           zParms=self.zParms)
        #CLM.showRGB()
        self.CLM.ComputeIndexCube(self.NPix)

        

        #self.CIGC=ClassInitGammaCube.ClassInitGammaCube(self.CLM,ScaleKpc=[200.,500.])
        self.CIGC=ClassInitGammaCube.ClassInitGammaCube(self.CM,self.CLM.MassFunction.GammaMachine,ScaleKpc=[500.])
        self.DicoChains = shared_dict.create("DicoChains")
        

        APP.registerJobHandlers(self)
        self.finaliseInit()

        self.InitCube(Compute=ComputeInitCube)
        
        
    def finaliseInit(self):
        AsyncProcessPool.init(ncpu=self.NCPU,
                              affinity=0)
        APP.startWorkers()

        
    def InitCube(self,Compute=False):
        print>>log,"Initialise GammaCube..."
        if Compute:
            Cube=self.CIGC.InitGammaCube()
            np.save("Cube",Cube)
        else:
            Cube=np.load("Cube.npy")
        
        self.CubeInit=Cube.copy()
        self.DicoChains["CubeInit"]=self.CubeInit
        self.CLM.MassFunction.GammaMachine.setWaveType(Kind=self.WaveDict["Type"],Th=3e-2,Mode="symmetric",Level=self.WaveDict["Level"])
        self.CLM.MassFunction.GammaMachine.setReferenceCube(self.DicoChains["CubeInit"])

        CubeInit=self.DicoChains["CubeInit"]
        CubeInit=np.random.randn(*(self.DicoChains["CubeInit"].shape))*0.0001+1

        
        self.CLM.MassFunction.GammaMachine.PlotGammaCube(Cube=CubeInit,FigName="Measured")
        X=self.CLM.MassFunction.GammaMachine.CubeToVec(CubeInit)
        
        self.CLM.MassFunction.updateGammaCube(X)
        self.CLM.MassFunction.GammaMachine.PlotGammaCube()

        self.NDim=self.CLM.MassFunction.GammaMachine.NParms
        self.NChain=self.NDim*4
        self.InitDicoChains(X)
        
        
    def _setReferenceCube(self):
        if self.CLM.MassFunction.GammaMachine.HasReferenceCube: return
        self.CLM.MassFunction.GammaMachine.DoPrint=False
        self.CLM.MassFunction.GammaMachine.setWaveType(Kind=self.WaveDict["Type"],Th=3e-2,Mode="symmetric",Level=self.WaveDict["Level"])
        self.CLM.MassFunction.GammaMachine.setReferenceCube(self.DicoChains["CubeInit"])
        
        
    def InitDicoChains(self,XInit):
        print>>log,"Initialise Markov Chains..."
        STD=np.max([1e-3,np.std(XInit)])
        self.DicoChains["X"]=np.random.randn(self.NChain,self.NDim)*STD+XInit.reshape((1,-1))
        self.DicoChains["ChainCube"]=np.zeros((self.NChain,self.NSlice,self.NPix,self.NPix),np.float32)

        #self.DicoChains["X"]=(10**(np.linspace(-1,4,self.NChain))).reshape((self.NChain,self.NDim))*XInit.reshape((1,-1))

        
        self.DicoChains["L"]=np.zeros((self.NChain,),np.float64)
        
        for iChain in range(self.NChain):
            APP.runJob("_evalChain:%i"%(iChain), 
                       self._evalChain,
                       args=(iChain,))#,serial=True)
        APP.awaitJobResults("_evalChain:*", progress="Compute L for init")
        self.DicoChains["Accepted"]=np.zeros((self.NChain,),int)
        #self.PlotL()

    def PlotL(self):
        import pylab
        X=self.DicoChains["X"]
        Y=self.DicoChains["L"]
        L=self.DicoChains["L"][:,0]
        print>>log,"max %f"%X[np.argmax(L)]
        pylab.clf()
        for i in range(4):
            pylab.subplot(2,2,i+1)
            pylab.scatter(np.log10(X.flatten()),Y[:,i].flatten())
        pylab.draw()
        pylab.show(False)
        pylab.pause(0.1)
        stop
        
    def _evalChain(self,iChain):
        self.DicoChains.reload()
        self._setReferenceCube()
        x=self.DicoChains["X"][iChain]
        self.DicoChains["L"][iChain]=self.CLM.log_prob(x)

        

    def showRGB(self,zLabels=False):
        DRGB=ClassDisplayRGB.ClassDisplayRGB()
        DRGB.setRGB_FITS(*self.CM.DicoDataNames["RGBNames"])
        DRGB.setRaDec(self.rac_deg,self.decc_deg)
        DRGB.setBoxArcMin(self.NPix*self.CellDeg*60.)
        DRGB.FitsToArray()
        DRGB.Display()#Scale="linear",vmin=,vmax=30)
        if zLabels:
            import pylab
            pylab.scatter(self.CM.Cat_s.l,self.CM.Cat_s.m)
            for i in range(self.CM.Cat_s.shape[0]):
                pylab.text(self.CM.Cat_s.l[i],self.CM.Cat_s.m[i],self.CM.Cat_s.z[i],color="red")
            pylab.draw()

    
    


        

        
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

    # ##########################################################

    def _evolveChainStretch(self,iChain,Sigma):
        self.DicoChains.reload()
        self._setReferenceCube()
        NChain,NDim=self.DicoChains["X"].shape
        while True:
            k=int(np.random.rand(1)[0]*NChain)
            H=NChain//2
            if iChain>=H:
                Cond1=(k<H)
            else:
                Cond1=(k>=H)
            if (k!=iChain)&(Cond1): break
                
        z=self.DistMachine.GiveSample(1)[0]
        X0=self.DicoChains["X"][iChain]
        X1=self.DicoChains["X"][k]
        
        X2=X0+z*(X1-X0)
            
        L1=self.CLM.log_prob(X2)
        L0=self.DicoChains["L"][iChain]
        #
        dd=L1-L0+(NDim-1)*np.log(z)
        if dd>10.:
            pp=1.
        else:
            pp=np.exp(dd)
            #pp=np.exp(L1-L0)
        p=np.min([1,pp])
        #print "   %f, %f --> %f"%(L1,L0,p)
        r=np.random.rand(1)[0]
        if r<p:
            self.DicoChains["X"][iChain]=X2[:]
            self.DicoChains["L"][iChain]=L1
            self.DicoChains["Accepted"][iChain]=1
            self.DicoChains["ChainCube"][iChain][...]=self.CLM.MassFunction.GammaMachine.GammaCube[...]
        else:
            self.DicoChains["Accepted"][iChain]=0

    def _evolveChainWalk(self,iChain,Sigma):
        T=ClassTimeIt.ClassTimeIt("_evolveChainWalk")
        self.DicoChains.reload()
        kArray=-1*np.ones((self.SWalk,),np.int32)
        NChain=self.DicoChains["X"].shape[0]
        for iS in range(self.SWalk):
            while True:
                k=int(np.random.rand(1)[0]*NChain)
                H=NChain//2
                if iChain>=H:
                    Cond1=(k<H)
                else:
                    Cond1=(k>=H)
                if (k!=iChain)&(Cond1)&(k not in kArray):
                    kArray[iS]=k
                    break
        T.timeit("sel")
        Xs=self.DicoChains["X"][kArray]
        meanXs=np.mean(Xs,axis=0)
        Xs0=Xs-meanXs.reshape((1,-1))
        covXs=np.dot(Xs0.T,Xs)/self.SWalk

        X0=self.DicoChains["X"][iChain]
        try:
            X2=X0+Sigma*np.random.multivariate_normal(np.zeros((X0.size,),np.float32),covXs,1)[0]
        except:
            X2=X0
            
        T.timeit("prob")
        L1=self.CLM.log_prob(X2)
        T.timeit("likelihood")
        L0=self.DicoChains["L"][iChain]
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
            self.DicoChains["X"][iChain]=X2[:]
            self.DicoChains["L"][iChain]=L1
            self.DicoChains["Accepted"][iChain]=1
        else:
            self.DicoChains["Accepted"][iChain]=0

        T.timeit("rest")
            

        
        


    def runMCMC(self):
        
        
        print>>log,"Run MCMC..."
        self.ListX=[]
        self.ListL=[]
        self.Accepted=self.DicoChains["Accepted"]
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
                    
            
            if iDone%10==0:
                self.PlotChains2(OutName="Test%6.6i.png"%iDone)
            iDone+=1

        
    def PlotChains2(self,OutName="Test"):
        GammaCube=self.CubeInit
        # ra0,ra1=self.CSC.rag.min(),self.CSC.rag.max()
        # dec0,dec1=self.CSC.decg.min(),self.CSC.decg.max()
        # self.extent=ra0,ra1,dec0,dec1
        aspect="auto"
        #pylab.ion()
        self.DicoChains.reload()
        XMean=np.mean(self.DicoChains["X"],axis=0)
        CubeMean=np.mean(self.DicoChains["ChainCube"],axis=0)
        #CubeMean/=self.NChain
        
        self.CLM.MassFunction.updateGammaCube(XMean)
        #print CubeMean-self.CLM.MassFunction.GammaMachine.GammaCube
        self.CLM.MassFunction.GammaMachine.PlotGammaCube(Cube=CubeMean,
                                                         FigName="Mean posterior",
                                                         OutName=OutName)

