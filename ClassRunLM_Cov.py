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

from DDFacet.Other import ModColor
from DDFacet.Other import ClassTimeIt
from DDFacet.Other.progressbar import ProgressBar
from DDFacet.Array import shared_dict
import ClassCatalogMachine
import ClassDiffLikelihoodMachine as ClassLikelihoodMachine
import ClassInitGammaCube
import ClassDisplayRGB

from DDFacet.Other.AsyncProcessPool import APP, WorkerProcessError
from DDFacet.Other import Multiprocessing
from DDFacet.Other import AsyncProcessPool

# from DDFacet.Other.AsyncProcessPool import APP, WorkerProcessError
# from DDFacet.Other import Multiprocessing
# from DDFacet.Other import AsyncProcessPool


def test(ComputeInitCube=False):
    rac_deg,decc_deg=241.20678,55.59485 # cluster
    FOV=0.15
    FOV=0.05

    SubField={"rac_deg":rac_deg,
              "decc_deg":decc_deg,
              "FOV":FOV,
              "CellDeg":0.001}
    LMMachine=ClassRunLM_Cov(SubField,
                             ComputeInitCube=ComputeInitCube)
    
    LMMachine.runLM()
    
class ClassRunLM_Cov():
    def __init__(self,
                 SubField,
                 NCPU=56,
                 ComputeInitCube=True,
                 BasisType="Cov",
                 ScaleKpc=100.):
        self.NCPU=NCPU
        self.rac_deg,self.decc_deg=SubField["rac_deg"],SubField["decc_deg"]
        self.rac,self.decc=self.rac_deg*np.pi/180,self.decc_deg*np.pi/180
        self.FOV=SubField["FOV"]
        self.CellDeg=SubField["CellDeg"]
        self.CellRad=self.CellDeg*np.pi/180
        
        self.NPix=int(self.FOV/self.CellDeg)
        if (self.NPix%2)!=0:
            self.NPix+=1
            
        log.print("Choosing NPix=%i"%self.NPix)

        self.CM=ClassCatalogMachine.ClassCatalogMachine()
        self.CM.Init()
        self.CM.cutCat(self.rac,self.decc,self.NPix,self.CellRad)
        self.zParms=self.CM.zg_Pars
        self.NSlice=self.zParms[-1]-1
        self.logMParms=self.CM.logM_Pars
        self.logM_g=np.linspace(*self.logMParms)

        self.DistMachine=GeneDist.ClassDistMachine()

        self.CLM=ClassLikelihoodMachine.ClassLikelihoodMachine(self.CM)
        self.CLM.MassFunction.setGammaGrid((self.CM.rac_main,self.CM.decc_main),
                                           (self.rac,self.decc),
                                           self.CellDeg,
                                           self.NPix,
                                           zParms=self.zParms,
                                           ScaleKpc=ScaleKpc)
        self.GM=self.CLM.MassFunction.GammaMachine

        # CLM.showRGB()
        self.CLM.ComputeIndexCube(self.NPix)
        


        # self.CIGC=ClassInitGammaCube.ClassInitGammaCube(self.CLM,ScaleKpc=[200.,500.])
        self.CIGC=ClassInitGammaCube.ClassInitGammaCube(self.CM,self.GM,ScaleKpc=[ScaleKpc])
        self.DicoChains = shared_dict.create("DicoChains")
        
        self.finaliseInit()

        self.GM.initCovMatrices(ScaleFWHMkpc=ScaleKpc)
        self.InitCube(Compute=ComputeInitCube)
        self.CLM.InitDiffMatrices()

    

        
    def finaliseInit(self):
        APP.registerJobHandlers(self)
        AsyncProcessPool.init(ncpu=self.NCPU,
                              affinity=0)
        APP.startWorkers()

        
    def InitCube(self,Compute=False):
        log.print("Initialise GammaCube...")
        if Compute:
            if "EffectiveOmega" not in self.CIGC.DicoCube.keys():
                self.CIGC.computeEffectiveOmega()
            Cube=self.CIGC.InitGammaCube()
            np.save("Cube",Cube)
        else:
            Cube=np.load("Cube.npy")

        
        self.CubeInit=Cube.copy()
        self.DicoChains["CubeInit"]=self.CubeInit
            
        CubeInit=self.DicoChains["CubeInit"]
        #CubeInit=np.random.randn(*(self.DicoChains["CubeInit"].shape))*0.0001+1
        
        self.GM.PlotGammaCube(Cube=CubeInit,FigName="Measured")
        X=self.GM.CubeToVec(CubeInit)
        
        self.CLM.MassFunction.updateGammaCube(X)
        self.GM.PlotGammaCube()

        self.NDim=self.CLM.MassFunction.GammaMachine.NParms
        self.X=X
        
    def runLM(self):
        Alpha=0.1
        g=self.X
        g.fill(0)
        print("Likelihood = %.5f"%(self.CLM.L(g)))
        iStep=0
        self.CLM.MassFunction.updateGammaCube(g)
        self.GM.PlotGammaCube(OutName="g%4.4i.png"%iStep)

        while True:
            g+= Alpha*self.CLM.dLdg(g)
            print("Likelihood = %.5f"%(self.CLM.L(g)))
            if iStep%1==0:
                self.CLM.MassFunction.updateGammaCube(g)
                self.GM.PlotGammaCube(OutName="g%4.4i.png"%iStep)
                
            iStep+=1
            
