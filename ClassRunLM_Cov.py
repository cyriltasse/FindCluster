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
#import ClassDiffLikelihoodMachine as ClassLikelihoodMachine
import ClassLogDiffLikelihoodMachine as ClassLikelihoodMachine
import ClassInitGammaCube
import ClassDisplayRGB

from DDFacet.Other.AsyncProcessPool import APP, WorkerProcessError
from DDFacet.Other import Multiprocessing
from DDFacet.Other import AsyncProcessPool

# from DDFacet.Other.AsyncProcessPool import APP, WorkerProcessError
# from DDFacet.Other import Multiprocessing
# from DDFacet.Other import AsyncProcessPool
import scipy.optimize
import GeneDist

# # ##############################
# # Catch numpy warning
# np.seterr(all='raise')
# import warnings
# warnings.filterwarnings('error')
# #with warnings.catch_warnings():
# #    warnings.filterwarnings('error')
# # ##############################

def test(DoPlot=False,ComputeInitCube=False):
    rac_deg,decc_deg=241.20678,55.59485 # cluster
    FOV=0.15
    FOV=0.05
#    FOV=0.02

    SubField={"rac_deg":rac_deg,
              "decc_deg":decc_deg,
              "FOV":FOV,
              "CellDeg":0.001}
    LMMachine=ClassRunLM_Cov(SubField,
                             DoPlot=DoPlot,
                             ComputeInitCube=ComputeInitCube)
    
    g=LMMachine.runLM()
    g=LMMachine.runMCMC()
    


    
    
class ClassRunLM_Cov():
    def __init__(self,
                 SubField,
                 DoPlot=False,
                 NCPU=56,
                 ComputeInitCube=True,
                 BasisType="Cov",
                 ScaleKpc=100.):
        self.NCPU=NCPU
        self.DoPlot=DoPlot
        self.rac_deg,self.decc_deg=SubField["rac_deg"],SubField["decc_deg"]
        self.rac,self.decc=self.rac_deg*np.pi/180,self.decc_deg*np.pi/180
        self.FOV=SubField["FOV"]
        self.CellDeg=SubField["CellDeg"]
        self.CellRad=self.CellDeg*np.pi/180
        np.random.seed(42)
        self.NPix=int(self.FOV/self.CellDeg)
        if (self.NPix%2)!=0:
            self.NPix+=1
        self.NPix=5
        
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
        pylab.close("all")


        # self.CIGC=ClassInitGammaCube.ClassInitGammaCube(self.CLM,ScaleKpc=[200.,500.])
        self.CIGC=ClassInitGammaCube.ClassInitGammaCube(self.CM,self.GM,ScaleKpc=[ScaleKpc])
        self.DicoChains = shared_dict.create("DicoChains")

        # ###########################
        # self.finaliseInit()
        # ###########################

        self.GM.initCovMatrices(ScaleFWHMkpc=ScaleKpc)
        self.simulCat()
        self.CLM.ComputeIndexCube(self.NPix)
        
        # self.InitCube(Compute=ComputeInitCube)
        
        # self.CLM.InitDiffMatrices()
        
    

        
    def finaliseInit(self):
        self.CIGC.finaliseInit()
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
        

        CubeInit[CubeInit<0.]=0.
#        if self.DoPlot: self.GM.PlotGammaCube(Cube=CubeInit,FigName="Measured")
        X=self.GM.CubeToVec(CubeInit)
        
#        self.CLM.MassFunction.updateGammaCube(X)
#        if self.DoPlot: self.GM.PlotGammaCube()

        self.NDim=self.CLM.MassFunction.GammaMachine.NParms
        self.X=np.float64(X)

    def simulCat(self):
        log.print("Simulating catalog...")
        self.X=np.random.randn(self.GM.NParms)
        # # self.X.fill(0.)
        
        # GammaCube=np.zeros((self.NSlice,self.NPix,self.NPix),np.float64)+5.
        # self.X=self.GM.CubeToVec(GammaCube)
        
        self.XSimul=self.X.copy()
        self.GM.PlotGammaCube(X=self.X,FigName="Simul")
        self.CubeSimul=self.GM.GammaCube.copy()
        
        GM=self.GM
        n_z=self.CM.DicoDATA["DicoSelFunc"]["n_z"]

        
        print(":!::")
        n_z.fill(1./self.CellRad**2)
        n_z*=30
        n_zt=self.CM.Cat_s.n_zt
        
        n_zt=[]
        X=[]
        Y=[]
        self.NCube=np.zeros_like(GM.GammaCube)
        for iSlice in range(self.NSlice):
            GammaSlice=GM.GammaCube[iSlice]
            ThisNzt=np.zeros((self.NSlice,),np.float32)
            ThisNzt[iSlice]=1
            ThisX=[]
            ThisY=[]
            for i in range(self.NPix):
                for j in range(self.NPix):
                    n=n_z[iSlice]*GammaSlice[i,j]*self.CellRad**2
                    #print("n=%f"%n)
                    #print(":!::")
                    # N=int(n)
                    N=scipy.stats.poisson.rvs(n,size=1)[0]
                    self.NCube[iSlice,i,j]=N
                    for iObj in range(N):
                        X.append(i)
                        Y.append(j)
                        ThisX.append(i)
                        ThisY.append(j)
                        n_zt.append(ThisNzt.copy()/self.CellRad**2)

            ThisX=np.float32(np.array(ThisX))
            ThisY=np.float32(np.array(ThisY))
            ThisX+=np.random.rand(ThisX.size)-0.5
            ThisY+=np.random.rand(ThisX.size)-0.5
            ax=GM.AxList[iSlice].scatter(ThisY,ThisX,s=0.2,c="red")
        pylab.draw()
        pylab.show(block=False)
        pylab.pause(0.1)
        # self.GM.PlotGammaCube(Cube=self.NCube,FigName="NCube")


        self.XSimul=self.X=self.GM.CubeToVec(self.NCube)
        
        
        
        n_zt=np.array(n_zt)
        X=np.array(X)
        Y=np.array(Y)
        self.CM.Cat_s=np.zeros((X.size,),self.CM.Cat_s.dtype)
        self.CM.Cat_s=self.CM.Cat_s.view(np.recarray)
        self.CM.Cat_s.xCube[:]=X[:]
        self.CM.Cat_s.yCube[:]=Y[:]
        self.CM.Cat_s.n_zt[:]=n_zt
        
    def runLM(self):
        T=ClassTimeIt.ClassTimeIt()
        T.disable()
        self.X=self.XSimul
        self.X=np.float64(self.X)
        g=self.X
        #g.fill(0)

        print("True Likelihood = %.5f"%(self.CLM.L(g)))
        iStep=0
        self.CLM.MassFunction.updateGammaCube(g)
        
        g=self.X

        GM=self.CLM.MassFunction.GammaMachine
        L_NParms=GM.L_NParms
            
        iStep=0
        def f(g):
            self.CLM.MassFunction.updateGammaCube(g)
            L=self.CLM.L(g)
            print(g,L)
            return L
        C=GeneDist.ClassDistMachine()
        g.fill(0)
        # g.flat[:]=np.random.randn(g.size)
        Alpha=1.
        #self.CLM.measure_dLdg(g)
        #self.CLM.measure_dJdg(g)
        #return
        L_L=[]
        L_g=[]
        
        while True:
            T.reinit()
            if iStep%1==0 and self.DoPlot:
                self.GM.PlotGammaCube(X=g,OutName="g%4.4i.png"%iStep)
                T.timeit("Plot Gamma Cube")
            # self.CLM.measure_dLdg(g)
            # stop
            T.timeit("Plot")
            L=self.CLM.L(g)
            T.timeit("Compute L")
            if iStep>0:
                if L<L_L[-1]:
                    fact=1.5
                    log.print("decreasing Alpha: %f -> %f"%(Alpha,Alpha/fact))
                    Alpha/=fact
                    g=L_g[-1]
                    continue
                elif L!=L_L[-1]:
                    log.print("  dL=%f"%(L-L_L[-1]))
                    dgest=np.median(np.abs(g-L_g[-1]))
                    log.print("  dg=%f"%(dgest))
                    if dgest<1e-4:
                        return g
            L_g.append(g.copy())
            L_L.append(L)
            
            log.print("Likelihood = %.5f"%(L))
            dldg=self.CLM.dLdg(g).flat[:]
            T.timeit("Compute J")
            dJdg=self.CLM.dJdg(g).flat[:]
            T.timeit("Compute H")
            # epsilon=np.sum(dldg**2)/np.sum(dldg**2*dJdg)
            # Alpha=1000*np.abs(epsilon)
            # print(epsilon)
            
            dldg=1/(1 + np.exp(-dldg))-0.5
            g.flat[:] += Alpha*dldg
            T.timeit("Step i+1")

            pylab.figure("hist")
            pylab.clf()
            ii=0
            for iSlice in range(self.CLM.NSlice):
                ThisNParms=L_NParms[iSlice]
                iPar=ii
                jPar=iPar+ThisNParms
                ii+=ThisNParms
                pylab.subplot(2,2,1)
                pylab.plot(dldg[iPar:jPar])
            pylab.subplot(2,2,2)
            pylab.plot(L_L)
            
            Sig=np.sqrt(np.abs(dJdg))
            Gamma=self.GM.giveGammaCube(g)
            e_Gamma=np.abs(Gamma-self.GM.giveGammaCube(g+Sig))

            ax=pylab.subplot(2,2,3)
            y0=Gamma.flatten()-e_Gamma.flatten()
            y1=Gamma.flatten()+e_Gamma.flatten()
            x=np.arange(y0.size)
            ax.fill_between(x,y0,y1, facecolor='gray', alpha=0.5)
            pylab.plot(Gamma.flatten(),color="black")
            pylab.plot(self.CubeSimul.flatten(),ls="--",color="black")

            ax=pylab.subplot(2,2,4)
            y0=Gamma.flatten()-e_Gamma.flatten()
            y1=Gamma.flatten()+e_Gamma.flatten()
            ys=self.CubeSimul.flatten()
            x=np.arange(y0.size)
            ax.fill_between(x,y0-ys,y1-ys, facecolor='gray', alpha=0.5)
            pylab.plot(Gamma.flatten()-ys,color="black")
            pylab.draw()
            pylab.show(block=False)
            pylab.pause(0.1)
            T.timeit("Plot Sim")
            
            iStep+=1

        # g0=np.zeros_like(g)
        # g0=np.random.randn(g.size)
        # Res=scipy.optimize.minimize(f,g0)
            


    def runMCMC(self,g0):
        T=ClassTimeIt.ClassTimeIt()
        T.disable()
        g=g0
        L=self.CLM.L(g)
        
        GM=self.CLM.MassFunction.GammaMachine
        L_NParms=GM.L_NParms
            
        iStep=0
        Alpha=1.
        L_L=[L]
        L_g=[g]
        
        while True:
            T.reinit()

            T.timeit("Compute L")
            g1=g+Alpha*np.random.randn(*g.shape)
            L1=self.CLM.L(g1)
            
            if iStep>0:
                if L<L_L[-1]:
                    fact=1.5
                    log.print("decreasing Alpha: %f -> %f"%(Alpha,Alpha/fact))
                    Alpha/=fact
                    g=L_g[-1]
                    continue
                elif L!=L_L[-1]:
                    log.print("  dL=%f"%(L-L_L[-1]))
                    dgest=np.median(np.abs(g-L_g[-1]))
                    log.print("  dg=%f"%(dgest))
                    if dgest<1e-4:
                        return g
            L_g.append(g.copy())
            L_L.append(L)
            
            log.print("Likelihood = %.5f"%(L))
            dldg=self.CLM.dLdg(g).flat[:]
            T.timeit("Compute J")
            dJdg=self.CLM.dJdg(g).flat[:]
            T.timeit("Compute H")
            # epsilon=np.sum(dldg**2)/np.sum(dldg**2*dJdg)
            # Alpha=1000*np.abs(epsilon)
            # print(epsilon)
            
            dldg=1/(1 + np.exp(-dldg))-0.5
            g.flat[:] += Alpha*dldg
            T.timeit("Step i+1")

            pylab.figure("hist")
            pylab.clf()
            ii=0
            for iSlice in range(self.CLM.NSlice):
                ThisNParms=L_NParms[iSlice]
                iPar=ii
                jPar=iPar+ThisNParms
                ii+=ThisNParms
                pylab.subplot(2,2,1)
                pylab.plot(dldg[iPar:jPar])
            pylab.subplot(2,2,2)
            pylab.plot(L_L)
            
            Sig=np.sqrt(np.abs(dJdg))
            Gamma=self.GM.giveGammaCube(g)
            e_Gamma=np.abs(Gamma-self.GM.giveGammaCube(g+Sig))

            ax=pylab.subplot(2,2,3)
            y0=Gamma.flatten()-e_Gamma.flatten()
            y1=Gamma.flatten()+e_Gamma.flatten()
            x=np.arange(y0.size)
            ax.fill_between(x,y0,y1, facecolor='gray', alpha=0.5)
            pylab.plot(Gamma.flatten(),color="black")
            pylab.plot(self.CubeSimul.flatten(),ls="--",color="black")

            ax=pylab.subplot(2,2,4)
            y0=Gamma.flatten()-e_Gamma.flatten()
            y1=Gamma.flatten()+e_Gamma.flatten()
            ys=self.CubeSimul.flatten()
            x=np.arange(y0.size)
            ax.fill_between(x,y0-ys,y1-ys, facecolor='gray', alpha=0.5)
            pylab.plot(Gamma.flatten()-ys,color="black")
            pylab.draw()
            pylab.show(block=False)
            pylab.pause(0.1)
            T.timeit("Plot Sim")
            
            iStep+=1

        # g0=np.zeros_like(g)
        # g0=np.random.randn(g.size)
        # Res=scipy.optimize.minimize(f,g0)
            
        
