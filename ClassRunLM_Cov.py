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
    np.save("gEst.npy",g)
    return
    g=np.load("gEst.npy")
    #    LMMachine.runMCMC(g)#LMMachine.XSimul.ravel())
    
    LMMachine.InitDicoChains(g)
    LMMachine.runMCMC()
    

def g_z(z):
    a=4.
    g=np.zeros_like(z)
    ind=np.where((z>(1./a))&(z<a))[0]
    g[ind]=1./z[ind]
    return g

    
    
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
            
        # self.NPix=5
        
        log.print("Choosing NPix=%i"%self.NPix)

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
                                           zParms=self.zParms,
                                           ScaleKpc=ScaleKpc)
        self.GM=self.CLM.MassFunction.GammaMachine

        # CLM.showRGB()
        #self.CLM.ComputeIndexCube(self.NPix)
        pylab.close("all")


        # self.CIGC=ClassInitGammaCube.ClassInitGammaCube(self.CLM,ScaleKpc=[200.,500.])
        self.CIGC=ClassInitGammaCube.ClassInitGammaCube(self.CM,self.GM,ScaleKpc=[ScaleKpc])
        self.DicoChains = shared_dict.create("DicoChains")

        ###########################
        self.finaliseInit()
        ###########################

        self.GM.initCovMatrices(ScaleFWHMkpc=ScaleKpc)

        self.XSimul=None
        self.simulCat()
        
        self.CLM.ComputeIndexCube(self.NPix)
        self.MoveType="Stretch"
        
        

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
        #n_z.fill(1./self.CellRad**2)
        #n_z*=30
        n_zt=self.CM.Cat_s.n_zt
        
        n_zt=[]
        X=[]
        Y=[]
        self.NCube=np.zeros_like(GM.GammaCube)
        for iSlice in range(self.NSlice):
            GammaSlice=GM.GammaCube[iSlice]
            ThisNzt=np.zeros((self.NSlice,),np.float32)
            ThisX=[]
            ThisY=[]
            # if iSlice>0: continue
            ThisNzt[iSlice]=1
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
            ax=GM.AxList[iSlice].scatter(ThisY,ThisX,s=0.5,c="red")
        pylab.draw()
        pylab.show(block=False)
        pylab.pause(0.1)
        #self.GM.PlotGammaCube(Cube=self.NCube,FigName="NCube")
        log.print("Total number of objects in the simulated catalog: %i"%len(X))

        # self.XSimul=self.X=self.GM.CubeToVec(self.NCube)
        
        
        
        n_zt=np.array(n_zt)
        X=np.array(X)
        Y=np.array(Y)
        self.CM.Cat_s=np.zeros((X.size,),self.CM.Cat_s.dtype)
        self.CM.Cat_s=self.CM.Cat_s.view(np.recarray)
        self.CM.Cat_s.xCube[:]=X[:]
        self.CM.Cat_s.yCube[:]=Y[:]
        self.CM.Cat_s.n_zt[:]=n_zt
        
    def runLM(self,NMaxSteps=300):
        T=ClassTimeIt.ClassTimeIt()
        T.disable()
        if self.XSimul is not None:
            self.X=self.XSimul
        else:
            self.X=np.zeros((self.GM.NParms,),np.float64)
        self.X=np.float64(self.X)
        g=self.X
        #g.fill(0)

        log.print("True Likelihood = %.5f"%(self.CLM.L(g)))
        # g.fill(0)
        # log.print("True Likelihood = %.5f"%(self.CLM.L(g)))
        iStep=0
        self.CLM.MassFunction.updateGammaCube(g)
        
        g=self.X

        GM=self.CLM.MassFunction.GammaMachine
        L_NParms=GM.L_NParms
        ssqs=np.concatenate(self.GM.L_ssqs)
        #ssqs/=np.min(ssqs)
            
        iStep=0
        def f(g):
            self.CLM.MassFunction.updateGammaCube(g)
            L=self.CLM.L(g)
            print(g,L)
            return L
        C=GeneDist.ClassDistMachine()
        #g.fill(0)
        g.flat[:]=np.random.randn(g.size)*0.1
        Alpha=1.
        # self.CLM.measure_dLdg(g)
        # self.CLM.measure_dJdg(g)
        # return
        L_L=[]
        L_g=[]
        StepPlot=10
        while True:
            T.reinit()
            log.print("======================== Doing step = %i ========================"%iStep)
            if iStep%StepPlot==0 and self.DoPlot:
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
                    if iStep>NMaxSteps:
                        return g
                    #if dgest<1e-4:
                    #    return g
            L_g.append(g.copy())
            L_L.append(L)
            
            log.print("Likelihood = %.5f"%(L))
            dldg=self.CLM.dLdg(g).flat[:]#/ssqs.flat[:]
            T.timeit("Compute J")
            
            # epsilon=np.sum(dldg**2)/np.sum(dldg**2*dJdg)
            # Alpha=1000*np.abs(epsilon)
            # print(epsilon)
            
            dldg=1/(1 + np.exp(-dldg))-0.5
            g.flat[:] += Alpha*dldg
            T.timeit("Step i+1")

            iStep+=1
            if iStep%StepPlot!=0: continue
            
            dJdg=self.CLM.dJdg(g).flat[:]
            T.timeit("Compute H")

            figH=pylab.figure("hist")
            pylab.clf()
            ii=0
            for iSlice in range(self.CLM.NSlice):
                ThisNParms=L_NParms[iSlice]
                iPar=ii
                jPar=iPar+ThisNParms
                ii+=ThisNParms
                pylab.subplot(2,2,1)
                #pylab.plot(dldg[iPar:jPar])
                C=GeneDist.ClassDistMachine()
                x,y=C.giveCumulDist(g[iPar:jPar],Ns=100,Norm=True)#,xmm=[-5,5])
                pylab.plot(x,y)
                #pylab.plot(g[iPar:jPar])

            pylab.subplot(2,2,2)
            pylab.plot(L_L)
            
            Gamma=self.GM.giveGammaCube(g)
            Sig=np.sqrt(1./np.abs(dJdg))
            
            Sig[Sig>1.]=1
            # Sig=(1./np.abs(dJdg))
            NTry=100
            GammaVar=np.zeros_like(Gamma)
            GammaStat=np.zeros((NTry,self.NSlice,self.NPix,self.NPix),np.float32)
            for iTry in range(NTry):
                GammaStat[iTry]=self.GM.giveGammaCube(g+Sig*np.random.randn(*g.shape))
            y0Cube=np.quantile(GammaStat,0.16,axis=0)
            y1Cube=np.quantile(GammaStat,0.84,axis=0)
            y0=y0Cube.flatten()
            y1=y1Cube.flatten()
            ycCube=np.quantile(GammaStat,0.50,axis=0)
            yc=ycCube.flatten()
            
            
            e_Gamma=GammaStd=(y1-y0)/2.#GammaVar/NTry
            
            #e_Gamma=np.abs(Gamma-self.GM.giveGammaCube(g+Sig))

            ax=pylab.subplot(2,2,3)
            #y0=Gamma.flatten()-e_Gamma.flatten()
            #y1=Gamma.flatten()+e_Gamma.flatten()
            x=np.arange(Gamma.size)
            ax.fill_between(x,y0,y1, facecolor='gray', alpha=0.5)
            pylab.plot(Gamma.flatten(),color="black")
            pylab.plot(yc.flatten(),color="black",ls=":")
            if self.XSimul is not None:
                ys=self.CubeSimul.flatten()
                pylab.plot(self.CubeSimul.flatten(),ls="--",color="black")
            else:
                ys=0
                
            ax=pylab.subplot(2,2,4)
            y0=Gamma.flatten()-e_Gamma.flatten()
            y1=Gamma.flatten()+e_Gamma.flatten()
            ym=y=Gamma.flatten()
            ey=e_Gamma.flatten()
            #v=((y-ys)/ey) # pylab.hist(,bins=50)
            # C=GeneDist.ClassDistMachine()
            # x,y=C.giveCumulDist(v,Ns=100,Norm=True,xmm=[-5,5])
            # x1,y1=C.giveCumulDist(np.random.randn(1000),Ns=100,Norm=True,xmm=[-5,5])
            # pylab.plot(x,y,color="black")
            # pylab.plot(x1,y1,ls=":",color="black")

            v0=y0-ys
            vc=yc-ys
            vm=ym-ys
            v1=y1-ys
            
            C=GeneDist.ClassDistMachine()
            x0,y0=C.giveCumulDist(v0,Ns=100,Norm=True,xmm=[-5,5])
            x1,y1=C.giveCumulDist(v1,Ns=100,Norm=True,xmm=[-5,5])
            xc,yc=C.giveCumulDist(vc,Ns=100,Norm=True,xmm=[-5,5])
            xm,ym=C.giveCumulDist(vm,Ns=100,Norm=True,xmm=[-5,5])

            # pylab.plot(x0,y0,color="black")
            # pylab.plot(x1,y1,color="black")
            pylab.fill_between(x0,y0,y1,color="gray")
            pylab.plot(xc,yc,color="black",ls=":")
            pylab.plot(xm,ym,color="black")
            #pylab.plot([0,0],[0,1],color="black",ls=":")
            pylab.grid()
            # ax.fill_between(x,y0-ys,y1-ys, facecolor='gray', alpha=0.5)
            # pylab.plot(Gamma.flatten()-ys,color="black")
            pylab.draw()
            pylab.show(block=False)
            pylab.pause(0.1)
            T.timeit("Plot Sim")
            figH.savefig("Hist%5.5i.png"%iStep)
            #self.GM.PlotGammaCube(Cube=y0Cube,FigName="Cube0")
            #self.GM.PlotGammaCube(Cube=y1Cube,FigName="Cube1")
            self.GM.PlotGammaCube(Cube=ycCube,FigName="CubeC")

        # g0=np.zeros_like(g)
        # g0=np.random.randn(g.size)
        # Res=scipy.optimize.minimize(f,g0)

    # ####################################################
    # ############# MCMC
    # ####################################################
        
    def InitDicoChains(self,XInit):
        self.gStart=XInit.copy()        
        log.print("Initialise Markov Chains...")
        self.NChain,self.NDim=XInit.size//2,XInit.size
        #self.NChain,self.NDim=10,XInit.size
        STD=0.1#np.max([1e-3,np.std(XInit)])
        log.print("Randomize Chains...")
        self.DicoChains["X"]=np.random.randn(self.NChain,self.NDim)*STD+XInit.reshape((1,-1))
        # log.print("Set ChainCube...")
        # self.DicoChains["ChainCube"]=np.zeros((self.NChain,self.NSlice,self.NPix,self.NPix),np.float32)


        
        log.print("Compute L...")
        self.DicoChains["L"]=np.zeros((self.NChain,),np.float64)
        
        for iChain in range(self.NChain):
            APP.runJob("_evalChain:%i"%(iChain), 
                       self._evalChain,
                       args=(iChain,))#,serial=True)
        APP.awaitJobResults("_evalChain:*", progress="Compute L for init")
        self.DicoChains["Accepted"]=np.zeros((self.NChain,),int)
        #self.PlotL()
        
    def _evalChain(self,iChain):
        self.DicoChains.reload()
        x=self.DicoChains["X"][iChain]
        self.DicoChains["L"][iChain]=self.CLM.L(x)

    def _evolveChainStretch(self,iChain,Sigma):
        self.DicoChains.reload()

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
            
        L1=self.CLM.L(X2)
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
            #self.DicoChains["ChainCube"][iChain][...]=self.CLM.MassFunction.GammaMachine.GammaCube[...]
        else:
            self.DicoChains["Accepted"][iChain]=0
        
    def runMCMC(self):
        
        
        log.print("Run MCMC...")
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
            log.print("Accepted fraction %.3f"%ff)
            if (len(LAccepted)>10) and (len(LAccepted)%10==0):
                FAcc=np.mean(LAccepted[-10::])
                log.print("  Mean accepted fraction %.3f"%FAcc)
                if FAcc>0.5:
                    Sigma*=1.05
                    log.print("    Accelerating proposal distribution (Sigma=%f)"%Sigma)
                elif FAcc<0.2:
                    Sigma*=0.95
                    log.print("    Decelerating proposal distribution (Sigma=%f)"%Sigma)
                    
            
            if iDone%10==0:
                self.PlotChains2(OutName="Test%6.6i.png"%iDone)
            iDone+=1

        
    def PlotChains2(self,OutName="Test"):
        
        MeanGammaStack=np.zeros((self.NSlice,self.NPix,self.NPix),np.float32)
        for iChain in range(self.NChain):
            MeanGammaStack+=self.GM.giveGammaCube(self.DicoChains["X"][iChain])
        MeanGammaStack/=self.NChain
        
        StdGammaStack=np.zeros((self.NSlice,self.NPix,self.NPix),np.float32)
        for iChain in range(self.NChain):
            StdGammaStack+=(self.GM.giveGammaCube(self.DicoChains["X"][iChain])-MeanGammaStack)**2
        StdGammaStack/=self.NChain
        StdGammaStack=np.sqrt(StdGammaStack)

        # ##########################
        CubeStart=self.GM.giveGammaCube(self.gStart)
        e_Gamma=StdGammaStack
        Gamma=MeanGammaStack
        y0=Gamma.flatten()-e_Gamma.flatten()
        y1=Gamma.flatten()+e_Gamma.flatten()
        x=np.arange(y0.size)
        if not self.XSimul:
            self.CubeSimul=0
            ysim=0
        else:
            ysim=self.CubeSimul.flatten()
        ystart=CubeStart.flatten()
        ymean=MeanGammaStack.flatten()
        pylab.figure("hist")
        pylab.clf()
        ax=pylab.subplot(1,2,1)
        ax.fill_between(x,y0-ysim,y1-ysim, facecolor='gray', alpha=0.5)
        pylab.plot(ymean-ysim,color="black")
        pylab.plot(ystart-ysim,ls=":",color="black")
        #pylab.plot(ysim,ls="--",color="black")
        

        ax=pylab.subplot(1,2,2)
        v=((MeanGammaStack.flat[:]-ysim)/StdGammaStack.flat[:]) # pylab.hist(,bins=50)
        C=GeneDist.ClassDistMachine()
        x,y=C.giveCumulDist(v,Ns=100,Norm=True)
        x1,y1=C.giveCumulDist(np.random.randn(1000),Ns=100,Norm=True)
        pylab.plot(x,y,color="black")
        pylab.plot(x1,y1,ls=":",color="black")
        pylab.draw()
        pylab.show(block=False)
        pylab.pause(0.1)

        # ax=pylab.subplot(1,2,2)
        # y0=Gamma.flatten()-e_Gamma.flatten()
        # y1=Gamma.flatten()+e_Gamma.flatten()
        # ys=self.CubeSimul.flatten()
        # x=np.arange(y0.size)
        # ax.fill_between(x,y0-ys,y1-ys, facecolor='gray', alpha=0.5)
        # pylab.plot(Gamma.flatten()-ys,color="black")


        # self.GM.PlotGammaCube(Cube=(MeanGammaStack-self.CubeSimul)/StdGammaStack)

        
        return
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








        
    # def runMCMC(self,g0,NBurn=100):
        
    #     T=ClassTimeIt.ClassTimeIt()
    #     T.disable()
    #     gStart=g0.copy()
    #     g=g0
    #     L=self.CLM.L(g)
        
    #     GM=self.CLM.MassFunction.GammaMachine
    #     L_NParms=GM.L_NParms
    #     ssqs=np.concatenate(self.GM.L_ssqs)
        
    #     iStep=0
    #     iAccepted=0
    #     Alpha=1.
    #     L_L=[L]
    #     L_g=[g]
    #     L_Accept=[]
    #     while True:
    #         T.reinit()

    #         T.timeit("Compute L")
    #         g1=g+Alpha*np.random.randn(*g.shape)/ssqs
    #         L1=self.CLM.L(g1)
    #         if L1>L:
    #             AcceptThis=True
    #         else:
    #             P=np.exp(L1-L)
    #             if np.random.rand(1)[0]<P:
    #                 AcceptThis=True
    #             else:
    #                 AcceptThis=False
    #         L_Accept.append(AcceptThis)
    #         NSurveyAcc=100
    #         print(iStep)
    #         if len(L_Accept)>NSurveyAcc and iStep%10==0:
    #             Frac=1.2
    #             L_Accept_s=L_Accept[-NSurveyAcc:]
    #             fAccepted=np.count_nonzero(L_Accept_s)/len(L_Accept_s)
    #             log.print("Accepted fraction %.1f %%"%(100*fAccepted))

    #             if fAccepted<0.2:
    #                 log.print("  decreasing alpha %.5f -> %.5f"%(Alpha,Alpha/Frac))
    #                 Alpha/=Frac
    #             elif fAccepted>0.5:
    #                 log.print("  increasing alpha %.5f -> %.5f"%(Alpha,Alpha*Frac))
    #                 Alpha*=Frac
                    
                              
            
    #         if AcceptThis:
    #             if iAccepted>NBurn:
    #                 g=g1.copy()
    #                 L_g.append(g1)
    #                 L_L.append(L1)
    #             iAccepted+=1
                
                
    #         iStep+=1
            
    #         if iStep>2000:
    #             break
            
    #     # End while
    #     NStack=len(L_g)#100
    #     MeanGammaStack=np.zeros_like(self.GM.GammaCube)
    #     for i in range(NStack):
    #         MeanGammaStack+=self.GM.giveGammaCube(L_g[i-NStack])
    #     MeanGammaStack/=NStack
        
    #     StdGammaStack=np.zeros_like(self.GM.GammaCube)
    #     for i in range(NStack):
    #         StdGammaStack+=(self.GM.giveGammaCube(L_g[i-NStack])-MeanGammaStack)**2
    #     StdGammaStack/=NStack
    #     StdGammaStack=np.sqrt(StdGammaStack)

    #     # ##########################
    #     CubeStart=self.GM.giveGammaCube(gStart)
    #     e_Gamma=StdGammaStack
    #     Gamma=MeanGammaStack
    #     y0=Gamma.flatten()-e_Gamma.flatten()
    #     y1=Gamma.flatten()+e_Gamma.flatten()
    #     x=np.arange(y0.size)

    #     ax=pylab.subplot(1,1,1)
    #     ax.fill_between(x,y0,y1, facecolor='gray', alpha=0.5)
    #     pylab.plot(Gamma.flatten(),color="black")
    #     pylab.plot(CubeStart.flatten(),ls=":",color="black")
    #     pylab.plot(self.CubeSimul.flatten(),ls="--",color="black")
    #     pylab.plot(ssqs)
    #     # ax=pylab.subplot(1,2,2)
    #     # y0=Gamma.flatten()-e_Gamma.flatten()
    #     # y1=Gamma.flatten()+e_Gamma.flatten()
    #     # ys=self.CubeSimul.flatten()
    #     # x=np.arange(y0.size)
    #     # ax.fill_between(x,y0-ys,y1-ys, facecolor='gray', alpha=0.5)
    #     # pylab.plot(Gamma.flatten()-ys,color="black")
        
    #     pylab.draw()
    #     pylab.show(block=False)
    #     pylab.pause(0.1)

    #     return MeanGammaStack,StdGammaStack
        
        
