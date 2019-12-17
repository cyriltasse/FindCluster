import numpy as np
import ClassMassFunction
import ClassCatalogMachine
import ClassDisplayRGB
from DDFacet.Other import ClassTimeIt
from DDFacet.ToolsDir.ModToolBox import EstimateNpix
np.random.seed(1)
from DDFacet.Other import logger
log = logger.getLogger("ClassLikelihoodMachine")
from DDFacet.Other.AsyncProcessPool import APP, WorkerProcessError

import ClassInitGammaCube

def test():
    CM=ClassCatalogMachine.ClassCatalogMachine()
    CM.Init()
    CLM=ClassLikelihoodMachine(CM)
    #CLM.showRGB()
    
    
    # CIGC=ClassInitGammaCube.ClassInitGammaCube(CLM,ScaleKpc=[200.,500.])
    # CIGC.finaliseInit()
    # Cube=CIGC.InitGammaCube()
    # np.save("Cube",Cube)
    
    Cube=np.load("Cube.npy")
    CLM.MassFunction.GammaMachine.PlotGammaCube(Cube=Cube,FigName="Measured")
    CLM.MassFunction.GammaMachine.setWaveType(Kind="haar",Th=3e-2,Mode="symmetric")
    #CLM.MassFunction.GammaMachine.setWaveType(Kind="rbio1.1",Th=1e-2,Mode="symmetric")
    CLM.MassFunction.GammaMachine.setReferenceCube(Cube)
    
    #Cube.fill(0.)
    #Cube[:,100,100]=1.
    X=CLM.MassFunction.GammaMachine.CubeToVec(Cube)

    # # ##############################
    # # Plot PSF
    # X.fill(0)
    # ii=0
    # for iSlice in range(len(CLM.MassFunction.GammaMachine.L_NParms)):
    #     n=CLM.MassFunction.GammaMachine.L_NParms[iSlice]
    #     X[ii]=1.
    #     X[ii:ii+(n-1)/2]=1.
    #     ii+=n
    # # ##############################


    # CLM.MassFunction.updateGammaCube(X)
    # Cr=CLM.MassFunction.GammaMachine.GammaCube-Cube
    # C=CLM.MassFunction.GammaMachine.GammaCube
    # CLM.MassFunction.GammaMachine.PlotGammaCube(Cube=C,FigName="Filtered")
    

    #X=np.random.randn(CLM.MassFunction.GammaMachine.NParms)


    CLM.MassFunction.updateGammaCube(X)
    CLM.MassFunction.GammaMachine.PlotGammaCube()


    T=ClassTimeIt.ClassTimeIt()
    for i in range(1000):
        CLM.log_prob(X)
    T.timeit()
    stop

class ClassLikelihoodMachine():
    def __init__(self,CM):
        self.CM=CM
        
        self.ScaleKpc=[500]
        
        self.zParms=self.CM.zg_Pars
        self.logMParms=self.CM.logM_Pars
        self.logM_g=np.linspace(*self.logMParms)

        # self.CM.MaskArrayCube=self.CM.MaskArray.

        self.MassFunction=ClassMassFunction.ClassMassFunction()
        
        self.MassFunction.setSelectionFunction(self.CM)

        self.NSlice=self.zParms[-1]-1
        APP.registerJobHandlers(self)

    def ComputeIndexCube(self,NPix):
        #self.IndexCube=np.array([i*NPix**2+self.CM.Cat_s.yCube*NPix+self.CM.Cat_s.xCube for i in range(self.NSlice)]).flatten()
        self.IndexCube=np.array([self.CM.Cat_s.yCube*NPix+self.CM.Cat_s.xCube]).flatten()
        
        
    def log_prob(self, X):
        T=ClassTimeIt.ClassTimeIt()
        T.disable()
        self.MassFunction.updateGammaCube(X)
        T.timeit("update cube")
        Cell=(1./3600)*np.pi/180
        # GammaCube=self.MassFunction.GammaMachine.GammaCube.copy()
        # for iz in range(self.NSlice):
        #     GammaCube[iz]=GammaCube[iz].T[::-1,:]
        GammaCube=self.MassFunction.GammaMachine.GammaCube
        n_z=self.CM.DicoDATA["DicoSelFunc"]["n_z"]
        Nx=np.sum(GammaCube*n_z.reshape((-1,1,1)),axis=0)
        T.timeit("Nx")
        Nx_Omega0=np.sum(Nx)*self.MassFunction.GammaMachine.CellRad**2
        T.timeit("Nx_Omega0")
        
        Ns=self.CM.Cat_s.shape[0]
        
        #gamma_xz=GammaCube.flat[self.IndexCube].reshape((self.NSlice,Ns)).T
        
        #gamma_xz=np.array([GammaCube[iz].flat[self.IndexCube] for iz in range(self.NSlice)]).reshape((self.NSlice,Ns)).T

        gamma_xz=np.zeros((self.NSlice,Ns),np.float64)
        for iS in range(self.CM.Cat_s.shape[0]):
            gamma_xz[:,iS]=GammaCube[:,self.CM.Cat_s.xCube[iS],self.CM.Cat_s.yCube[iS]]
            
        gamma_xz=gamma_xz.T
        T.timeit("gamma_xz")

        # gamma_xz2=np.zeros_like(GammaCube)
        # gamma_xz2[:,self.CM.Cat_s.xCube[iS],self.CM.Cat_s.yCube[iS]]=32
        # import pylab
        # pylab.figure("test")
        # pylab.clf()
        # pylab.imshow(gamma_xz2[1],interpolation="nearest")
        # pylab.draw()
        # pylab.show(False)
        
        Nx_Omega1=np.sum(gamma_xz*n_z.reshape((1,-1)))*Cell**2
        T.timeit("Nx_Omega1")
        S0=np.sum(gamma_xz*self.CM.Cat_s.n_zt*Cell**2,axis=1)
        S0[S0<=0]=1e-100
        Nx1_Omega1=np.sum(np.log(S0))
        T.timeit("Nx1_Omega1")
        L=-np.float64(Nx_Omega0)+np.float64(Nx_Omega1)+np.float64(Nx1_Omega1)

        return L,-Nx_Omega0,Nx_Omega1,Nx1_Omega1

        
