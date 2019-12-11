import numpy as np
import ClassMassFunction
import ClassCatalogMachine
import ClassDisplayRGB
from DDFacet.Other import ClassTimeIt
from DDFacet.ToolsDir.ModToolBox import EstimateNpix
np.random.seed(1)
from DDFacet.Other import logger
log = logger.getLogger("ClassLikelihoodMachine")

import ClassInitGammaCube

def test():
    CM=ClassCatalogMachine.ClassCatalogMachine()
    CM.Init()
    CLM=ClassLikelihoodMachine(CM)
    #CLM.showRGB()
    
    
    # CIGC=ClassInitGammaCube.ClassInitGammaCube(CLM)
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
        
        rac,decc=241.20678,55.59485 # cluster
        self.FOV=0.15
        
        self.CellDeg=0.001/1.
        self.CellRad=self.CellDeg*np.pi/180

        
        self.NPix=int(self.FOV/self.CellDeg)
        self.NPix, _ = EstimateNpix(float(self.NPix), Padding=1)
        self.NPix=148
        print>>log,"Choosing NPix=%i"%self.NPix
        self.ScaleKpc=[500]
        self.rac_deg,self.decc_deg=rac,decc
        self.rac,self.decc=rac*np.pi/180,decc*np.pi/180
        
        self.zParms=self.CM.zg_Pars
        self.logMParms=self.CM.logM_Pars
        self.logM_g=np.linspace(*self.logMParms)
        self.CM.cutCat(self.rac,self.decc,self.NPix,self.CellRad)

        # self.CM.MaskArrayCube=self.CM.MaskArray.

        self.MassFunction=ClassMassFunction.ClassMassFunction()
        
        self.MassFunction.setGammaGrid((self.CM.rac_main,self.CM.decc_main),
                                       (self.rac,self.decc),
                                       self.CellDeg,
                                       self.NPix,
                                       zParms=self.zParms,
                                       ScaleKpc=self.ScaleKpc)
        self.MassFunction.setSelectionFunction(self.CM)

        self.NSlice=self.zParms[-1]-1
        self.IndexCube=np.array([i*self.NPix**2+self.CM.Cat_s.yCube*self.NPix+self.CM.Cat_s.xCube for i in range(self.NSlice)]).flatten()
        
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

        
    def log_prob(self, X):
        T=ClassTimeIt.ClassTimeIt()
        T.disable()
        self.MassFunction.updateGammaCube(X)
        T.timeit("update cube")
        GammaCube=self.MassFunction.GammaMachine.GammaCube
        n_z=self.CM.DicoDATA["DicoSelFunc"]["n_z"]
        Nx=np.sum(GammaCube*n_z.reshape((-1,1,1)),axis=0)
        T.timeit("Nx")
        Nx_Omega0=np.sum(Nx)*self.CellRad**2
        T.timeit("Nx_Omega0")
        
        Ns=self.CM.Cat_s.shape[0]
        gamma_xz=GammaCube.flat[self.IndexCube].reshape((self.NSlice,Ns)).T
        T.timeit("gamma_xz")
        
        Nx_Omega1=np.sum(gamma_xz*n_z.reshape((1,-1)))
        T.timeit("Nx_Omega1")
        Nx1_Omega1=np.sum(np.log(np.sum(gamma_xz*self.CM.Cat_s.n_zt,axis=1)))
        T.timeit("Nx1_Omega1")

        return -Nx_Omega0+Nx_Omega1+Nx1_Omega1

        
        
    # def log_prob(self, x):

    #     #self.MassFunction.
    #     z0,z1=self.z0z1
    #     X=self.Cat_s.x
    #     Y=self.Cat.y
        
    #     ni=self.Cat.n_i
    #     iz=self.Cat.iz
    #     T=ClassTimeIt.ClassTimeIt()
    #     T.disable()
    #     # ######################################
    #     # Init Mass Function for that one X
    #     if self.MassFuncLogProb is None:
    #         self.MassFuncLogProb=ClassMassFunction.ClassMassFunction()
    #         T.timeit("MassFunc")
    #         self.MassFuncLogProb.setGammaFunction((self.rac,self.decc),
    #                                               self.CellDeg,
    #                                               self.NPix,
    #                                               z=self.zParms,
    #                                               ScaleKpc=self.ScaleKpc)
            
    #     MassFunc=self.MassFuncLogProb
    #     T.timeit("SetGammaFunc")
    #     LX=[x]
    #     MassFunc.CGM.computeGammaCube(LX)
    #     GammaSlice=MassFunc.CGM.GammaCube[0]
    #     T.timeit("ComputeGammaFunc")
    #     # ######################################

    #     OmegaSr=((1./3600)*np.pi/180)**2
    #     L=0
    #     for iLogM in range(self.logM_g.size-1):
    #         logM0,logM1=self.logM_g[iLogM],self.logM_g[iLogM+1]
    #         logMm=(logM0+logM1)/2.
    #         zm=(z0+z1)/2.
    #         dz=z1-z0
    #         dlogM=logM1-logM0
    #         dV_dz=cosmo.differential_comoving_volume(zm).to_value()
    #         T.timeit("Cosmo")
    #         V=dz*dV_dz*self.CellRad**2
            
    #         n0=MassFunc.givePhiM(zm,logMm)*dlogM*V
    #         T.timeit("n0")
    #         L+=-np.sum(GammaSlice)*n0
    #         for iS in range(self.Cat.ra.size):
    #             n=MassFunc.give_N((self.Cat.ra[iS],self.Cat.dec[iS]),
    #                               (z0,z1),
    #                               (logM0,logM1),
    #                               OmegaSr)
    #             L+=np.log(n)

    #         T.timeit("log(n)")
    #     return L

