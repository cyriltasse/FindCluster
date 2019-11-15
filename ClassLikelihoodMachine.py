import numpy as np
import ClassMassFunction

def test():
    CLM=ClassLikelihoodMachine()
    

class ClassLikelihoodMachine():
    def __init__(self):
        rac,decc=241.20678,55.59485 # cluster
        self.CellDeg=0.001
        self.CellRad=self.CellDeg*np.pi/180
        self.NPix=101
        self.ScaleKpc=500
        self.rac_deg,self.decc_deg=rac,decc
        self.rac,self.decc=rac*np.pi/180,decc*np.pi/180
        self.zParms=[0.6,0.7,2]
        self.logMParms=[10,10.5,2]
        self.logM_g=np.linspace(*self.logMParms)
        self.MassFunction=ClassMassFunction.ClassMassFunction()
        self.MassFunction.setGammaGrid((self.rac,self.decc),
                                       self.CellDeg,
                                       self.NPix,
                                       z=self.zParms,
                                       ScaleKpc=self.ScaleKpc)

    def log_prob(self, x):

        self.MassFunction.
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

            T.timeit("log(n)")
        return L

    def setData(self,DicoData):
        self.DicoData=DicoData

        
        
