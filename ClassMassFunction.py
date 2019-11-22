import ClassGammaMachine
from astropy.cosmology import WMAP9 as cosmo
import numpy as np
from DDFacet.Other import ClassTimeIt
import pylab
import ClassSelectionFunction

def testPhiM():
    z=np.linspace(0,3,10)
    M=np.linspace(8,12,100)
    CMF=ClassMassFunction()
    pylab.clf()
    for iz in range(z.size):
        Phi=CMF.givePhiM(z[iz],M)
        pylab.plot(M,np.log10(Phi))
        pylab.draw()
        pylab.show(False)
        pylab.pause(0.1)
        



        
class ClassMassFunction():
    def __init__(self,Model="Leja19"):
        self.Model=Model
        self.CGM=None
        self.GSF=None
        
    def setSelectionFunction(self,CM):
        self.GSF=ClassSelectionFunction.ClassSelectionFunction(CM)

        
    def setGammaGrid(self,
                     radec,
                     CellDeg,
                     NPix,
                     zParms=None,
                     ScaleKpc=500):
        Mode="ConvGaussNoise"
        Mode="ConvPaddedFFT"
        CGM=ClassGammaMachine.ClassGammaMachine(radec,
                                                CellDeg,
                                                NPix,
                                                zParms=zParms,
                                                ScaleKpc=ScaleKpc,
                                                Mode=Mode)

        self.GammaMachine=CGM

    def updateGammaCube(self,X):
        self.GammaMachine.computeGammaCube(X)

        
    def givePhiM(self,z,M):
        if self.Model=="Leja19":
            return self.givePhiM_Leja19(z,M)
        elif self.Model=="Fontana06":
            return self.givePhiM_Fontana06(z,M)
        
    def givePhiM_Leja19(self,z,M):
        
        Phi1_c0= -2.44
        Phi1_c1= -3.08
        Phi1_c2= -4.14
        Phi2_c0= -2.89
        Phi2_c1= -3.29
        Phi2_c2= -3.51
        Ms_c0=10.79
        Ms_c1=10.88
        Ms_c2=10.84
        Alpha1=-0.28
        Alpha2=-1.48


        def parameter_at_z0(y,z0,z1=0.2,z2=1.6,z3=3.0):
            y1, y2, y3 = y
            a = (((y3 - y1) + (y2 - y1) / (z2 - z1) * (z1 - z3)) /
                 (z3**2 - z1**2 + (z2**2 - z1**2) / (z2 - z1) * (z1 - z3)))
            b = ((y2 - y1) - a * (z2**2 - z1**2)) / (z2 - z1)
            c = y1 - a * z1**2 - b * z1
            return a * z0**2 + b * z0 + c
        
        Phi1_C=(Phi1_c0,Phi1_c1,Phi1_c2)
        Phi2_C=(Phi2_c0,Phi2_c1,Phi2_c2)
        Ms_C=(Ms_c0,Ms_c1,Ms_c2)

        def Phi(z,M,Phi_C,Ms_C,Alpha):
            Phi_s = 10**(parameter_at_z0(Phi_C,z))
            M_s   = parameter_at_z0(Ms_C,z)
            return np.log(10) * Phi_s * 10**((M-M_s)*(Alpha+1)) * np.exp(-10**(M-M_s))
            #return Phi_s * np.log(10) * (10**(M-M_s))**(1+Alpha) * np.exp(-10**(M-M_s))
        Phi1=Phi(z,M,Phi1_C,Ms_C,Alpha1)
        Phi2=Phi(z,M,Phi2_C,Ms_C,Alpha2)

        return Phi1+Phi2
    
    def givePhiM_Fontana06(self,z,M):
        M0s= 11.16
        M1s= +0.17
        M2s= -0.07
        alpha0s= -1.18
        alpha1s= -0.082
        Phi0s= 0.0035
        Phi1s= -2.20
        H=0.7
        # From Fontana et al. 06
        def Phi_s(z): return Phi0s*(1+z)**Phi1s
        def M_s(z): return M0s+M1s*z+M2s*z**2
        def alpha_s(z): return alpha0s+alpha1s*z
        Phi=Phi_s(z) * np.log(10) * (10**(M-M_s(z)))**(1+alpha_s(z)) * np.exp(-10**(M-M_s(z)))
        return Phi#/H**3

    def give_N(self,(ra,dec),(z0,z1),(logM0,logM1),OmegaSr):
        # print ra,dec
        zm=(z0+z1)/2.
        dz=z1-z0
        dV_dz=cosmo.differential_comoving_volume(zm).to_value()
        
        V=dz*dV_dz*OmegaSr
        
        dlogM=logM1-logM0
        logMm=(logM1+logM0)/2.
        Phi=self.givePhiM(zm,logMm)

        n0=Phi*dlogM*V

        n=n0
        if self.GammaMachine:
            G=self.GammaMachine.giveGamma(zm,ra,dec)
            n*=G

        if self.GSF:
            s=self.GSF.giveSelFunc(self,zm,logMm)
            n*=s
            
        # print "  V=%f"%V
        # print "  G=%f"%G
        # print ra,dec,z0,z1,logM0,logM1,n
        return n
