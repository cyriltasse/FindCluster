import ClassGammaMachine
from astropy.cosmology import WMAP9 as cosmo
import numpy as np
from DDFacet.Other import ClassTimeIt

def testFontana():

    z=np.linspace(0,3,10)
    M=np.linspace(8,12,100)
    pylab.clf()
    for iz in range(z.size):
        Phi=givePhiM(z[iz],M)
        pylab.plot(M,np.log10(Phi))
        pylab.draw()
        pylab.show(False)
        pylab.pause(0.1)
        

class ClassMassFunction():
    def __init__(self,Model="Fontana"):
        self.Model=Model
        self.CGM=None

    def setGammaFunction(self,
                         radec,
                         CellDeg,
                         NPix,
                         z=[0.01,2.,40],
                         ScaleKpc=500,LX=None):
        Mode="ConvGaussNoise"
        Mode="ConvPaddedFFT"
        CGM=ClassGammaMachine.ClassGammaMachine(radec,
                                                CellDeg,
                                                NPix,
                                                z=z,ScaleKpc=ScaleKpc,
                                                Mode=Mode)

        self.CGM=CGM
        if LX is not None: self.GammaCube=self.CGM.computeGammaCube(LX)
        
    def givePhiM(self,z,M):
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
        return Phi/H**3

    def give_N(self,(ra,dec),(z0,z1),(logM0,logM1),OmegaSr):
        # print ra,dec
        zm=(z0+z1)/2.
        dz=z1-z0
        dV_dz=cosmo.differential_comoving_volume(zm).to_value()
        
        V=dz*dV_dz*OmegaSr
        
        dlogM=logM1-logM0
        Mm=(logM1+logM0)/2.
        Phi=self.givePhiM(zm,Mm)

        n0=Phi*dlogM*V

        n=n0
        if self.CGM:
            G=self.CGM.giveGamma(zm,ra,dec)
            n*=G
            
        # print "  V=%f"%V
        # print "  G=%f"%G
        # print ra,dec,z0,z1,logM0,logM1,n
        return n
