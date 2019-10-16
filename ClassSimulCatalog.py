#!/usr/bin/env python
import numpy as np
import ClassGeneDist
import pylab
import numpy as np
import pylab
import tables
import astropy.io.fits as pyfits
from astropy.cosmology import WMAP9 as cosmo
import tables
import matplotlib.pyplot as pylab
from astropy.wcs import WCS
from astropy.io import fits as fitsio
from pyrap.images import image as CasaImage
from DDFacet.Other import logger
log = logger.getLogger("ClassOverdensityMap")
#from DDFacet.Array import shared_dict
#from DDFacet.Other.AsyncProcessPool import APP, WorkerProcessError
#from DDFacet.Other import Multiprocessing
from DDFacet.Other import ModColor
from DDFacet.Other.progressbar import ProgressBar
#from DDFacet.Other import AsyncProcessPool
import ClassDisplayRGB
import ClassSaveFITS
from DDFacet.ToolsDir import ModCoord
import scipy.signal
import ClassMassFunction


def givePhiM(z,M):
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


class ClassSimulCatalog():
    def __init__(self,
                 ra,dec,
                 z=[0.01,2.,40],ScaleKpc=500,CellDeg=0.01,NPix=71,XSimul=None,logMParms=[10,10.5,2]):
        self.rac_deg,self.dec_deg=ra,dec
        self.rac=ra*np.pi/180
        self.decc=dec*np.pi/180
        self.CellDeg=CellDeg
        self.CellRad=CellDeg*np.pi/180
        boxDeg=CellDeg*NPix
        self.boxDeg=boxDeg
        self.boxRad=boxDeg*np.pi/180
        self.NPix=NPix
        self.ScaleKpc=ScaleKpc
        
        self.zg=np.linspace(*z)

        self.logM_g=np.linspace(*logMParms)
        
        self.MassFunc=ClassMassFunction.ClassMassFunction()

        self.MassFunc.setGammaFunction((self.rac,self.decc),
                                       CellDeg,
                                       NPix,
                                       z=z,
                                       ScaleKpc=ScaleKpc)

        self.rag=self.MassFunc.CGM.rag
        self.decg=self.MassFunc.CGM.decg

        self.XSimul=XSimul
        if self.XSimul is None:
            self.XSimul=np.random.randn(self.MassFunc.CGM.NParms)
            #self.XSimul.fill(0)
            self.XSimul[0]=10.
        LX=[self.XSimul]
        
        self.MassFunc.CGM.computeGammaCube(LX)
        

    def doSimul(self):
        Cra=[]
        Cdec=[]
        CM=[]
        Cx=[]
        Cy=[]
        CPhi_i=[]
        Ciz=[]
        for iz in range(self.zg.size-1)[::-1]:
            #print iz
            z0,z1=self.zg[iz],self.zg[iz+1]
            zm=(self.zg[iz]+self.zg[iz+1])/2.
            dz=self.zg[iz+1]-self.zg[iz]
            # dV_dz=cosmo.differential_comoving_volume(zm).to_value()

            # V=self.CellRad**2*dz*dV_dz
            # V2=self.CellRad**2/(4.*np.pi)*(cosmo.comoving_volume(z1).to_value()-cosmo.comoving_volume(z0).to_value())
            # stop
            
            ##ras=np.sort(self.rag.flatten())
            decs=np.sort(self.decg.flatten())
            ras=self.rag.T.flatten()
            decs=self.decg.flatten()
            dra=np.abs(np.median(ras[1::]-ras[0:-1]))
            ddec=np.abs(np.median(decs[1::]-decs[0:-1]))
            for iM in range(self.logM_g.size-1):
                M0,M1=self.logM_g[iM],self.logM_g[iM+1]
                dlogM=M1-M0
                Mm=(M1+M0)/2.
                # Phi=givePhiM(zm,Mm)
                # n0=Phi*dlogM*V
                # print Phi,dlogM,V,n0
                for ipix in range(self.NPix):
                    for jpix in range(self.NPix):
                        # #print "====",ipix,jpix
                        # G=self.MassFunc.CGM.GammaCube[iz,ipix,jpix]
                        # #print "V=%f"%V
                        # #print "G=%f"%G
                        # n=G*n0
                        # N=scipy.stats.poisson.rvs(n)

                        OmegaSr=self.CellRad**2
                        
                        n2=self.MassFunc.give_N((self.rag[ipix,jpix],self.decg[ipix,jpix]),
                                               (z0,z1),
                                               (M0,M1),
                                               OmegaSr)
                        N=scipy.stats.poisson.rvs(n2)
                        # print n,n2
                        # print zm,ipix,jpix,n,N
                        if N>=1:
                            ra=(np.random.rand(N)-0.5)*dra+self.rag[ipix,jpix]
                            dec=(np.random.rand(N)-0.5)*ddec+self.decg[ipix,jpix]
                            M=np.random.rand(N)*dlogM+M0
                            Cra+=ra.tolist()
                            Cdec+=dec.tolist()
                            CM+=M.tolist()
                            Cx+=[ipix]*ra.size
                            Cy+=[jpix]*dec.size
                            # CPhi_i+=[n0]*dec.size
                            Ciz+=[iz]*dec.size
                            #print Phi,dlogM,V,n0,n,N
        self.Cat=np.zeros((len(Cra),),dtype=[("ra",np.float32),("dec",np.float32),
                                             ("x",np.int32),("y",np.int32),
                                             ("iz",np.int32),
                                             ("logM",np.float32),
                                             ("n_i",np.float32)])
        self.Cat=self.Cat.view(np.recarray)
        self.Cat.ra[:]=np.array(Cra)
        self.Cat.dec[:]=np.array(Cdec)
        self.Cat.logM[:]=np.array(CM)
        self.Cat.x[:]=np.array(Cx)
        self.Cat.y[:]=np.array(Cy)
        # self.Cat.n_i[:]=np.array(CPhi_i)
        self.Cat.iz[:]=np.array(Ciz)
        


                            
                            
def testSimul():

    rac,decc=241.20678,55.59485 # cluster
    CellDeg=0.001
    NPix=5
    ScaleKpc=500
    CSC=ClassSimulCatalog(rac,decc,
                          #z=[0.01,2.,40],
                          z=[0.6,0.7,2],
                          ScaleKpc=ScaleKpc,CellDeg=CellDeg,NPix=NPix)
    CSC.doSimul()
    

if __name__=="__main__":
    testSimul()
    
