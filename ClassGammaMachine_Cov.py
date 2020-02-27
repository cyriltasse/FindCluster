from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from astropy.cosmology import WMAP9 as cosmo
import numpy as np
import scipy.signal
import ClassFFT
import os
import matplotlib.pyplot as pylab
from DDFacet.ToolsDir import ModCoord
from DDFacet.Other import ClassTimeIt
from DDFacet.Other import logger
from DDFacet.Other import MyPickle
from DDFacet.Array import ModLinAlg
log = logger.getLogger("ClassGammaMachine")
import ClassCovMatrix

class ClassGammaMachine():
    def __init__(self,
                 radec_main,
                 radec,
                 CellDeg,
                 NPix,
                 zParms=[0.01,2.,40],
                 Mode="ConvGaussNoise",
                 ScaleKpc=100):
        self.DoPrint=True
        self.radec=radec
        self.rac,self.decc=self.radec
        self.CellDeg=CellDeg
        self.NPix=NPix
        self.CellRad=CellDeg*np.pi/180
        self.zg=np.linspace(*zParms)
        self.Mode=Mode
        self.ScaleKpc=ScaleKpc
        self.rac_main,self.decc_main=self.radec_main=radec_main

        self.CoordMachine = ModCoord.ClassCoordConv(self.rac_main, self.decc_main)
        l0,m0=self.CoordMachine.radec2lm(self.rac, self.decc)
        #log.print( (l0,m0))
        N0=(NPix)//2
        N1=NPix-1-N0
        self.N0N1=(N0,N1)
        #log.print( (N0,N1))
        lg,mg=np.mgrid[-N0*self.CellRad+l0:N1*self.CellRad+l0:1j*self.NPix,-N0*self.CellRad+m0:N1*self.CellRad+m0:1j*self.NPix]
        rag,decg=self.CoordMachine.lm2radec(lg.flatten(),mg.flatten())
        self.rag=rag.reshape(lg.shape)
        self.decg=decg.reshape(lg.shape)
        self.lg,self.mg=lg,mg

        self.GammaCube=None
        self.NSlice=self.zg.size-1

        self.L_NParms=[]
        self.HasReferenceCube=False
        self.initCovMatrices(ScaleFWHMkpc=ScaleKpc)
        log.print("Number of free parameters %i"%self.NParms)
                             
    def initCovMatrices(self,
                        ScaleFWHMkpc=100.,
                        Type="Normal"):
        
        FileOut="Cov_%s_%ikpc.DicoCov"%(Type,ScaleFWHMkpc)
        
        Parms={"CellDeg":self.CellDeg,
               "NPix":self.NPix,
               "CellRad":self.CellRad,
               "zg":self.zg.tolist()}
        
        if os.path.isfile(FileOut):
            log.print("Loading %s..."%FileOut)
            DicoSave=MyPickle.Load(FileOut)
            if DicoSave["Parms"]==Parms:
                log.print("   all parameters are the same... ")
                self.L_SqrtCov=DicoSave["L_SqrtCov"]
                self.L_NParms=DicoSave["L_NParms"]
                self.L_Hinv=DicoSave["L_Hinv"]
                self.NParms=np.sum(self.L_NParms)
                return
            
        f=2.*np.sqrt(2.*np.log(2.))
        Sig_kpc=ScaleFWHMkpc/f
        self.L_SqrtCov=[]
        self.L_Hinv=[]
        
        for iz in range(self.NSlice):
            z0,z1=self.zg[iz],self.zg[iz+1]
            zm=(z0+z1)/2.
            SigmaRad=Sig_kpc*cosmo.arcsec_per_kpc_comoving(zm).to_value()/(3600.)*np.pi/180
            CCM=ClassCovMatrix.ClassCovMatrix(NPix=self.NPix,SigmaPix=SigmaRad/self.CellRad)
            CCM.buildGaussianCov()
            A=CCM.sqrtCs
            Hinv=ModLinAlg.invSVD((A.T).dot(A))
            self.L_Hinv.append(Hinv)
            self.L_SqrtCov.append(CCM.sqrtCs)
            self.L_NParms.append(CCM.k)
            
        self.NParms=np.sum(self.L_NParms)
        
        DicoSave={"Parms":Parms,
                  "L_SqrtCov":self.L_SqrtCov,
                  "L_NParms":self.L_NParms,
                  "L_Hinv":self.L_Hinv}
        
        log.print("Saving sqrt(Cov) in %s"%FileOut)
        MyPickle.Save(DicoSave,FileOut)
        
        
    def CubeToVec(self,Cube):
        ii=0
        Cube=np.complex64(Cube.copy())
        XOut=np.zeros((self.NParms,),np.float32)
        for iSlice in range(self.NSlice):
            N=self.L_NParms[iSlice]
            if N==0: continue
            Slice=Cube[iSlice]
            y=Slice.reshape((-1,1))-1.
            A=self.L_SqrtCov[iSlice]
            x0=(A.T).dot(y)
            Hinv=self.L_Hinv[iSlice]#ModLinAlg.invSVD((A.T).dot(A))
            x=np.dot(Hinv,x0)
            # print(np.mean(np.abs(x.imag)) / np.mean(np.abs(x.real)))
            XOut[ii:ii+N]=(x.real.ravel())[:]
            ii+=N
        return XOut
            
    def PlotGammaCube(self,Cube=None,FigName="Gamma Cube",OutName=None):
        if Cube is None:
            Cube=self.GammaCube
        import pylab
        fig=pylab.figure(FigName)
        pylab.clf()
        for iPlot in range(9):
            S=Cube[iPlot]
            pylab.subplot(3,3,iPlot+1)
            pylab.imshow(S,interpolation="nearest",vmin=0.,vmax=10.)
            pylab.title("[%f - %f]"%(S.min(),S.max()))
        pylab.draw()
        pylab.show(False)
        pylab.pause(0.1)
        if OutName: fig.savefig(OutName)

    def computeGammaCube(self,X):
        LX=[]
        ii=0
        T=ClassTimeIt.ClassTimeIt("Gamma")
        T.disable()
        for iSlice in range(self.NSlice):
            N=self.L_NParms[iSlice]
            if N==0:
                LX.append([])
                continue
            #print ii,iSlice,self.NSlice,len(self.L_NParms)
            LX.append(X[ii:ii+self.L_NParms[iSlice]])
            ii+=N
        T.timeit("unpack")
        GammaCube=np.zeros((self.zg.size-1,self.NPix,self.NPix),np.float32)
        for iz in range(self.zg.size-1):
            A=self.L_SqrtCov[iz]
            x=LX[iz].reshape((-1,1))
            y=(A.dot(x)).reshape((self.NPix,self.NPix))
            GammaCube[iz,:,:]=1.+y
        T.timeit("Slices")
        self.GammaCube=GammaCube

