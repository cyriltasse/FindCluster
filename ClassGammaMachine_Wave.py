

from astropy.cosmology import WMAP9 as cosmo
import numpy as np
import scipy.signal
import ClassFFT
import matplotlib.pyplot as pylab
from DDFacet.ToolsDir import ModCoord
from DDFacet.Other import ClassTimeIt
from DDFacet.Other import logger
log = logger.getLogger("ClassGammaMachine")
import pywt

class ClassGammaMachine():
    def __init__(self,
                 radec_main,
                 radec,
                 CellDeg,
                 NPix,
                 zParms=[0.01,2.,40],
                 Mode="ConvGaussNoise",
                 ScaleKpc=100):
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

        nn=NPix//2
        lg,mg=np.mgrid[-nn*self.CellRad+l0:nn*self.CellRad+l0:1j*self.NPix,-nn*self.CellRad+m0:nn*self.CellRad+m0:1j*self.NPix]
        rag,decg=self.CoordMachine.lm2radec(lg.flatten(),mg.flatten())
        self.rag=rag.reshape(lg.shape)
        self.decg=decg.reshape(lg.shape)
        self.lg,self.mg=lg,mg

        self.GammaCube=None
        self.NSlice=self.zg.size-1
        self.SliceFunction=self.giveSlice_Wave
        self.L_NParms=[]
        self.setWaveType(Kind="haar")


    def setWaveType(self,Kind="db10",Level=5,Th=1e-2,Mode='periodization'):
        # Type="bior1.3"
        # Type="db10"
        # Type="coif10"
        # Type="sym2"
        # Type="rbio1.3"
        # Type="haar"
        self.Kind=Kind
        self.Mode=Mode
        self.Level=Level
        self.Th=Th
        print>>log,"Using wavelet kind %s [level=%i] with a threshold of %f"%(Kind,Level,Th)

    def setReferenceCube(self,Cube):
        level=7
        self.L_Ind=[]
        self.L_Slice=[]
        self.L_arrShape=[]
        for iSlice in range(Cube.shape[0]):
            Slice=Cube[iSlice]
            z0,z1=self.zg[iSlice],self.zg[iSlice+1]
            zm=(z0+z1)/2.
            c = pywt.wavedec2(np.real(Slice),
                              self.Kind,
                              mode=self.Mode,
                              level=self.Level)
            arr, slices = pywt.coeffs_to_array(c)

            abs_arr=np.abs(arr).flatten()
            ind=np.where(abs_arr > self.Th*abs_arr.max())[0]
            NParms=ind.size
            self.L_NParms.append(NParms)
            self.L_Ind.append(ind)
            self.L_Slice.append(slices)
            self.L_arrShape.append(arr.shape)
            print>>log,"  -- Slice #%i zm=%.2f [z=%.2f -> z=%.2f]: %i parameters"%(iSlice,zm,z0,z1,NParms)

            
        self.NParms=np.sum(self.L_NParms)
        print>>log,"Total number of free parameters: %i"%self.NParms


    def CubeToVec(self,Cube):
        ii=0
        Cube=np.complex64(Cube.copy())
        XOut=np.zeros((self.NParms,),np.float32)
        for iSlice in range(self.NSlice):
            Slice=Cube[iSlice]
            z0,z1=self.zg[iSlice],self.zg[iSlice+1]
            zm=(z0+z1)/2.
            c = pywt.wavedec2(np.real(Slice),
                              self.Kind,
                              mode=self.Mode,
                              level=self.Level)
            arr, slices = pywt.coeffs_to_array(c)
            x=arr.flat[self.L_Ind[iSlice]]
            N=self.L_NParms[iSlice]
            XOut[ii:ii+N]=x[:]
            ii+=N
        return XOut
            
            
    def PlotGammaCube(self,Cube=None,FigName="Gamma Cube"):
        if Cube is None:
            Cube=self.GammaCube
        import pylab
        pylab.figure(FigName)
        pylab.clf()
        for iPlot in range(9):
            S=Cube[iPlot]
            pylab.subplot(3,3,iPlot+1)
            pylab.imshow(S,interpolation="nearest")#,vmin=0.,vmax=2.)
            pylab.title("[%f - %f]"%(S.min(),S.max()))
            pylab.draw()
            pylab.show(False)
            pylab.pause(0.1)


    def computeGammaCube(self,X):
        LX=[]
        ii=0
        T=ClassTimeIt.ClassTimeIt("Gamma")
        T.disable()
        for iSlice in range(self.NSlice):
            LX.append(X[ii:ii+self.L_NParms[iSlice]])
            ii+=self.L_NParms[iSlice]
        T.timeit("unpack")
        GammaCube=np.zeros((self.zg.size-1,self.NPix,self.NPix),np.float32)
        for iz in range(self.zg.size-1):
            z0,z1=self.zg[iz],self.zg[iz+1]
            GammaCube[iz,:,:]=self.SliceFunction(LX,iz)
        T.timeit("Slices")
        self.GammaCube=GammaCube

    def giveSlice_Wave(self,LX,iSlice):

        T=ClassTimeIt.ClassTimeIt("Slice")
        T.disable()
        x=LX[iSlice]
        z0=self.zg[iSlice]
        z1=self.zg[iSlice+1]

        arr=np.zeros((self.L_arrShape[iSlice]),np.float32)
        arr.flat[self.L_Ind[iSlice]]=x[:]
        slices=self.L_Slice[iSlice]
        
        c=pywt.array_to_coeffs(arr,slices,output_format='wavedec2')
        Slice=pywt.waverec2(c, self.Kind, mode=self.Mode)


        Slice=np.real(Slice)

        # pylab.clf()
        # pylab.imshow(GammaCube[iz,:,:],interpolation="nearest",vmin=0,vmax=2.)
        # pylab.draw()
        # pylab.show(False)
        # pylab.pause(0.1)
        
        Slice[Slice<0]=1e-10
        return Slice

    

