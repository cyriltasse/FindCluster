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
import ClassCovMatrix_Gauss
import ClassCovMatrix_Sim3D_2
import GeneDist


from DDFacet.Other.AsyncProcessPool import APP, WorkerProcessError
from DDFacet.Other import AsyncProcessPool
from DDFacet.Array import shared_dict
import ClassShapiroWilk
def GiveNXNYPanels(Ns,ratio=800/500):
    nx=int(round(np.sqrt(Ns/ratio)))
    #ny=int(nx*ratio)
    ny=Ns//nx
    if nx*ny<Ns: ny+=1
    return nx,ny

class ClassGammaMachine():
    def __init__(self,
                 radec_main,
                 radec,
                 CellDeg,
                 NPix,
                 zParms=[0.01,2.,40],
                 Mode="Sim3D",
                 ScaleKpc=100,
                 CM=None):
        self.CM=CM
        self.DoPrint=True
        self.radec=radec
        self.rac,self.decc=self.radec
        self.CellDeg=CellDeg
        self.NPix=NPix
        self.CellRad=CellDeg*np.pi/180
        self.zg=np.linspace(*zParms)
        self.zmg=(self.zg[0:-1]+self.zg[1:])/2.
        self.Mode=Mode
        self.ScaleKpc=ScaleKpc
        self.rac_main,self.decc_main=self.radec_main=radec_main
        self.CurrentX=None
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
        
        self.TypeSqrtC=np.float64
        self.TypeCube=np.float64
        
        # self.TypeSqrtC=np.float32
        # self.TypeCube=np.float32
        
        i0=int(lg.min()/self.CellRad+N0)
        i1=i0+NPix+1
        j0,j1=mg.min()/self.CellRad+N0,mg.max()/self.CellRad+N0
        self.ThisMask=np.zeros((NPix,NPix),np.float32)
        # self.ThisMask.fill(0)
        for i in range(NPix):
            for j in range(NPix):
                ii,jj=self.CM.RaDecToMaskPix(self.rag[i,j],self.decg[i,j])
                self.ThisMask[i,j]=self.CM.MaskArray[0,0,int(ii),int(jj)]
        
        self.GammaCube=None
        self.NSlice=self.zg.size-1

        self.L_NParms=[]
        self.HasReferenceCube=False
                             
    def setCovMat(self,DicoCov):
        self.MaxValueSigmoid=DicoCov["MaxValueSigmoid"]
        self.a_Sigmoid=DicoCov["a_Sigmoid"]
        self.ScaleCov=DicoCov["ScaleCov"]
        self.L_SqrtCov=DicoCov["L_SqrtCov"]
        self.L_NParms=DicoCov["L_NParms"]
        self.L_Hinv=DicoCov["L_Hinv"]
        self.L_ssqs=DicoCov["L_ssqs"]
        self.NParms=np.sum(self.L_NParms)


    def computeGammaSlice(self,X,iSliceCompute,ScaleCube="linear"):

        CurrentX=X.copy()
        CurrentScaleCube=ScaleCube
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
            if iSliceCompute==iSlice:
                i0,i1=ii,ii+self.L_NParms[iSlice]
            ii+=N
            
        T.timeit("unpack")
        GammaCube=np.zeros((self.NPix,self.NPix),self.TypeCube)
        for iz in [iSliceCompute]:
            A=self.L_SqrtCov[iz]
            x=LX[iz].reshape((-1,1))
            # y=(A.dot(x)).reshape((self.NPix,self.NPix))
            y=np.dot(self.TypeSqrtC(A),self.TypeSqrtC(x)).reshape((self.NPix,self.NPix))
            # GammaCube[iz,:,:]=1.+y
            
            if self.ScaleCov=="log":
                #print(ScaleCube)
                if ScaleCube=="linear":
                    GammaCube[:,:]=np.exp(y)
                elif ScaleCube=="log":
                    GammaCube[:,:]=y/np.log(10)
            elif self.ScaleCov=="Sigmoid":
                GammaCube[:,:]=ClassCovMatrix_Sim3D_2.Sigmoid(y,a=self.a_Sigmoid,MaxVal=self.MaxValueSigmoid)
                stop
        T.timeit("Slices")
        return GammaCube,i0,i1

        

    def computeGammaCube(self,X,ScaleCube="linear"):

        if self.CurrentX is not None:
            if np.allclose(self.CurrentX,X) and self.CurrentScaleCube==ScaleCube:
                #log.print("Current cube ok, skipping computation...")
                return
        self.CurrentX=X.copy()
        self.CurrentScaleCube=ScaleCube
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
        GammaCube=np.zeros((self.zg.size-1,self.NPix,self.NPix),self.TypeCube)
        for iz in range(self.zg.size-1):
            A=self.L_SqrtCov[iz]
            x=LX[iz].reshape((-1,1))
            #y=(A.dot(x)).reshape((self.NPix,self.NPix))
            y=np.dot(self.TypeSqrtC(A),self.TypeSqrtC(x)).reshape((self.NPix,self.NPix))
            # GammaCube[iz,:,:]=1.+y
            
            if self.ScaleCov=="log":
                #print(ScaleCube)
                if ScaleCube=="linear":
                    GammaCube[iz,:,:]=np.exp(y)
                elif ScaleCube=="log":
                    GammaCube[iz,:,:]=y/np.log(10)
            elif self.ScaleCov=="Sigmoid":
                GammaCube[iz,:,:]=ClassCovMatrix_Sim3D_2.Sigmoid(y,a=self.a_Sigmoid,MaxVal=self.MaxValueSigmoid)
                stop
        T.timeit("Slices")
        self.GammaCube=GammaCube

    def giveGammaCube(self,X,ScaleCube="linear"):
        self.computeGammaCube(X,ScaleCube=ScaleCube)
        return self.GammaCube
    
    def PlotGammaCube(self,X=None,Cube=None,FigName="Gamma Cube",OutName=None,ScaleCube="linear",vmm=None,DicoSourceXY=None,LSlice=None):
        # return
        if X is not None:
            self.computeGammaCube(X,ScaleCube=ScaleCube)
        if Cube is None:
            Cube=self.GammaCube


            
        import pylab

        fact=1.5
        figsize=(13/fact,8/fact)
        fig=pylab.figure(FigName,figsize=figsize)
        self.CurrentFig=fig
        self.AxList=[]

        if LSlice is None:
            LSlice=range(self.NSlice)

        Nx,Ny=GiveNXNYPanels(len(LSlice),ratio=figsize[0]/figsize[1])
        pylab.clf()
            
        for iPlot,iSlice in enumerate(LSlice):
            S=Cube[iSlice]
                
            ax=pylab.subplot(Nx,Ny,iPlot+1)
            self.AxList.append(ax)
            # if np.count_nonzero(np.isnan(S))>0:
            #     stop
            if vmm is not None:
                vmin,vmax=vmm[iSlice]
            else:
                Snn=S[np.logical_not(np.isnan(S))]
                if Snn.size>0:
                    vmin,vmax=Snn.min(),Snn.max()
                else:
                    vmin,vmax=-1.,1.
            cmap=pylab.cm.cubehelix
            cmap=pylab.cm.plasma
            cmap=pylab.cm.inferno
            cmap=pylab.cm.cividis 
            pylab.imshow(S,
                         interpolation="nearest",
                         vmin=vmin,vmax=vmax,
                         origin="lower",
                         cmap=cmap)
            #pylab.title("[%.2f - %.2f]"%(S.min(),S.max()))
            pylab.title("%.2f < z < %.2f"%(self.zg[iPlot],self.zg[iPlot+1]))
            if DicoSourceXY is not None:
                s=DicoSourceXY["P"][:,iSlice]
                rgba_colors = np.zeros((s.size,4))
                rgba_colors[:,1:3] = 0
                rgba_colors[:, 3] = s
                ax.scatter(DicoSourceXY["X"],DicoSourceXY["Y"],s=3, color=rgba_colors)#,c="black",2*s[:,iSlice])
                
        #pylab.tight_layout()
        pylab.draw()
        pylab.show(block=False)
        pylab.pause(0.1)
        if OutName: fig.savefig(OutName)
        
        # fig=pylab.figure("%s.Mask"%FigName)
        # pylab.clf()
        # for iPlot in range(9):
        #     pylab.subplot(3,3,iPlot+1)
        #     pylab.imshow(self.ThisMask,interpolation="nearest")#,vmin=0.,vmax=10.)
        #     pylab.title("[%f - %f]"%(S.min(),S.max()))
        # pylab.draw()
        # pylab.show(False)
        # pylab.pause(0.1)

    def PlotCumulDistX(self,X=None,FigName="Hist X",vmm=None):


            

        fact=1.8
        figsize=(13/fact,8/fact)
        fig=pylab.figure(FigName,figsize=figsize)
        Nx,Ny=GiveNXNYPanels(self.NSlice,ratio=figsize[0]/figsize[1])
        fig.clf()
        ii=0
        for iPlot in range(self.NSlice):
            iSlice=iPlot
            N=self.L_NParms[iSlice]
            x=X[ii:ii+self.L_NParms[iSlice]].flatten()
            ii+=N
            ax=pylab.subplot(Nx,Ny,iPlot+1)
            C=GeneDist.ClassDistMachine()
            x,y=C.giveCumulDist(x,Ns=1000,Norm=True)#,xmm=[-5,5])
            ax.plot(x,y,color="black")
            ax.plot(x,ClassShapiroWilk.Phi(x),ls=":",color="black")
            
        pylab.draw()
        pylab.show(block=False)
        pylab.pause(0.1)
