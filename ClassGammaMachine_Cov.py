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
    ny=int(nx*ratio)
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
        #self.TypeSqrtC=np.float32
        #self.TypeCube=np.float32

        i0=int(lg.min()/self.CellRad+N0)
        i1=i0+NPix+1
        j0,j1=mg.min()/self.CellRad+N0,mg.max()/self.CellRad+N0
        self.ThisMask=np.zeros((NPix,NPix),np.float32)
        for i in range(NPix):
            for j in range(NPix):
                ii,jj=self.CM.RaDecToMaskPix(self.rag[i,j],self.decg[i,j])
                self.ThisMask[i,j]=self.CM.MaskArray[0,0,int(ii),int(jj)]
                

        
        
        self.GammaCube=None
        self.NSlice=self.zg.size-1

        self.L_NParms=[]
        self.HasReferenceCube=False
                             
        self.DicoCovSVD = shared_dict.create("DicoCovSVD")
        
        APP.registerJobHandlers(self)

        
    def initCovMatrices(self,
                        ScaleFWHMkpc=100.):
        
        FileOut="Cov_%s_%ikpc.DicoCov"%(self.Mode,ScaleFWHMkpc)
        
        Parms={"CellDeg":self.CellDeg,
               "NPix":self.NPix,
               "CellRad":self.CellRad,
               "zg":self.zg.tolist()}
        
        if os.path.isfile(FileOut):
            log.print("Loading %s..."%FileOut)
            DicoSave=MyPickle.Load(FileOut)
            if DicoSave["Parms"]==Parms:
                log.print("   all parameters are the same... ")
                self.MaxValueSigmoid=DicoSave["MaxValueSigmoid"]
                self.a_Sigmoid=DicoSave["a_Sigmoid"]
                self.ScaleCov=DicoSave["ScaleCov"]
                self.L_SqrtCov=DicoSave["L_SqrtCov"]
                self.L_NParms=DicoSave["L_NParms"]
                self.L_Hinv=DicoSave["L_Hinv"]
                self.L_ssqs=DicoSave["L_ssqs"]
                self.NParms=np.sum(self.L_NParms)
                log.print("Number of free parameters %i"%self.NParms)
                log.print("  with ScaleCov=%s"%self.ScaleCov)
                
                return
            
        f=2.*np.sqrt(2.*np.log(2.))
        Sig_kpc=ScaleFWHMkpc/f
        self.L_SqrtCov=[]
        self.L_Hinv=[]
        self.L_ssqs=[]
        for iz in range(self.NSlice):
            APP.runJob("_computeSVD:%i"%(iz),
                       self._computeSVD,
                       args=(iz,Sig_kpc))#,serial=True)

        APP.awaitJobResults("_computeSVD:*", progress="Compute SqrtCov")
        self.DicoCovSVD.reload()
        for iz in range(self.NSlice):
            sqrtCs=(self.DicoCovSVD["sqrtCs_%i"%iz].copy())
            self.L_Hinv.append(self.DicoCovSVD["Hinv_%i"%iz].copy())
            self.L_ssqs.append(self.DicoCovSVD["ssqs_%i"%iz].copy())
            self.L_SqrtCov.append(sqrtCs)
            self.L_NParms.append(sqrtCs.shape[1])
        self.NParms=np.sum(self.L_NParms)
        log.print("Number of free parameters %i"%self.NParms)
        
        DicoSave={"Parms":Parms,
                  "ScaleCov":self.DicoCovSVD["ScaleCov_%i"%iz],
                  "a_Sigmoid":self.DicoCovSVD["a_Sigmoid_%i"%iz],
                  "MaxValueSigmoid":self.DicoCovSVD["MaxValueSigmoid_%i"%iz],
                  "L_SqrtCov":self.L_SqrtCov,
                  "L_ssqs":self.L_ssqs,
                  "L_NParms":self.L_NParms,
                  "L_Hinv":self.L_Hinv}
        self.ScaleCov=DicoSave["ScaleCov"]
        self.a_Sigmoid=DicoSave["a_Sigmoid"]
        self.MaxValueSigmoid=DicoSave["MaxValueSigmoid"]
        log.print("Saving sqrt(Cov) in %s"%FileOut)
        MyPickle.Save(DicoSave,FileOut)
        
    def toSparse(self,C,Th=1e-2):
        Ca=np.abs(C)
        indi,indj=np.where(Ca>Th*Ca.max())
        data=C[indi,indj]
        M=C.shape[0]
        Cs=scipy.sparse.coo_matrix((data, (indi, indj)), shape=(M, M))
        S=Cs.size/M**2
        return Cs,S

    def _computeSVD(self,iz,Sig_kpc):
        z0,z1=self.zg[iz],self.zg[iz+1]
        zm=(z0+z1)/2.
        SigmaRad=Sig_kpc*cosmo.arcsec_per_kpc_comoving(zm).to_value()/(3600.)*np.pi/180
        self.Scale_Cov_X=None
        if self.Mode=="Gauss":
            CCM=ClassCovMatrix_Gauss.ClassCovMatrix()
            C=CCM.giveCovMat(CellSizeRad=self.CellRad,
                             NPix=self.NPix,
                             zm=zm,
                             SigmaPix=SigmaRad/self.CellRad)
            
        elif self.Mode=="Sim3D":
            CCM=ClassCovMatrix_Sim3D_2.ClassCovMatrix()
            C=CCM.giveCovMat(CellSizeRad=self.CellRad,
                             NPix=self.NPix,
                             zm=zm)
            
            #self.Scale_Cov_X="log"C.Scale_Cov_X

        #print(":!::")
        #C=np.eye(self.NPix**2,self.NPix**2)
        Cs,Sparsity=self.toSparse(C)
        M=C.shape[0]
        log.print("  Non-zeros are %.1f%% of the matrix size [%ix%i]"%(Sparsity*100,M,M))
        self.Cs=Cs
        CellSize=1.
        A=(CellSize*M)**2
        # k=np.max([A/Sig**2,1.])
        # k=int(k*3)
        k=M-1#int(np.min([M-1,k]))
        log.print("Choosing k=%i [M=%i]"%(k,M))
        self.k=k
        #print(":!::")
        #Us,ss,Vs=scipy.sparse.linalg.svds(Cs,k=k)
        Us,ss,Vs=np.linalg.svd(C)

        sss=ss[ss>0]
        log.print("  log Singular value Max/Min: %5.2f"%(np.log10(sss.max()/sss.min())))
        ssqs=np.sqrt(ss)
        
        ind=np.where(ssqs>0)[0]
        ind=np.where(ssqs>1e-2*ssqs.max())[0]
        S0=Us.shape

        Us=Us[:,ind]
        ssqs=ssqs.flat[ind]
        A= sqrtCs =Us*ssqs.reshape(1,ssqs.size)
        # print(S0,Us.shape,ssqs.shape)
        
        Hinv=ModLinAlg.invSVD((A.T).dot(A))

        self.DicoCovSVD["sqrtCs_%i"%iz]=A
        self.DicoCovSVD["ssqs_%i"%iz]=ssqs
        self.DicoCovSVD["Hinv_%i"%iz]=Hinv
        self.DicoCovSVD["MaxValueSigmoid_%i"%iz]=CCM.MaxValueSigmoid
        self.DicoCovSVD["ScaleCov_%i"%iz]=CCM.ScaleCov
        self.DicoCovSVD["a_Sigmoid_%i"%iz]=CCM.a_Sigmoid
            
    def CubeToVec(self,Cube):
        ii=0
        Cube=self.TypeCube(Cube.copy())
        XOut=np.zeros((self.NParms,),np.float32)
        for iSlice in range(self.NSlice):
            N=self.L_NParms[iSlice]
            if N==0: continue
            Slice=Cube[iSlice].copy()
            Slice[Slice==0]=1e-6
            #y=Slice.reshape((-1,1))-1.
            if self.ScaleCov=="log":
                y=np.log(Slice.reshape((-1,1)))
            elif self.ScaleCov=="Sigmoid":
                y=ClassCovMatrix_Sim3D_2.Logit(x,a_Sigmoid=self.a_Sigmoid,MaxVal=self.MaxValueSigmoid)
            A=self.L_SqrtCov[iSlice]
            x0=(A.T).dot(y)
            Hinv=self.L_Hinv[iSlice]#ModLinAlg.invSVD((A.T).dot(A))
            x=np.dot(Hinv,x0)
            # print(np.mean(np.abs(x.imag)) / np.mean(np.abs(x.real)))
            XOut[ii:ii+N]=(x.real.ravel())[:]
            ii+=N
        return XOut

    def PlotGammaCube(self,X=None,Cube=None,FigName="Gamma Cube",OutName=None,vmm=None):
        # return
        if X is not None:
            self.computeGammaCube(X)
        if Cube is None:
            Cube=self.GammaCube


            
        import pylab

        fact=1.8
        figsize=(13/fact,8/fact)
        fig=pylab.figure(FigName,figsize=figsize)
        self.CurrentFig=fig
        self.AxList=[]
        Nx,Ny=GiveNXNYPanels(self.NSlice,ratio=figsize[0]/figsize[1])
        pylab.clf()

        for iPlot in range(self.NSlice):
            S=Cube[iPlot]
            ax=pylab.subplot(Nx,Ny,iPlot+1)
            self.AxList.append(ax)
            if np.count_nonzero(np.isnan(S))>0: stop
            if vmm is not None:
                vmin,vmax=vmm
            else:
                vmin,vmax=S.min(),S.max()
            pylab.imshow(S,interpolation="nearest",vmin=vmin,vmax=vmax)
            pylab.title("[%.2f - %.2f]"%(S.min(),S.max()))
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

    def computeGammaCube(self,X):

        if self.CurrentX is not None:
            if np.allclose(self.CurrentX,X):
                #log.print("Current cube ok, skipping computation...")
                return
        self.CurrentX=X.copy()
        
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
                GammaCube[iz,:,:]=np.exp(y)
            elif self.ScaleCov=="Sigmoid":
                GammaCube[iz,:,:]=ClassCovMatrix_Sim3D_2.Sigmoid(y,a=self.a_Sigmoid,MaxVal=self.MaxValueSigmoid)
                stop
        T.timeit("Slices")
        self.GammaCube=GammaCube

    def giveGammaCube(self,X):
        self.computeGammaCube(X)
        return self.GammaCube
    
