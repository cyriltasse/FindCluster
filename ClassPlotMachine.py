import numpy as np
import matplotlib.pyplot as plt
import GeneDist
import ClassSimulCatalog
import matplotlib.pyplot as pylab
import ClassMassFunction
from astropy.cosmology import WMAP9 as cosmo
from DDFacet.Other import logger
log = logger.getLogger("PlotMachine")

from DDFacet.Other import ModColor
from DDFacet.Other import ClassTimeIt
from DDFacet.Other.progressbar import ProgressBar
from DDFacet.Array import shared_dict
import ClassCatalogMachine
#import ClassDiffLikelihoodMachine as ClassLikelihoodMachine
import ClassLogDiffLikelihoodMachine as ClassLikelihoodMachine
import ClassInitGammaCube
import ClassDisplayRGB

import scipy.optimize
import GeneDist


class ClassPlotMachine():
    def __init__(self,
                 CLM,
                 XSimul=None):
        self.CLM=CLM
        self.GM=CLM.MassFunction.GammaMachine
        self.L_L=[]
        if XSimul is not None:
            self.LTrue=self.CLM.L(XSimul)
            log.print("Set XSimul with L=%f"%self.LTrue)
            self.XSimul=XSimul
            self.CubeSimul=self.GM.giveGammaCube(self.XSimul)

            
    def PlotHist(self,g):
        GM=self.GM
        L_NParms=GM.L_NParms
        dJdg=self.CLM.dJdg(g).flat[:]
        L=self.CLM.L(g)
        self.L_L.append(L)
        LTrue=self.LTrue
        L_L=self.L_L
        
        figH=pylab.figure("hist")
        figH.clf()
        ii=0
        pylab.subplot(2,2,1)
        for iSlice in range(self.CLM.NSlice):
            ThisNParms=L_NParms[iSlice]
            iPar=ii
            jPar=iPar+ThisNParms
            ii+=ThisNParms
            C=GeneDist.ClassDistMachine()
            x,y=C.giveCumulDist(g[iPar:jPar],Ns=100,Norm=True)#,xmm=[-5,5])
            pylab.plot(x,y)

        pylab.subplot(2,2,2)
        pylab.plot((L_L),color="black")
        pylab.plot(([LTrue]*len(L_L)),ls="--",color="black")
        Sig=np.sqrt(1./np.abs(dJdg))
        
        #Sig[Sig>1.]=1
        # Sig=(1./np.abs(dJdg))
        NTry=200
        
        GammaStat=np.zeros((NTry,self.GM.NSlice,self.GM.NPix,self.GM.NPix),np.float32)
        for iTry in range(NTry):
            GammaStat[iTry]=(self.GM.giveGammaCube(g+Sig*np.random.randn(*g.shape)))

        Cube_mean=np.mean(GammaStat,axis=0)
        CubeSimul=self.CubeSimul

        Scale="log"
        if Scale=="log":
            GammaStat=np.log10(GammaStat)
            CubeSimul=np.log10(CubeSimul)
        qList=[[0.15e-2,0.9985],
               [2.5e-2,0.975],
               [0.16,0.84]]
        ys=CubeSimul.flatten()
        def FillBetween(G,q0,q1):
            C=GeneDist.ClassDistMachine()
            Cube_q0=np.quantile(G,q0,axis=0)
            Cube_q1=np.quantile(G,q1,axis=0)
            v0=Cube_q0.flatten()-ys
            v1=Cube_q1.flatten()-ys
            x0,y0=C.giveCumulDist(v0,Ns=1000,Norm=True)
            x1,y1=C.giveCumulDist(v1,Ns=1000,Norm=True)
            x=np.concatenate([x0,x1[::-1]])
            y=np.concatenate([y0,y1[::-1]])
            pylab.fill(x,y,color="black",alpha=0.2)
            pylab.plot(x,y,color="black",alpha=0.2)
        
        ax=pylab.subplot(2,2,3)
        for (q0,q1) in qList:
            FillBetween(GammaStat,q0,q1)
            
        Cube_q50=np.quantile(GammaStat,0.5,axis=0)
        v0=Cube_q50.flatten()-ys
        x0,y0=C.giveCumulDist(v0,Ns=1000,Norm=True)
        pylab.plot(x0,y0,ls="--",color="black")

        Cube_mean=np.mean(GammaStat,axis=0)
        v0=Cube_mean.flatten()-ys
        x0,y0=C.giveCumulDist(v0,Ns=1000,Norm=True)
        pylab.plot(x0,y0,ls=":",color="red")
        
        pylab.xlim(-5,5)
        pylab.grid()
        pylab.draw()
        pylab.show(block=False)
        pylab.pause(0.1)

        # self.GM.PlotGammaCube(Cube=(MeanCube-self.CubeSimul)/eCube,FigName="eCube",vmm=(-3,3))
        
        self.GM.PlotGammaCube(Cube=(Cube_mean),FigName="log(MeanCube)")
        # self.GM.PlotCumulDistX(g)
            
        # figH.savefig("Hist%5.5i.png"%iStep)
        # #self.GM.PlotGammaCube(Cube=y0Cube,FigName="Cube0")
        # #self.GM.PlotGammaCube(Cube=y1Cube,FigName="Cube1")
        # #self.GM.PlotGammaCube(Cube=ycCube,FigName="CubeC")
        # eCube=ycCube/((y1Cube-y0Cube)/2.)
        # #self.GM.PlotGammaCube(Cube=eCube,FigName="eCube")
        # self.GM.PlotGammaCube(Cube=(MeanCube-self.CubeSimul)/eCube,FigName="eCube",vmm=(-3,3))
        # self.GM.PlotGammaCube(Cube=np.log(MeanCube),FigName="log(MeanCube)")
        # self.GM.PlotCumulDistX(g)
