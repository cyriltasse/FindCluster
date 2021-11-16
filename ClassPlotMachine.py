import numpy as np
import matplotlib.pyplot as plt
import GeneDist
import ClassSimulCatalog
import matplotlib.pyplot as pylab
import ClassMassFunction
from astropy.cosmology import WMAP9 as cosmo
from DDFacet.Other import logger
log = logger.getLogger("PlotMachine")
import os

from DDFacet.Other import ModColor
from DDFacet.Other import ClassTimeIt
from DDFacet.Other.progressbar import ProgressBar
from DDFacet.Array import shared_dict
import ClassCatalogMachine
#import ClassDiffLikelihoodMachine as ClassLikelihoodMachine
import ClassLogDiffLikelihoodMachine as ClassLikelihoodMachine
import ClassInitGammaCube
import ClassDisplayRGB
import copy
import scipy.optimize
import GeneDist
def GiveNXNYPanels(Ns,ratio=800/500):
    nx=int(round(np.sqrt(Ns/ratio)))
    #ny=int(nx*ratio)
    ny=Ns//nx
    if nx*ny<Ns: ny+=1
    return nx,ny
import psutil
import time
from mpl_toolkits.axes_grid1 import make_axes_locatable

import scipy.special
erf=scipy.special.erf
G=scipy.special.gamma
logG=scipy.special.loggamma
def Phi(x,mu=0.,sig=1.):
    return 0.5*(1+erf((x-mu)/(sig*np.sqrt(2))))
def Sigmoid(x,a=1):
    return 1./(1+np.exp(-x/a))
mdot=np.linalg.multi_dot

import ClassShapiroWilk

def FillBetween(ax,Gin,q0,q1,CubeSimul,iSlice=None,PlotMed=False):
    if iSlice is not None:
        s=slice(iSlice,iSlice+1)
    else:
        s=slice(None)
    G=Gin[:,s,:,:]
    
    C=GeneDist.ClassDistMachine()
    Cube_q0=np.quantile(G,q0,axis=0)
    Cube_q1=np.quantile(G,q1,axis=0)
    
    
    v0=Cube_q0.flatten()-CubeSimul[s,:,:].flatten()
    v1=Cube_q1.flatten()-CubeSimul[s,:,:].flatten()
    v0=v0[v0!=0]
    v1=v1[v1!=0]
    x0,y0=C.giveCumulDist(v0,Ns=1000,Norm=True)
    x1,y1=C.giveCumulDist(v1,Ns=1000,Norm=True)
    x=np.concatenate([x0,x1[::-1]])
    y=np.concatenate([y0,y1[::-1]])
    ax.fill(x,y,color="black",alpha=0.2)
    ax.plot(x,y,color="black",alpha=0.2)
    if PlotMed:
        v0=np.quantile(G,0.5,axis=0).flatten()-CubeSimul[s,:,:].flatten()
        v0=v0[v0!=0]
        x0,y0=C.giveCumulDist(v0,Ns=1000,Norm=True)
        ax.plot(x0,y0,ls="--",color="black")


class ClassPlotMachine():
    def __init__(self,
                 CLM,
                 XSimul=None,
                 StepPlot=100,
                 DicoSourceXY=None,
                 PlotID=None):
        self.PlotID=PlotID
        self.DicoSourceXY=DicoSourceXY
        self.CLM=CLM
        self.GM=CLM.MassFunction.GammaMachine
        self.L_L=[]
        self.ScaleCube="log"
        #self.ScaleCube="linear"
        self.iFig=0
        self.iCall=0
        self.StepPlot=StepPlot
        os.system("rm *.png 2>/dev/null")
        self.LTrue=None
        self.XSimul=None
        if XSimul is not None:
            self.LTrue=self.CLM.logP(XSimul)
            log.print("Set XSimul with L=%f"%self.LTrue)
            self.XSimul=XSimul
            self.CubeSimul=self.GM.giveGammaCube(self.XSimul,ScaleCube=self.ScaleCube)
            for iSlice in range(self.GM.NSlice):
                self.CubeSimul[iSlice][self.GM.ThisMask==1]=0.
                
            # P=self.CubeSimul.copy()
            # P.fill(np.nan)
            # self.GM.PlotGammaCube(Cube=P,FigName="Points",
            #                       DicoSourceXY=DicoSourceXY)
            # self.SaveFig()
            self.GM.PlotGammaCube(Cube=self.CubeSimul,FigName="Simul LogCube",
                                  DicoSourceXY=DicoSourceXY)
            self.SaveFig()
        
    def Plot(self,g,NTry=100,Force=False,FullHessian=False,gArray=None):
        if (self.iCall%self.StepPlot==0) or Force:
            log.print("Call %i... plotting"%self.iCall)
            self.PlotL(g)
            self.giveGammaStat(g,
                               NTry=NTry,
                               gArray=gArray,
                               FullHessian=FullHessian)
            self.PlotHist(g,
                          NTry=NTry,
                          FullHessian=FullHessian,
                          gArray=gArray)
            self.PlotBest(g)
            self.PlotFigPubli(g)
            # self.PlotHistEigen(g)
            self.PlotLogDiff(g)
            self.iFig+=1
        self.iCall+=1

    def PlotL(self,g):
        if len(self.L_L)<=2:
            return
        fig=pylab.figure("Likelihood")
        fig.clf()
        ax=pylab.subplot(1,1,1)
        ax.plot(np.array(self.L_L),color="black")
        ax.plot((np.array([self.LTrue]*len(self.L_L))),ls="--",color="black")
        ax.set_xlabel("Iteration")
        ax.set_ylabel(r"$-\log(\mathcal{P}\{\gamma|D\})$")
        self.SaveFig()
        
        
    def PlotBest(self,g):
        CubeBest=self.GM.giveGammaCube(g,ScaleCube=self.ScaleCube)
        for iSlice in range(self.GM.NSlice):
            CubeBest[iSlice][self.GM.ThisMask==1]=np.nan

        DicoSourceXY={}
        DicoSourceXY["X"]=self.CLM.Cat_s.yCube
        DicoSourceXY["Y"]=self.CLM.Cat_s.xCube
        LP=self.CLM.Cat_s.n_zt
        LP=np.array(LP)/np.sum(np.array(LP),axis=1).reshape((-1,1))
        DicoSourceXY["P"]=LP

        self.GM.PlotGammaCube(Cube=CubeBest,FigName="Best LogCube",DicoSourceXY=DicoSourceXY)
        # for iSlice in range(self.GM.NSlice):
        #     ax0=self.GM.AxList[iSlice]
        #     s=DicoSourceXY["P"][:,iSlice]
        #     rgba_colors = np.zeros((s.size,4))
        #     rgba_colors[:,1:3] = 0
        #     rgba_colors[:, 3] = s
        #     Ns=self.CLM.Cat_s.xCube.size
        #     dx=0#np.random.rand(Ns)-0.5
        #     dy=0#np.random.rand(Ns)-0.5
        #     ax0.scatter(DicoSourceXY["X"]+dx,DicoSourceXY["Y"]+dy,s=3, color=rgba_colors)#,c="black",2*s[:,iSlice])
        # pylab.draw()
        # pylab.show(block=False)
        # pylab.pause(0.1)
        self.SaveFig()
        
            
    def PlotFigPubli(self,g):
        if self.XSimul is None: return
        CubeSimul=self.GM.giveGammaCube(self.XSimul,ScaleCube=self.ScaleCube)
        CubeBest=self.GM.giveGammaCube(g,ScaleCube=self.ScaleCube)
        for iSlice in range(self.GM.NSlice):
            CubeSimul[iSlice][self.GM.ThisMask==1]=np.nan
            CubeBest[iSlice][self.GM.ThisMask==1]=np.nan


        LSlice=(np.arange(15)[0::3]).tolist()#[0,4,9,14]
        NNSlice=len(LSlice)
             
        Nx,Ny=3,NNSlice
        W=3
        fig=pylab.figure("Gamma Slice",figsize=(W*NNSlice,W*Nx))
        fig.clf()
        left=0.1
        bottom=0.1
        fig.subplots_adjust(hspace=0,
                            wspace=0,
                            left = left,
                            right = 0.95,
                             bottom = bottom,
                            top = 0.95)

        vmin=1e6
        vmax=-1e6
        LVMM=[]
        for iiSlice,iSlice in enumerate(LSlice):
            Ss=CubeSimul[iSlice]
            Sb=CubeBest[iSlice]
            ind=np.where(self.GM.ThisMask==0)
            v0=np.min([vmin,Ss[ind].min(),Sb[ind].min()])
            v1=np.max([vmax,Ss[ind].max(),Sb[ind].max()])
            LVMM.append((v0,v1))
            
        # print(vmin,vmax)
        
        for iiSlice,iSlice in enumerate(LSlice):
            ax0=fig.add_subplot(Nx,Ny,1+iiSlice)
            DoX=0
            DoY=(iiSlice==0)
            self.PlotFigPubliSliceType(g,iSlice,ax0,CubeSimul,CubeBest,vmm=LVMM[iiSlice],Type="GammaSimul",Axes=[DoX,DoY])
            
        for iiSlice,iSlice in enumerate(LSlice):
            DoY=(iiSlice==0)
            DoX=0
            ax0=fig.add_subplot(Nx,Ny,NNSlice+1+iiSlice)
            self.PlotFigPubliSliceType(g,iSlice,ax0,CubeSimul,CubeBest,vmm=LVMM[iiSlice],Type="PointSimul",Axes=[DoX,DoY])

        for iiSlice,iSlice in enumerate(LSlice):
            DoY=(iiSlice==0)
            DoX=1
            ax0=fig.add_subplot(Nx,Ny,2*NNSlice+1+iiSlice)
            self.PlotFigPubliSliceType(g,iSlice,ax0,CubeSimul,CubeBest,vmm=LVMM[iiSlice],Type="GammaBest",Axes=[DoX,DoY])

        pylab.draw()
        pylab.show(block=False)
        pylab.pause(0.01)
        self.SaveFig(fig)

            

        bottom=0.17
        top=0.95
        fig=pylab.figure("Error Slice",figsize=(W*NNSlice,1*W))
        fig.clf()
        fig.subplots_adjust(hspace=0,
                            wspace=0,
                            left = left,
                            right = 0.95,
                            bottom = bottom,
                            top = top)

        
        GammaStat=self.GammaStat
        NTry=GammaStat.shape[0]
        for iSlice in range(self.GM.NSlice):
            CubeSimul[iSlice][self.GM.ThisMask==1]=0.
            for iTry in range(NTry):
                GammaStat[iTry,iSlice][self.GM.ThisMask==1]=0.
        
        cmap=copy.copy(pylab.cm.cividis)
        cmap.set_bad('gray',1.)

        mm0,mm1=1e6,-1e6
        for iiSlice,iSlice in enumerate(LSlice):
            ind=np.where(self.GM.ThisMask.flatten()==0)[0]
            y=CubeBest[iSlice].ravel()
            ys=CubeSimul[iSlice].ravel()
            ey=self.SigmaCube[iSlice].ravel()
            y=y[ind]
            ys=ys[ind]
            ey=ey[ind]
            mm0=np.min([mm0,y.min(),ys.min()])
            mm1=np.max([mm1,y.max(),ys.max()])
        
        for iiSlice,iSlice in enumerate(LSlice):
            ind=np.where(self.GM.ThisMask.flatten()==0)[0]
            y=CubeBest[iSlice].ravel()
            ys=CubeSimul[iSlice].ravel()
            ey=self.SigmaCube[iSlice].ravel()
            y=y[ind]
            ys=ys[ind]
            ey=ey[ind]
        
            DoY=(iiSlice==0)
            DoX=1
            ax0=fig.add_subplot(1,NNSlice,1+iiSlice)

        
            ax0.hexbin(y.flatten(),ys.flatten(), gridsize=50, cmap=cmap,mincnt=1)#,extent=(x.min()-0.1,x.max()+0.1,-5,5))
            
                
            # ax0.errorbar(ys,y,ey, label='first', marker='o', linestyle="",alpha=0.1,color="black")
            
            ax0.plot([LVMM[iiSlice][0],LVMM[iiSlice][1]],[LVMM[iiSlice][0],LVMM[iiSlice][1]], linestyle="--",color="black")
            ax0.grid(color='black', linestyle=':')
            ax0.set_xlim(mm0,mm1)
            ax0.set_ylim(mm0,mm1)

            
            # ax0.set_xlabel(r"$\log{\left(\gamma\right)}$")
            # ax0.set_ylabel(r"$\log{\left(\widehat{\gamma}\right)}$")

            ax0.set_xlabel("$ \log ( \gamma )$")
            #secax = ax0.secondary_xaxis('top')#, functions=(deg2rad, rad2deg))
            #secax.set_xlabel(r"$ \log ( \gamma )$")
            if DoY:
                ax0.set_ylabel(r"$\log( \widehat{\gamma} )$")
            else:
                ax0.set_yticklabels([])


        pylab.draw()
        #mng = pylab.get_current_fig_manager()
        #mng.resize(*mng.window.maxsize())
        pylab.show(block=False)
        pylab.pause(0.01)
        self.SaveFig(fig)



        fig=pylab.figure("Dist Error Slice",figsize=(W*NNSlice,1*W))
        fig.clf()
        fig.subplots_adjust(hspace=0,
                            wspace=0,
                            left = left,
                            right = 0.95,
                            bottom = bottom,
                            top = top)
                
                
        for iiSlice,iSlice in enumerate(LSlice):
            ind=np.where(self.GM.ThisMask.flatten()==0)[0]
            y=CubeBest[iSlice].ravel()
            ys=CubeSimul[iSlice].ravel()
            ey=self.SigmaCube[iSlice].ravel()
            y=y[ind]
            ys=ys[ind]
            ey=ey[ind]
        
            DoY=(iiSlice==0)
            DoX=1
            ax0=fig.add_subplot(1,NNSlice,1+iiSlice)

            # C=GeneDist.ClassDistMachine()
            # x0,y0=C.giveCumulDist((y-ys)/ey,Ns=1000,Norm=True)
            # # ax0.hexbin(y.flatten(),ys.flatten(), gridsize=50, cmap=cmap,mincnt=1)#,extent=(x.min()-0.1,x.max()+0.1,-5,5))
            # # ax0.errorbar(ys,y,ey, label='first', marker='o', linestyle="",alpha=0.1,color="black")
            # for NTry in range(100):
            #     xx=np.random.randn(y.size)
            #     x1,y1=C.giveCumulDist(xx,Ns=1000,Norm=True)
            #     ax0.plot(x1,y1, linestyle="-",color="gray",alpha=0.1)
            # ax0.plot(x0,y0, linestyle="-",color="black")



            ax = ax0

            FillBetween(ax,GammaStat,0.15e-2,0.9985,CubeSimul,iSlice=iSlice)
            FillBetween(ax,GammaStat,2.5e-2,0.975,CubeSimul,iSlice=iSlice)
            FillBetween(ax,GammaStat,0.16,0.84,CubeSimul,iSlice=iSlice,PlotMed=True)
            y=np.sort([0,0.15e-2,0.9985,2.5e-2,0.975,0.16,0.84,1])
            x=np.zeros_like(y)
            ax.plot(x,y,ls=":", linewidth=2,color="black")
            ax.scatter(x,y,color="red",s=5,edgecolors="red")
            ax.grid(color='black', linestyle=':')#, linewidth=2)
            ax.set_ylim(0,1)
            ax.set_xlim(-2,2)


            
            
            # ax0.set_xlabel(r"$\log{\left(\gamma\right)}$")
            # ax0.set_ylabel(r"$\log{\left(\widehat{\gamma}\right)}$")

            ax0.set_xlabel(r"$\log( \widehat{\gamma} )-\log ( \gamma )$")
            if DoY:
                ax0.set_ylabel("Cumulative distribution")
            else:
                ax0.set_yticklabels([])

                
            # p0 = ax0.get_position().get_points().flatten()
            # dx=p0[1]-p0[0]
            # dy=p0[3]-p0[2]
            # W=0.4
            # ax = fig.add_axes([p0[0], 0.93, p0[2]-p0[0], 0.95])


        pylab.draw()
        #mng = pylab.get_current_fig_manager()
        #mng.resize(*mng.window.maxsize())
        pylab.show(block=False)
        pylab.pause(0.01)
        self.SaveFig(fig)


            
    def PlotFigPubliSliceType(self,g,iSlice,ax0,CubeSimul,CubeBest,vmm=None,Type="GammaSimul",Axes=[True,True]):

        rac_deg,decc_deg,CellDeg,NPix=self.CLM.rac_deg,self.CLM.decc_deg,self.CLM.CellDeg,self.CLM.NPix
        Dx=CellDeg*NPix/2*60
        cmap=copy.copy(pylab.cm.cividis)
        cmap.set_bad('gray',1.)

        vmin,vmax=vmm
        
        
        if Type=="GammaSimul":
            im=ax0.imshow(CubeSimul[iSlice],
                          interpolation="nearest",
                          aspect="auto",
                          extent=(-Dx,Dx,-Dx,Dx),
                          #alpha=0.5,
                          vmin=vmin,
                          vmax=vmax,
                          origin="lower",
                          cmap=cmap)
            ax0.grid(color='black', linestyle=':')#, linewidth=2)
            ax0.set_title("%.2f < z < %.2f"%(self.GM.zg[iSlice],self.GM.zg[iSlice+1]))
            
            # add color bar below chart
            ax=ax0
            # divider = make_axes_locatable(ax)
            # cax = divider.new_vertical(size = '5%', pad = 0.5)
            # fig.add_axes(cax)
            # fig.colorbar(im, cax = cax, orientation = 'horizontal')
            
            # p0 = ax0.get_position().get_points().flatten()
            # fig=pylab.gcf()
            # ax_cbar1 = fig.add_axes([p0[0], 0.93, p0[2]-p0[0], 0.95])
            # pylab.colorbar(im, cax=ax_cbar1, cmap=cmap, orientation='horizontal')
            # ax_cbar1.set_title("%.2f < z < %.2f"%(self.GM.zg[iSlice],self.GM.zg[iSlice+1]))
            # #ax_cbar1.xaxis.set_ticks_position('top')

        elif Type=="PointSimul":
            

            s=self.DicoSourceXY["P"][:,iSlice]
            rgba_colors = np.zeros((s.size,4))
            rgba_colors[:,1:3] = 0
            rgba_colors[:, 3] = s
            Ns=self.CLM.Cat_s.xCube.size
            
            X=np.linspace(-Dx,Dx,NPix)
            Y=X.copy()
            X, Y = np.meshgrid(X, Y)
            Z = self.GM.ThisMask.copy()
            #Z[Z==0]=np.nan
            Z[Z==1]=0.5

            cmap1=copy.copy(pylab.cm.binary)
            cmap1.set_bad('gray',1.)
            ax0.imshow(Z,
                       interpolation="nearest",
                       aspect="auto",
                       extent=(-Dx,Dx,-Dx,Dx),
                       #alpha=0.5,
                       vmin=0,
                       vmax=1,
                       origin="lower",
                       cmap=cmap1)
            # ax0.contour(X,Y,Z, [0.5], cmap=cmap)
            ax0.scatter((self.DicoSourceXY["X"]-NPix/2)*CellDeg*60,(self.DicoSourceXY["Y"]-NPix/2)*CellDeg*60,s=3, color=rgba_colors)#,c="black",2*s[:,iSlice])
        
        
            ax0.grid(color='black', linestyle=':')#, linewidth=2)
            #ax0.axis('off')
            ax0.set_xlim(-Dx,Dx)#0,self.GM.NPix)
            ax0.set_ylim(-Dx,Dx)#0,self.GM.NPix)

        elif Type=="GammaBest":
            ax0.imshow(CubeBest[iSlice],
                       interpolation="nearest",
                       aspect="auto",
                       extent=(-Dx,Dx,-Dx,Dx),
                       vmin=vmin,
                       vmax=vmax,
                       origin="lower",
                       cmap=cmap)
            ax0.grid(color='black', linestyle=':')#, linewidth=2)
            
            
        if Axes[0]:
            ax0.set_xlabel(r"$\Delta_x$ [arcmin]")
        else:
            ax0.set_xticklabels([])
            
        if Axes[1]:
            ax0.set_ylabel(r"$\Delta_y$ [arcmin]")
        else:
            ax0.set_yticklabels([])


        
        
    def PlotLogDiff(self,g):
        if self.XSimul is None: return
        CubeBest=self.GM.giveGammaCube(g,ScaleCube=self.ScaleCube)
        fig=pylab.figure("logDiff")
        cmap=pylab.cm.cubehelix
        cmap=pylab.cm.plasma
        cmap=pylab.cm.cividis

        Lc=cmap(np.linspace(0.1,0.9,self.GM.NSlice))
        ax=pylab.subplot(1,1,1)
        ax.cla()
        # for iSlice in range(self.GM.NSlice):
        #     n=self.CLM.CM.DicoDATA["DicoSelFunc"]["n_z"][iSlice]
        #     y=CubeBest[iSlice].flatten()*n
        #     ys=self.CubeSimul[iSlice].flatten()*n
        #     ax.scatter(np.log10(ys),np.log10(y/ys),
        #                   s=5,
        #                   color=Lc[iSlice],
        #                   alpha=0.1,
        #                   linewidth=0)

        n=self.CLM.CM.DicoDATA["DicoSelFunc"]["n_z"].reshape((-1,1,1))
        y=(CubeBest)
        ys=(self.CubeSimul)
        ey=self.SigmaCube
        #x,y=np.log10(ys),np.log10(y/ys)
        x=ys
        #x=(x+np.log(n)).flatten()
        x=(x).flatten()
        #ax.hexbin(x,((y-ys)).flatten(), gridsize=50, cmap='inferno',extent=(x.min()-0.1,x.max()+0.1,-2,2))
        cm='cubehelix'
        cm="plasma"
        cm="cividis"
        ax.hexbin(x,((y-ys)/ey).flatten(), gridsize=50, cmap=cm,extent=(x.min()-0.1,x.max()+0.1,-5,5),mincnt=1)
        pylab.xlabel("$ \log ( \gamma )$")
        pylab.ylabel("$(\log( \widehat{\gamma} ) - \log ( \gamma ))/\sigma_{\widehat{\gamma}}$")
        # pylab.xlim(-1.5,1.5)
        # pylab.ylim(-2,2)
        pylab.draw()
        pylab.show(block=False)
        pylab.pause(0.1)
        self.SaveFig()

        


        
    def giveGammaStat(self,g,NTry=100,gArray=None,FullHessian=False):
        
        if gArray is not None:
            NTry=gArray.shape[0]
        elif not FullHessian:
            log.print("  using diagonal Hessian...")
            dJdg=self.CLM.d2logPdg2(g,Diag=True).flat[:]
            Sig=np.sqrt(1./np.abs(dJdg))#/2.
            gArray=np.array([g+Sig*np.random.randn(*g.shape) for iTry in range(NTry)])
        elif FullHessian:
            while True:
                m=psutil.virtual_memory()
                if m.percent<50.: break
                #print(m.percent)
                time.sleep(1)
            log.print("  building full Hessian...")
            dJdG=self.CLM.d2logPdg2(g,Diag=False)
            log.print("  .T")
            dJdG=(dJdG+dJdG.T)/2.
            # idJdG=ModLinAlg.invSVD(dJdg)
            while True:
                m=psutil.virtual_memory()
                if m.percent<50.: break
                #print(m.percent)
                time.sleep(3)
            log.print("  doing SVD...")
            #dJdG=np.diag(np.diag(dJdG))
            Us,ss,Vs=np.linalg.svd(dJdG)

            sss=ss[ss>0]
            log.print("  log Singular value Max/Min: %5.2f"%(np.log10(sss.max()/sss.min())))

            Th=1e-6
            ind=np.where(ss<ss.max()*Th)[0]
            ssqs=1./np.sqrt(ss)
            ssqs[ind]=0
            # ind=np.where(ssqs>1e-2*ssqs.max())[0]
            # Us=Us[:,ind]
            # ssqs=ssqs.flat[ind]
            Us=Us[:,:]
            ssqs=ssqs.flat[:]
            
            while True:
                m=psutil.virtual_memory()
                if m.percent<50.: break
                #print(m.percent)
                time.sleep(3)
            sqrtCs =Us*ssqs.reshape(1,ssqs.size)
            
            while True:
                m=psutil.virtual_memory()
                if m.percent<50.: break
                #print(m.percent)
                time.sleep(3)
            log.print("  creating random set...")
            gArray=np.array([ (g.flatten()+np.dot(sqrtCs,np.random.randn(ssqs.size,1)).flatten()) for iTry in range(NTry)])
            
        GammaStat=np.zeros((NTry,self.GM.NSlice,self.GM.NPix,self.GM.NPix),np.float32)
        for iTry in range(NTry):
            GammaStat[iTry]=(self.GM.giveGammaCube(gArray[iTry],ScaleCube=self.ScaleCube))
        self.GammaStat=GammaStat



        
        # Cube_q0=np.quantile(self.GammaStat,0.15e-2,axis=0)
        # Cube_q1=np.quantile(self.GammaStat,0.9985,axis=0)
        # self.SigmaCube=(Cube_q1-Cube_q0)/6.
        
        Cube_q0=np.quantile(self.GammaStat,0.16,axis=0)
        Cube_q1=np.quantile(self.GammaStat,0.84,axis=0)
        self.SigmaCube=(Cube_q1-Cube_q0)/2.
        self.MedianCube=np.quantile(self.GammaStat,0.5,axis=0)
        self.GammaStat=GammaStat
        
        for iSlice in range(self.GM.NSlice):
            self.MedianCube[iSlice][self.GM.ThisMask==1]=np.nan
            self.SigmaCube[iSlice][self.GM.ThisMask==1]=np.nan

        return GammaStat
    
    def PlotHist(self,g,NTry=100,gArray=None,FullHessian=False):
        if self.XSimul is None: return
        GM=self.GM
        L_NParms=GM.L_NParms

        L=self.CLM.logP(g)
        #self.L_L.append(L)
        LTrue=self.LTrue
        L_L=self.L_L
        
        C=GeneDist.ClassDistMachine()
        
        GammaStat=self.GammaStat
        
        ScaleCube=self.ScaleCube

        CubeBest=self.GM.giveGammaCube(g,ScaleCube=ScaleCube)
        for iSlice in range(self.GM.NSlice):
            CubeBest[iSlice][self.GM.ThisMask==1]=0.
            for iTry in range(NTry):
                GammaStat[iTry,iSlice][self.GM.ThisMask==1]=0.

        
        CubeSimul=self.CubeSimul

        Cube_mean=np.mean(GammaStat,axis=0)
            
        ys=CubeSimul.flatten()
                
        # # ##################################        
        # figH=pylab.figure("hist")
        # figH.clf()
        # ii=0
        # pylab.subplot(2,2,1)
        # for iSlice in range(self.CLM.NSlice):
        #     ThisNParms=L_NParms[iSlice]
        #     iPar=ii
        #     jPar=iPar+ThisNParms
        #     ii+=ThisNParms
        #     x,y=C.giveCumulDist(g[iPar:jPar],Ns=100,Norm=True)#,xmm=[-5,5])
        #     pylab.plot(x,y,color="gray")

        # pylab.plot(x,(Phi(x)),color="black",ls="--")
        # pylab.subplot(2,2,2)
        # pylab.plot((L_L),color="black")
        # pylab.plot(([LTrue]*len(L_L)),ls="--",color="black")

        # ax=pylab.subplot(2,2,3)
        
        # FillBetween(GammaStat,0.15e-2,0.9985)
        # FillBetween(GammaStat,2.5e-2,0.975)
        # FillBetween(GammaStat,0.16,0.84,PlotMed=True)
            
        # # Cube_q50=np.quantile(GammaStat,0.5,axis=0)
        # # v0=Cube_q50.flatten()-ys
        # # x0,y0=C.giveCumulDist(v0,Ns=1000,Norm=True)
        # # pylab.plot(x0,y0,ls="--",color="black")

        # Cube_mean=np.mean(GammaStat,axis=0)
        # v0=Cube_mean.flatten()-ys
        # x0,y0=C.giveCumulDist(v0,Ns=1000,Norm=True)
        # pylab.plot(x0,y0,ls=":",color="red")
        
        # pylab.xlim(-5,5)
        # pylab.grid()
        # pylab.draw()
        # pylab.show(block=False)
        # pylab.pause(0.1)
        # # ##################################        

        # self.GM.PlotGammaCube(Cube=(MeanCube-self.CubeSimul)/eCube,FigName="eCube",vmm=(-3,3))
        
        # self.GM.PlotGammaCube(Cube=(Cube_mean),FigName="log(MeanCube)")
        
        # self.GM.PlotCumulDistX(g)
            
        # figH.savefig("Hist%5.5i.png"%iStep)
        # #self.GM.PlotGammaCube(Cube=y0Cube,FigName="Cube0")
        # #self.GM.PlotGammaCube(Cube=y1Cube,FigName="Cube1")
        # #self.GM.PlotGammaCube(Cube=ycCube,FigName="CubeC")
        # eCube=ycCube/((y1Cube-y0Cube)/2.)
        # #self.GM.PlotGammaCube(Cube=eCube,FigName="eCube")
        # self.GM.PlotGammaCube(Cube=(MeanCube-self.CubeSimul)/eCube,FigName="eCube",vmm=(-3,3))
        # self.GM.PlotCumulDistX(g)

        vmm=[[self.CubeSimul[iSlice].min(),self.CubeSimul[iSlice].max()] for iSlice in range(self.CubeSimul.shape[0])]
        #self.GM.PlotGammaCube(Cube=CubeBest,FigName="Best LogCube",vmm=vmm)
        #self.SaveFig()
        fact=1.8/2.
        figsize=(13/fact,8/fact)
        Nx,Ny=GiveNXNYPanels(self.GM.NSlice,ratio=figsize[0]/figsize[1])
        
        fig=pylab.figure("Diff Sim",figsize=figsize)
        #fig, axes = pylab.subplots(num="Diff Sim",ncols=Ny, nrows=Nx, figsize=(5,5))#, sharex=True, sharey=True)
        fig.clf()
        fig.subplots_adjust(hspace=0,
                            wspace=0,
                            left = 0.05,
                            right = 0.95,
                            bottom = 0.05,
                            top = 0.95)
        
        
        ii=0
        
        #        for iPlot,ax0 in enumerate(axes.flat):
        vmin=-2
        vmax=2
        cmap=pylab.cm.cubehelix
        cmap=pylab.cm.plasma
        cmap=pylab.cm.cividis
        for iPlot in range(self.GM.NSlice):
            iSlice=iPlot
            ax0=fig.add_subplot(Nx,Ny,iPlot+1)
            ax0.imshow(CubeBest[iSlice]-CubeSimul[iSlice],
                       interpolation="nearest",
                       aspect="auto",
                       #extent=(-5,5,0,1),
                       alpha=0.5,
                       vmin=vmin,
                       vmax=vmax,
                       origin="lower",
                       cmap=cmap)
#            ax0.scatter(self.DicoSourceXY[iSlice]["X"],self.DicoSourceXY[iSlice]["Y"],color="black",s=7)


            s=self.DicoSourceXY["P"][:,iSlice]
            rgba_colors = np.zeros((s.size,4))
            rgba_colors[:,1:3] = 0
            rgba_colors[:, 3] = s
            Ns=self.CLM.Cat_s.xCube.size
            dx=0#np.random.rand(Ns)-0.5
            dy=0#np.random.rand(Ns)-0.5
            ax0.scatter(self.DicoSourceXY["X"]+dx,self.DicoSourceXY["Y"]+dy,s=3, color=rgba_colors)#,c="black",2*s[:,iSlice])
            
            # rgba_colors[:,1:3] = 1
            #ax0.scatter(self.CLM.Cat_s.xCube+dx,self.CLM.Cat_s.yCube+dy,s=3, color=rgba_colors)#,c="black",2*s[:,iSlice])
            
            ax0.axis('off')
            ax0.set_xlim(0,self.GM.NPix)
            ax0.set_ylim(0,self.GM.NPix)
            
            ax = fig.add_axes(ax0.get_position(), frameon=False)
            FillBetween(ax,GammaStat,0.15e-2,0.9985,CubeSimul,iSlice=iSlice)
            FillBetween(ax,GammaStat,2.5e-2,0.975,CubeSimul,iSlice=iSlice)
            FillBetween(ax,GammaStat,0.16,0.84,CubeSimul,iSlice=iSlice,PlotMed=True)
            y=np.sort([0,0.15e-2,0.9985,2.5e-2,0.975,0.16,0.84,1])
            x=np.zeros_like(y)
            ax.plot(x,y,ls=":", linewidth=2,color="black")
            ax.scatter(x,y,color="red",s=5,edgecolors="red")
            ax.grid(color='black', linestyle=':')#, linewidth=2)
            
            #pylab.grid()
            pylab.xlim(vmin,vmax)
            pylab.ylim(0,1)
        pylab.draw()
        #mng = pylab.get_current_fig_manager()
        #mng.resize(*mng.window.maxsize())
        pylab.show(block=False)
        pylab.pause(0.01)
        self.SaveFig(fig)

        
        
    def PlotW(self,X,FigName="WDist"):

        fact=1.8
        figsize=(13/fact,8/fact)
        fig=pylab.figure(FigName,figsize=figsize)
        fig.clf()
        Nx,Ny=GiveNXNYPanels(self.GM.NSlice,ratio=figsize[0]/figsize[1])
        fig.clf()
        ii=0
        
        for iPlot,CAD in enumerate(self.CLM.LCSW):
            iSlice=iPlot
            N=self.GM.L_NParms[iSlice]
            x=X[ii:ii+self.GM.L_NParms[iSlice]].flatten()
            ii+=N
            W=CAD.giveW(x)
            ax=pylab.subplot(Nx,Ny,iPlot+1)
            pylab.plot(CAD.empirical_PW.x,CAD.empirical_PW.y,color="black")
            pylab.plot(CAD.empirical_PW.x,np.exp(CAD.logP_W(CAD.empirical_PW.x)),color="black",ls="--")
            pylab.scatter([W],[np.exp(CAD.logP_W(W))],color="red")
        pylab.draw()
        pylab.show(block=False)
        pylab.pause(0.01)


    def PlotHistEigen(self,X,FigName="Hist Eigen"):

        fact=1.8
        figsize=(13/fact,8/fact)
        fig=pylab.figure(FigName,figsize=figsize)
        fig.clf()
        Nx,Ny=GiveNXNYPanels(self.GM.NSlice,ratio=figsize[0]/figsize[1])
        fig.clf()
        ii=0
        
        for iPlot in range(self.GM.NSlice):
            iSlice=iPlot
            N=self.GM.L_NParms[iSlice]
            x=X[ii:ii+self.GM.L_NParms[iSlice]].flatten()
            ii+=N

            NBin=x.size//20
            NBin=np.max([2,NBin])
            BinX=np.int32(np.linspace(0,x.size+1,NBin))
            ax=pylab.subplot(Nx,Ny,iPlot+1)
            cmap=pylab.cm.cubehelix
            cmap=pylab.cm.plasma
            cmap=pylab.cm.cividis
            Lc=cmap(np.linspace(0.1,0.9,NBin))[::-1]
            
            for iBin in range(BinX.size-1):
                i0,i1=BinX[iBin],BinX[iBin+1]
                xx=x[i0:i1]

                C=GeneDist.ClassDistMachine()
                x0,y0=C.giveCumulDist(xx,Ns=1000,Norm=True)
                pylab.plot(x0,y0,color=Lc[iBin])

            xx=np.linspace(-5,5,1000)
            pylab.plot(xx,Phi(xx),color="black",ls="--")
#            pylab.xlim((-5,5))
        pylab.draw()
        pylab.show(block=False)
        pylab.pause(0.01)


        
    def SaveFig(self,fig=None):
        if fig is None:
            fig=pylab.gcf()
        fname=fig.get_label().replace(" ","_")
        if self.PlotID is None:
            oname="%s_%4.4i.png"%(fname,self.iFig)
        else:
            oname="%s_%s_%4.4i.png"%(self.PlotID,fname,self.iFig)
            
            
        log.print(ModColor.Str("Saving fig %s as %s"%(fname,oname),col="blue"))
        fig.savefig(oname)
