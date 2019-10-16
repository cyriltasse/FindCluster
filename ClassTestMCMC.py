import os
os.environ["OMP_NUM_THREADS"] = "1"
from multiprocessing import Pool
#import emcee
import numpy as np
import matplotlib.pyplot as plt
import GeneDist
import ClassSimulCatalog
import matplotlib.pyplot as pylab
import ClassMassFunction
from astropy.cosmology import WMAP9 as cosmo

def g_z(z):
    a=2.
    g=np.zeros_like(z)
    ind=np.where((z>(1./a))&(z<a))[0]
    g[ind]=1./z[ind]
    return g


class ClassTestMCMC():
    def __init__(self):
        rac,decc=241.20678,55.59485 # cluster
        self.CellDeg=0.001
        self.CellRad=self.CellDeg*np.pi/180
        self.NPix=101
        self.ScaleKpc=500
        
        np.random.seed(6)
        self.XSimul=np.random.randn(9)*2
        #self.XSimul=np.random.randn(1)
        #self.XSimul.fill(0)
        self.XSimul[0]=10.
        self.rac_deg,self.decc_deg=rac,decc
        self.rac,self.decc=rac*np.pi/180,decc*np.pi/180
        
        self.zParms=[0.6,0.7,2]
        self.logMParms=[10,10.5,2]
        self.logM_g=np.linspace(*self.logMParms)
        CSC=ClassSimulCatalog.ClassSimulCatalog(rac,decc,
                                                #z=[0.01,2.,40],
                                                z=self.zParms,
                                                ScaleKpc=self.ScaleKpc,
                                                CellDeg=self.CellDeg,
                                                NPix=self.NPix,
                                                XSimul=self.XSimul,
                                                logMParms=self.logMParms)
        CSC.doSimul()


        self.CSC=CSC
        

        
        self.Cat=CSC.Cat
        self.DistMachine=GeneDist.ClassDistMachine()
        z=np.linspace(-10,10,1000)
        g=g_z(z)
        G=np.cumsum(g)
        G/=G[-1]
        self.DistMachine.setCumulDist(z,G)
        self.z0z1=CSC.zg[0],CSC.zg[1]
        
    # def log_prob(self, x):
    #     z0,z1=self.z0z1
    #     X=self.Cat.x
    #     Y=self.Cat.y
    #     ni=self.Cat.n_i
    #     iz=self.Cat.iz
    #     GammaSlice=self.CSC.CGM.SliceFunction(x,z0,z1)
    #     nx,ny=GammaSlice.shape
    #     ind=nx*ny*iz+ny*X+Y
    #     G=GammaSlice.flat[ind]
    #     n=G*ni
    #     L=self.CellRad**2*(np.sum(GammaSlice))+np.sum(np.log(n))
    #     return L

    def log_prob(self, x):
        z0,z1=self.z0z1
        X=self.Cat.x
        Y=self.Cat.y
        
        ni=self.Cat.n_i
        iz=self.Cat.iz

        # ######################################
        # Init Mass Function for that one X
        MassFunc=ClassMassFunction.ClassMassFunction()
        MassFunc.setGammaFunction((self.rac,self.decc),
                                  self.CellDeg,
                                  self.NPix,
                                  z=self.zParms,
                                  ScaleKpc=self.ScaleKpc)
        LX=[x]
        MassFunc.CGM.computeGammaCube(LX)
        GammaSlice=MassFunc.CGM.GammaCube[0]
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
            V=dz*dV_dz*self.CellRad**2
            
            n0=MassFunc.givePhiM(zm,logMm)*dlogM*V
            L+=-np.sum(GammaSlice)*n0
            for iS in range(self.Cat.ra.size):
                n=MassFunc.give_N((self.Cat.ra[iS],self.Cat.dec[iS]),
                                  (z0,z1),
                                  (logM0,logM1),
                                  OmegaSr)
                L+=np.log(n)
                #L+=np.log(OmegaSr)
        return L

    
    
    def runMCMC(self):
        
        GammaCube=self.CSC.MassFunc.CGM.GammaCube
        ra0,ra1=self.CSC.rag.min(),self.CSC.rag.max()
        dec0,dec1=self.CSC.decg.min(),self.CSC.decg.max()
        #pylab.ion()
        vmin=0
        vmax=np.max(GammaCube[0])
        pylab.figure("Simul")
        pylab.clf()
        pylab.imshow(GammaCube[0].T[::-1,:],extent=(ra0,ra1,dec0,dec1),cmap="cubehelix",vmin=vmin,vmax=vmax)
        #s=self.Cat.logM
        #s0,s1=s.min(),s.max()
        s=10.#(s-s0)/(s1-s0)*20+5.
        pylab.scatter(self.CSC.Cat.ra,self.CSC.Cat.dec,s=s,linewidths=0)
        #pylab.colorbar()
        pylab.draw()
        pylab.show(False)
        pylab.pause(0.1)

        NDim = self.CSC.MassFunc.CGM.NParms
        NChain = 4*NDim
        
        self.X=np.random.randn(NChain,NDim)*np.mean(np.abs(self.CSC.XSimul))
        #self.X=self.CSC.XSimul+np.random.randn(NChain,NDim)*1.
        
        self.X1=self.X.copy()
        self.Accepted=np.zeros((NChain,),int)
        self.L=np.array([self.log_prob(x) for x in self.X])
        self.L1=self.L.copy()
        self.ListX=[]
        self.ListL=[]
        iDone=0
        while True:
            iDone+=1
            for iChain in range(NChain):
                #print iChain,NChain
                while True:
                    k=int(np.random.rand(1)[0]*NChain)
                    if k!=iChain: break
                
                z=self.DistMachine.GiveSample(1)[0]
                X0=self.X[iChain]
                X1=self.X[k]
                
                X2=X0+z*(X1-X0)
                L1=self.log_prob(X2)
                L0=self.L[iChain]
                pp=z**(NDim-1)*np.exp(L1-L0)
                p=np.min([1,pp])
                print "   %f, %f --> %f"%(L1,L0,p)
                r=np.random.rand(1)[0]
                if r<p:
                    self.X1[iChain]=X2[:]
                    self.L1[iChain]=L1
                    self.Accepted[iChain]=1
                else:
                    self.X1[iChain]=X0[:]
                    self.L1[iChain]=L0
                    self.Accepted[iChain]=0
                    
            print np.count_nonzero(self.Accepted)/float(self.Accepted.size)
            self.X[:]=self.X1[:]
            self.L[:]=self.L1[:]
            self.ListX.append(self.X.copy())
            self.ListL.append(self.L.copy())
            pylab.figure("Fit")
            pylab.clf()
            #pylab.plot(self.X.T)
            pylab.subplot(2,1,1)
            pylab.plot(np.array(self.ListX)[:,:,0])
            pylab.subplot(2,1,2)
            pylab.plot(np.array(self.ListL)[:,:,0])
            pylab.draw()
            pylab.show(False)
            pylab.pause(0.1)
            
            z0,z1=self.z0z1
            #X=np.mean(self.X,axis=0)
            fig=pylab.figure("Fit imshow",figsize=(10,5))
            pylab.clf()
            
            MassFunc=ClassMassFunction.ClassMassFunction()
            MassFunc.setGammaFunction((self.rac,self.decc),
                                      self.CellDeg,
                                      self.NPix,
                                      z=self.zParms,
                                      ScaleKpc=self.ScaleKpc)

            LSlice=[]
            for x in self.X:
                LX=[x]
                MassFunc.CGM.computeGammaCube(LX)
                LSlice.append(MassFunc.CGM.GammaCube[0])
            GammaSlice=np.mean(np.array(LSlice),axis=0)
            stdGammaSlice=np.std(np.array(LSlice),axis=0)
            pylab.subplot(1,2,1)
            pylab.imshow(GammaSlice.T[::-1,:],vmin=vmin,vmax=vmax,extent=(ra0,ra1,dec0,dec1),cmap="cubehelix")
            pylab.subplot(1,2,2)
            pylab.imshow(stdGammaSlice.T[::-1,:],vmin=vmin,vmax=vmax,extent=(ra0,ra1,dec0,dec1),cmap="cubehelix")
            pylab.draw()
            pylab.show(False)
            pylab.pause(0.1)
            fig.savefig("Overd%3.3i.png"%iDone)
            
def test():
    C_MCMC=ClassTestMCMC()
    C_MCMC.runMCMC()
