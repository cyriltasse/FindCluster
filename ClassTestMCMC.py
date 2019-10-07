import os
os.environ["OMP_NUM_THREADS"] = "1"
from multiprocessing import Pool
import emcee
import numpy as np
import matplotlib.pyplot as plt
import GeneDist
import ClassSimulCatalog
import matplotlib.pyplot as pylab


def g_z(z):
    a=2.
    g=np.zeros_like(z)
    ind=np.where((z>(1./a))&(z<a))[0]
    g[ind]=1./z[ind]
    return g


class ClassTestMCMC():
    def __init__(self):
        rac,decc=241.20678,55.59485 # cluster
        CellDeg=0.001
        self.CellRad=CellDeg*np.pi/180
        NPix=101
        ScaleKpc=500
        CSC=ClassSimulCatalog.ClassSimulCatalog(rac,decc,
                                                #z=[0.01,2.,40],
                                                z=[0.6,0.7,2],
                                                ScaleKpc=ScaleKpc,CellDeg=CellDeg,NPix=NPix)
        CSC.doSimul()


        self.CSC=CSC

        ra0,ra1=self.CSC.rag.min(),self.CSC.rag.max()
        dec0,dec1=self.CSC.decg.min(),self.CSC.decg.max()
        #pylab.ion()
        pylab.figure("Simul")
        pylab.clf()
        pylab.imshow(self.CSC.GammaCube[0].T[::-1,:],extent=(ra0,ra1,dec0,dec1))#,vmin=0,vmax=1.)
        #s=self.Cat.logM
        #s0,s1=s.min(),s.max()
        s=1.#(s-s0)/(s1-s0)*20+5.
        pylab.scatter(self.CSC.Cat.ra,self.CSC.Cat.dec,s=s,linewidths=0)
        pylab.draw()
        pylab.show(False)
        pylab.pause(0.1)
        
        
        self.Cat=CSC.Cat
        self.DistMachine=GeneDist.ClassDistMachine()
        z=np.linspace(-10,10,1000)
        g=g_z(z)
        G=np.cumsum(g)
        G/=G[-1]
        self.DistMachine.setCumulDist(z,G)
        self.z0z1=CSC.zg[0],CSC.zg[1]
        
    def log_prob(self, x):
        z0,z1=self.z0z1
        X=self.Cat.x
        Y=self.Cat.y
        ni=self.Cat.n_i
        iz=self.Cat.iz
        GammaSlice=self.CSC.CGM.SliceFunction(x,z0,z1)
        nx,ny=GammaSlice.shape
        ind=nx*ny*iz+ny*X+Y
        G=GammaSlice.flat[ind]
        n=G*ni
        L=self.CellRad**2*(np.sum(GammaSlice))+np.sum(np.log(n))
        return L
        
    def runMCMC(self):
        
        NDim = self.CSC.CGM.NParms
        NChain = 2*NDim
        
        self.X=np.random.randn(NChain,NDim)*np.mean(np.abs(self.CSC.XSimul))
        self.X=self.CSC.XSimul+np.random.randn(NChain,NDim)*1e-3
        self.X1=self.X.copy()
        self.Accepted=np.zeros((NChain,),int)
        self.L=np.array([self.log_prob(x) for x in self.X])
        self.L1=self.L.copy()
        while True:
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

            pylab.clf()
            pylab.plot(self.X.T)
            pylab.draw()
            pylab.show(False)
            pylab.pause(0.1)
            
            # z0,z1=self.z0z1
            # #X=np.mean(self.X,axis=0)
            # pylab.figure("Fit")
            # pylab.clf()
            # GammaSlice=np.mean([self.CSC.CGM.SliceFunction(x,z0,z1) for x in self.X],axis=0)
            # pylab.imshow(GammaSlice.T[::-1,:])#,vmin=0,vmax=1.)#,extent=(ra0,ra1,dec0,dec1))
            # pylab.draw()
            # pylab.show(False)
            # pylab.pause(0.1)

            
def test():
    C_MCMC=ClassTestMCMC()
    C_MCMC.runMCMC()
