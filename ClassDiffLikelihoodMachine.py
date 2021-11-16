import numpy as np

import numpy as np
import ClassMassFunction
import ClassCatalogMachine
import ClassDisplayRGB
from DDFacet.Other import ClassTimeIt
from DDFacet.ToolsDir.ModToolBox import EstimateNpix
#np.random.seed(1)
from DDFacet.Other import logger
log = logger.getLogger("ClassLikelihoodMachine")
from DDFacet.Other.AsyncProcessPool import APP, WorkerProcessError

import ClassInitGammaCube

class ClassLikelihoodMachine():
    def __init__(self,CM):
        self.CM=CM
        
        self.zParms=self.CM.zg_Pars
        self.logMParms=self.CM.logM_Pars
        self.logM_g=np.linspace(*self.logMParms)

        self.MassFunction=ClassMassFunction.ClassMassFunction()
        
        self.MassFunction.setSelectionFunction(self.CM)

        self.NSlice=self.zParms[-1]-1
        
        # APP.registerJobHandlers(self)

    def ComputeIndexCube(self,NPix):
        self.NPix=NPix
        self.IndexCube=np.array([i*np.int64(NPix**2)+np.int64(self.CM.Cat_s.xCube*NPix)+np.int64(self.CM.Cat_s.yCube) for i in range(self.NSlice)]).flatten()
        self.IndexCube_xy=(np.int64(self.CM.Cat_s.xCube*NPix)+np.int64(self.CM.Cat_s.yCube)).flatten()
        indy,indx=np.where(self.MassFunction.GammaMachine.ThisMask==0)
        #self.IndexCube_Mask=np.array([i*(np.int64(indy).flatten()*NPix+np.int64(indx).flatten()) for i in range(self.NSlice)]).flatten()
        self.IndexCube_Mask=(np.int64(indy).flatten()*NPix+np.int64(indx).flatten())

    def InitDiffMatrices(self):
        #self.A=[SqrtCov.dot(for SqrtCov in L_SqrtCov]
        
        Ns=self.CM.Cat_s.shape[0]
        n_z=self.CM.DicoDATA["DicoSelFunc"]["n_z"]
        n_zt=self.CM.Cat_s.n_zt

        self.CellRad_0=self.MassFunction.GammaMachine.CellRad
        
        L_SqrtCov=self.MassFunction.GammaMachine.L_SqrtCov
        L_NParms=self.MassFunction.GammaMachine.L_NParms
        A=np.zeros((self.NPix**2,self.MassFunction.GammaMachine.NParms),np.float32)
        ii=0
        
        for iSlice in range(self.NSlice):
            SqrtCov=L_SqrtCov[iSlice]
            NParms=L_NParms[iSlice]
            A[:,ii:ii+NParms]=(n_z[iSlice]*L_SqrtCov[iSlice])
            ii+=NParms

            
        self.S_dNdg=np.sum(A[self.IndexCube_Mask,:],axis=0)
        self.S_dNdg*= self.CellRad_0**2
        
        B=np.zeros((Ns,self.MassFunction.GammaMachine.NParms),np.float32)
        Bn=np.zeros((Ns,self.MassFunction.GammaMachine.NParms),np.float32)
        Bc=np.zeros((Ns,),np.float32)
        ii=0
        
        for iSlice in range(self.NSlice):
            NParms=L_NParms[iSlice]
            SqrtCov_s=L_SqrtCov[iSlice][self.IndexCube_xy,:]
            Bn[:,ii:ii+NParms]=n_z[iSlice]*SqrtCov_s
            B[:,ii:ii+NParms]=n_zt[:,iSlice].reshape((Ns,1))*SqrtCov_s
            Bc[:]+=n_zt[:,iSlice]
            ii+=NParms
        
        self.CellRad_1=(.001/3600)*np.pi/180
        self.S_dAdg_N = Bn * self.CellRad_1**2
        self.S_dAdg_A = B * self.CellRad_1**2
        self.S_dAdg_B = Bc.reshape((-1,1)) * self.CellRad_1**2


        # ###################################

        # g=np.ones((self.MassFunction.GammaMachine.NParms,1),np.float32)

        # A=np.zeros((Ns,self.NSlice),np.float32)
        # ii=0
        # for iSlice in range(self.NSlice):
        #     SqrtCov=L_SqrtCov[iSlice]
        #     NParms=L_NParms[iSlice]
        #     SqrtCov_s=L_SqrtCov[iSlice][self.IndexCube_xy,:]
        #     gs=(g.flat[ii:ii+NParms]).reshape((-1,1))
        #     n=n_zt[:,iSlice].reshape((-1,1))*CellRad_1**2
        #     A[:,iSlice]=(n*(1.+np.dot(SqrtCov_s,gs))).flatten()
        #     ii+=NParms
        # A0=np.sum(A,axis=1)
        
        # A1=self.S_dAdg_B+np.dot(self.S_dAdg_A,g)
        # print(A0,A1)
        # stop
        
        
    def L(self,g0):
        g=g0.reshape((-1,1))
        Nx=np.dot(self.S_dNdg,g)
        Nxi= np.dot(self.S_dAdg_N,g)
        
        Ns=self.CM.Cat_s.shape[0]
        n_z=self.CM.DicoDATA["DicoSelFunc"]["n_z"]
        n_zt=self.CM.Cat_s.n_zt

        # L_SqrtCov=self.MassFunction.GammaMachine.L_SqrtCov
        # L_NParms=self.MassFunction.GammaMachine.L_NParms
        
        # A=np.zeros((Ns,self.NSlice),np.float32)
        # ii=0
        # for iSlice in range(self.NSlice):
        #     SqrtCov=L_SqrtCov[iSlice]
        #     NParms=L_NParms[iSlice]
        #     SqrtCov_s=L_SqrtCov[iSlice][self.IndexCube_xy,:]
        #     gs=(g.flat[ii:ii+NParms]).reshape((-1,1))
        #     n=n_zt[:,iSlice].reshape((-1,1))*self.CellRad_1**2
        #     A[:,iSlice]=(n*(1.+np.dot(SqrtCov_s,gs))).flatten()
        #     ii+=NParms

        
        #SumPi_z=np.sum(A,axis=1)
        
        SumPi_z=self.S_dAdg_B+np.dot(self.S_dAdg_A,g)
        #print(SumPi_z,SumPi_z1)
        #stop
        Sum_Log_SumPi_z=np.sum(np.log(SumPi_z),axis=0)
        
        
        #Sum_Log_SumPi_z1=np.sum(np.log(self.S_dAdg_B+np.dot(self.S_dAdg_A,g)))
        
        return -np.sum(Nx) + np.sum(Nxi) + Sum_Log_SumPi_z - (1/2.)*np.dot(g.T,g)
    
        

    def dLdg(self,g):
        Ns=self.CM.Cat_s.shape[0]
        n_z=self.CM.DicoDATA["DicoSelFunc"]["n_z"]
        n_zt=self.CM.Cat_s.n_zt
        #GammaCube=self.MassFunction.GammaMachine.GammaCube
        #SqrtCg=GammaCube-1.

        dNxdg= - self.S_dNdg

        a=np.dot(self.S_dAdg_A,g.reshape((-1,1)))
        dAxdg= np.sum(self.S_dAdg_N,axis=0) +  np.sum( self.S_dAdg_A / (self.S_dAdg_B + a ) , axis=0)

        
        dLdg=dNxdg+dAxdg-g

        return dLdg

        
        
    def log_prob(self, X):
        T=ClassTimeIt.ClassTimeIt()
        T.disable()
        self.MassFunction.updateGammaCube(X)
        T.timeit("update cube")
        Cell=(.001/3600)*np.pi/180
        GammaCube=self.MassFunction.GammaMachine.GammaCube
        n_z=self.CM.DicoDATA["DicoSelFunc"]["n_z"]
        Nx=np.sum(GammaCube*n_z.reshape((-1,1,1)),axis=0)
        T.timeit("Nx")
        Nx_Omega0=np.sum(Nx)*self.MassFunction.GammaMachine.CellRad**2
        T.timeit("Nx_Omega0")
        
        Ns=self.CM.Cat_s.shape[0]
        
        gamma_xz=GammaCube.flat[self.IndexCube].reshape((self.NSlice,Ns)).T
        
        # gamma_xz0=np.array([GammaCube[iz].flat[self.IndexCube] for iz in range(self.NSlice)]).reshape((self.NSlice,Ns)).T
        # gamma_xz=np.zeros((self.NSlice,Ns),np.float64)
        # for iS in range(self.CM.Cat_s.shape[0]):
        #     gamma_xz[:,iS]=GammaCube[:,self.CM.Cat_s.xCube[iS],self.CM.Cat_s.yCube[iS]]
        # gamma_xz=gamma_xz.T
        
        T.timeit("gamma_xz")

        # gamma_xz2=np.zeros_like(GammaCube)
        # gamma_xz2[:,self.CM.Cat_s.xCube[iS],self.CM.Cat_s.yCube[iS]]=32
        # import pylab
        # pylab.figure("test")
        # pylab.clf()
        # pylab.imshow(gamma_xz2[1],interpolation="nearest")
        # pylab.draw()
        # pylab.show(False)
        
        Nx_Omega1=np.sum(gamma_xz*n_z.reshape((1,-1)))*Cell**2
        T.timeit("Nx_Omega1")

        # S0=np.sum(gamma_xz*self.CM.Cat_s.n_zt*Cell**2,axis=1)
        # S0[S0<=0]=1e-100
        # Nx1_Omega1=np.sum(np.log(S0))


        S0=np.sum(gamma_xz*self.CM.Cat_s.n_zt*Cell**2,axis=1)
        S0[S0<=0]=1e-100
        Nx1_Omega1=np.sum(np.log(S0))


        T.timeit("Nx1_Omega1")
        L=-np.float64(Nx_Omega0)+np.float64(Nx_Omega1)+np.float64(Nx1_Omega1)

        return L
        return L,-Nx_Omega0,Nx_Omega1,Nx1_Omega1

