import numpy as np

import numpy as np
import ClassMassFunction
import ClassCatalogMachine
import ClassDisplayRGB
from DDFacet.Other import ClassTimeIt
from DDFacet.ToolsDir.ModToolBox import EstimateNpix
np.random.seed(1)
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
        self.IndexCube_xy_Slice=(np.int64(self.CM.Cat_s.xCube*NPix)+np.int64(self.CM.Cat_s.yCube)).flatten()
        indy,indx=np.where(self.MassFunction.GammaMachine.ThisMask==0)
        #self.IndexCube_Mask=np.array([i*(np.int64(indy).flatten()*NPix+np.int64(indx).flatten()) for i in range(self.NSlice)]).flatten()
        self.IndexCube_Mask=(np.int64(indy).flatten()*NPix+np.int64(indx).flatten())


        
        
    def L(self,g0):
        self.CellRad_0=self.MassFunction.GammaMachine.CellRad
        self.CellRad_1=(.001/3600)*np.pi/180

        g=g0.reshape((-1,1))
        GM=self.MassFunction.GammaMachine
        L_SqrtCov=GM.L_SqrtCov
        L_NParms=GM.L_NParms
        Ns=self.CM.Cat_s.shape[0]
        n_z=self.CM.DicoDATA["DicoSelFunc"]["n_z"]
        n_zt=self.CM.Cat_s.n_zt
        
        SumNx_0=np.sum(n_z.reshape(-1,1,1)*GM.GammaCube)*self.CellRad_0**2
        Nx_1=np.zeros((Ns,),np.float32)
        Ax_1=np.zeros((Ns,),np.float32)
        for iSlice in range(self.NSlice):
            Gamma_i=GM.GammaCube[iSlice].flat[self.IndexCube_xy_Slice]
            Nx_1[:]+=n_z[iSlice]*Gamma_i*self.CellRad_1**2
            Ax_1[:]+=np.log10(n_zt[:,iSlice]*Gamma_i*self.CellRad_1**2)
        SumNx_1=np.sum(Nx_1)
        
        SumAx_1=np.sum(Ax_1)
        return -SumNx_0 + SumNx_1 + SumAx_1 + - (1/2.)*np.dot(g.T,g)

    def dLdg(self,g):
        g=g0.reshape((-1,1))
        GM=self.MassFunction.GammaMachine
        L_SqrtCov=GM.L_SqrtCov
        L_NParms=GM.L_NParms
        NParms=GM.NParms
        Ns=self.CM.Cat_s.shape[0]
        n_z=self.CM.DicoDATA["DicoSelFunc"]["n_z"]
        n_zt=self.CM.Cat_s.n_zt

        ii=0
        J=np.zeros((NParms,),np.float32)
        for iSlice in range(self.NSlice):
            ThisNParms=L_NParms[iSlice]
            iPar=ii
            jPar=iPar+ThisNParms
            ii+=ThisNParms
            GammaSlice=GM.GammaCube[iSlice]
            SqrtCov=L_SqrtCov[iSlice]
            
            dNx_0_dg=n_z[iSlice]*np.sum(SqrtCov[:,:]*GammaSlice.reshape((-1,1)),axis=0)*self.CellRad_0**2
            SqrtCov_xy=SqrtCov[self.IndexCube_xy_Slice,:]
            dNx_1_dg=n_z[iSlice]*np.sum(SqrtCov_xy[:,:]*GammaSlice.flat[self.IndexCube_xy_Slice].reshape((-1,1)),axis=0)*self.CellRad_1**2
            

