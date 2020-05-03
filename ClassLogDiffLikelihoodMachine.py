import numpy as np
import pylab
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
from ClassAndersonDarling import *
from ClassShapiroWilk import *
from ClassEigenShapiroWilk import *

class ClassLikelihoodMachine():
    def __init__(self,CM):
        self.CM=CM
        
        self.zParms=self.CM.zg_Pars
        self.logMParms=self.CM.logM_Pars
        self.logM_g=np.linspace(*self.logMParms)

        self.MassFunction=ClassMassFunction.ClassMassFunction()
        
        self.MassFunction.setSelectionFunction(self.CM)

        self.NSlice=self.zParms[-1]-1
        self.MAP=1
        
            
        # APP.registerJobHandlers(self)

    def ComputeIndexCube(self,NPix):
        self.NPix=NPix
        self.IndexCube=np.array([i*np.int64(NPix**2)+np.int64(self.CM.Cat_s.xCube*NPix)+np.int64(self.CM.Cat_s.yCube) for i in range(self.NSlice)]).flatten()
        
        X,Y=self.CM.Cat_s.xCube,self.CM.Cat_s.yCube
        
        # # X=np.int64(np.random.rand(X.size)*NPix)
        # # Y=np.int64(np.random.rand(X.size)*NPix)
        # # X.fill(NPix//2)
        # # Y.fill(NPix//2)
        self.MassFunction.GammaMachine.ThisMask.fill(0)
        
        self.IndexCube_xy_Slice=(np.int64(X*NPix)+np.int64(Y)).flatten()
        indy,indx=np.where(self.MassFunction.GammaMachine.ThisMask==0)
        #self.IndexCube_Mask=np.array([i*(np.int64(indy).flatten()*NPix+np.int64(indx).flatten()) for i in range(self.NSlice)]).flatten()
        self.IndexCube_Mask_Slice=(np.int64(indy).flatten()*NPix+np.int64(indx).flatten())
        self.IndexCube_Mask=np.array([i*NPix**2+(np.int64(indy).flatten()*NPix+np.int64(indx).flatten()) for i in range(self.NSlice)]).flatten()
        if self.MAP:
            GM=self.MassFunction.GammaMachine
            self.LCAD=[]
            self.LCSW=[]
            self.PowerMAP=1.#4.#2.5
            # for iSlice in range(self.NSlice):
            #     CAD=ClassAndersonDarlingMachine()
            #     CAD.generatePA2(GM.L_NParms[iSlice],NTry=2000)
            #     self.LCAD.append(CAD)
                
            # self.CSWFull=ClassShapiroWilk()
            # self.CSWFull.Init(GM.NParms,NTry=2000)
            self.CSW=ClassEigenShapiroWilk(GM.L_NParms)

            # for iSlice in range(self.NSlice):
            #     log.print("======================================="%iSlice)
            #     log.print("Init ShapiroWilk for slice %i"%iSlice)
            #     CAD=ClassShapiroWilk()
            #     CAD.Init(GM.L_NParms[iSlice],NTry=2000)
            #     self.LCSW.append(CAD)
                
    def measure_dlogPdg(self,g0,DoPlot=0):
        g=g0.copy()
        L0=self.logL(g,DoPlot=DoPlot)

        dL_dg0=self.dlogLdg(g)
        NN=g.size
        dg=.001

        NTest=100
        # Parm_id=np.arange(NN)[::-1][0:NTest]
        Parm_id=np.int64(np.random.rand(NTest)*g.size)
        dL_dg1=np.zeros((Parm_id.size,),np.float64)
        
        for i in range(Parm_id.size):
            g1=g.copy()
            g1[Parm_id[i]]+=dg
            L1=self.logL(g1,DoPlot=DoPlot)
            dL_dg1[i]=(L0-L1)/((g-g1)[Parm_id[i]])
            if (L0-L1)==0:
                print("  0 diff for iParm = %i"%i)

        pylab.figure("Jacob")
        pylab.clf()
        # pylab.plot(dL_dg0[Parm_id]/dL_dg1)
        pylab.plot(dL_dg0[Parm_id],label="Computed")
        pylab.plot(dL_dg1,ls="--",label="Measured")
        pylab.legend()
        pylab.draw()
        pylab.show(block=False)
        pylab.pause(0.1)

    # #############################################

    def measure_d2logPdg2(self,g0,DoPlot=0):
        
        g=g0.copy()

        NN=g.size
        dg=1e-2

        NTest=1
        Parm_id=np.int64(np.random.rand(NTest)*g.size)
        Parm_id=np.arange(NN)

        dJdg0=self.d2logPdg2(g,Diag=True)
        dJdg0_Full=self.d2logPdg2(g,
                                  Diag=False,
                                  ParmId=Parm_id)
        
        #dJdg2=self.buildFulldJdg(g,ParmId=Parm_id)
        pylab.figure("Hessian")
        pylab.clf()
        pylab.plot(dJdg0[Parm_id],label="CalcDiag",color="black")
        pylab.plot(np.diag(dJdg0_Full)[Parm_id],label="np.diag(CalcFull)",ls="--",lw=2)
        # #pylab.plot(np.diag(dJdg2)[Parm_id])
        # pylab.draw()
        # pylab.show(block=False)
        # pylab.pause(0.1)

        dLdg_0=self.dlogPdg(g)
        dJdg1_row=np.zeros((Parm_id.size,dLdg_0.size),np.float64)
        dJdg1=np.zeros((Parm_id.size,),np.float64)
        for i in range(Parm_id.size):
            print("%i/%i"%(i,Parm_id.size))
            g1=g.copy()
            g1[Parm_id[i]]+=dg
            dLdg_1=self.dlogPdg(g1)
            dJdg1[i]=(dLdg_0[Parm_id[i]]-dLdg_1[Parm_id[i]]) / ((g-g1)[Parm_id[i]])
            dJdg1_row[i,:]=(dLdg_1-dLdg_0) / dg

        # pylab.plot(dL_dg0[Parm_id]/dL_dg1)
        pylab.plot(dJdg1,label="measure of diag",ls="-.",lw=2)
        pylab.legend()
        pylab.draw()
        pylab.show(block=False)
        pylab.pause(0.1)

        
        #I0=np.log10(np.abs(dJdg0_Full))
        #I1=np.log10(np.abs(dJdg1_row))
        I0=(np.abs(dJdg0_Full))
        I1=(np.abs(dJdg1_row))
        I0[I0<=0]=1e-10
        I1[I1<=0]=1e-10
        I1=np.log10(I1)
        I0=np.log10(I0)
        v0=np.min([I0.min(),I1.min()])
        v1=np.max([I0.max(),I1.max()])
        #m=np.median(I0[I0>0])
        #v0,v1=0,100*m
        
        pylab.figure("Full Hessian")
        pylab.clf()
        
        ax=pylab.subplot(2,2,1)
        pylab.imshow(I1,interpolation="nearest",vmin=v0,vmax=v1)
        pylab.title("measured")
        pylab.subplot(2,2,2,sharex=ax,sharey=ax)
        pylab.imshow(I0,interpolation="nearest",vmin=v0,vmax=v1)
        pylab.title("computed")
        pylab.subplot(2,2,3,sharex=ax,sharey=ax)
        pylab.imshow(I1-I0,interpolation="nearest",vmin=-1,vmax=1)
        pylab.title("resid")
        pylab.colorbar()
        
        # ind=np.int64(np.random.rand(1000)*dJdg1_row.size)
        # x=dJdg1_row.T.flatten()[ind]
        # y=dJdg0_Full[Parm_id,:].T.flatten()[ind]
        # #x=np.log10(np.abs(x))
        # #y=np.log10(np.abs(y))
        # #x=np.log10(np.abs(x))
        # #y=np.log10(np.abs(y))
        # pylab.plot(x,label="row measured",color="black")
        # pylab.plot(y,ls="--",label="row computed")
        # # pylab.plot(dJdg1_row.T,label="row measured",color="black")
        # # pylab.plot(dJdg0_Full[Parm_id,:].T,ls="--",label="row computed")
        # pylab.legend()
        
        pylab.draw()
        pylab.show(False)
        pylab.pause(0.1)

        

        
    # ################################################

    def logP(self,g):
        logL=self.logL(g)
        
        #logL=0.
        if self.MAP:
            logL+=self.CSW.logP_x(g.flatten())
        return logL
    

    def dlogPdg(self,g):
        NParms=self.MassFunction.GammaMachine.NParms
        J=np.zeros((NParms,),np.float32)
        J+=self.dlogLdg(g)
        #J.fill(0)
        if self.MAP:
            J+=self.CSW.dlogPdx(g.flatten())
        return J
    
    def d2logPdg2(self,g0,Diag=True,ParmId=None):
        if Diag:
            H=self.d2logLdg2(g0)
            #H.fill(0)
            if self.MAP:
                H[:]+=self.CSW.d2logPdx2_Diag(g0.flatten())
        else:
            # H=np.zeros((g0.size,g0.size),np.float64)
            # dg=1e-3
            # dLdg0=self.dlogLdg(g0)
            # if ParmId is None:
            #     ParmId=np.arange(g0.size)
            # for ii,i in enumerate(ParmId):
            #     print("%i/%i"%(ii,ParmId.size))
            #     g1=g0.copy()
            #     g1[i]+=dg
            #     dLdg1=self.dlogLdg(g1)
            #     H[i]=(dLdg1-dLdg0) / dg
                
            H=self.d2logLdg2_Full(g0)
            #H.fill(0)
            if self.MAP:
                H+=self.CSW.d2logPdx2_Full(g0)
                
        return H
    
    # ################################################
        
    def logprob(self,g):
        return np.float64(self.logP(g)),np.float64(self.dlogPdg(g))


    def logL(self,g0,DoPlot=0):
        GM=self.MassFunction.GammaMachine
        g0=np.float64(g0)
        
        GM.computeGammaCube(g0)
        self.CellRad_0=self.MassFunction.GammaMachine.CellRad
        self.CellRad_1=(.00001/3600)*np.pi/180

        g=g0.reshape((-1,1))
        L_SqrtCov=GM.L_SqrtCov
        L_NParms=GM.L_NParms
        Ns=self.CM.Cat_s.shape[0]
        n_z=self.CM.DicoDATA["DicoSelFunc"]["n_z"]
        n_zt=self.CM.Cat_s.n_zt

        TypeSum=np.float64
        #TypeSum=np.float32

        
        # self.funcNormLog=np.log10
        # self.fNormLog=np.log(10)

        self.funcNormLog=np.log
        self.fNormLog=1.

        
        if DoPlot: GM.PlotGammaCube(Cube=GM.GammaCube,FigName="JacobCube")
        
        Nx_1=np.zeros((Ns,),np.float64)
        # Ax_1=np.zeros((Ns,),np.float32)
        L_Ax_1_z=[]
        for iSlice in range(self.NSlice):
            Gamma_i=GM.GammaCube[iSlice].flat[self.IndexCube_xy_Slice]
            Nx_1[:]+=n_z[iSlice]*Gamma_i*self.CellRad_1**2
            Ax_1_z=n_zt[:,iSlice]*Gamma_i*self.CellRad_1**2
            #Ax_1[:]+=Ax_1_z
            L_Ax_1_z.append(Ax_1_z)

        L=np.float64([0.])
        # #######################
        # SumNx_0=np.sum(TypeSum((n_z.reshape(-1,1,1)*self.CellRad_0**2)*GM.GammaCube))
        SumNx_0=np.sum(TypeSum(n_z.reshape(-1,1,1)*GM.GammaCube).flat[self.IndexCube_Mask])*self.CellRad_0**2/self.fNormLog
        L+=-SumNx_0

        # #######################
        SumNx_1=np.sum(TypeSum(Nx_1))
        L+= SumNx_1

        
        # #######################
        self.Ax_1_z=np.array(L_Ax_1_z)
        self.Ax_1=np.sum(TypeSum(self.Ax_1_z),axis=0)
        SumAx_1=np.sum(self.funcNormLog(TypeSum(self.Ax_1)))
        L+= SumAx_1
        
        self.gCurrentL=g0.copy()
        return L[0]

    
    
    def dlogLdg(self,g0,Slice=None):

        T=ClassTimeIt.ClassTimeIt()
        T.disable()
        GM=self.MassFunction.GammaMachine
        GM.computeGammaCube(g0)
        g=g0.reshape((-1,1))
        T.timeit("Compute Gamma")
        L_SqrtCov=GM.L_SqrtCov
        L_NParms=GM.L_NParms
        NParms=GM.NParms
        Ns=self.CM.Cat_s.shape[0]
        n_z=self.CM.DicoDATA["DicoSelFunc"]["n_z"]
        n_zt=self.CM.Cat_s.n_zt

        # Sum_z_Ax_1_z=np.sum(self.Ax_1_z,axis=0)

        L_Ax_1_z=[]
        for iSlice in range(self.NSlice):
            Gamma_i=GM.GammaCube[iSlice].flat[self.IndexCube_xy_Slice]
            Ax_1_z=n_zt[:,iSlice]*Gamma_i*self.CellRad_1**2
            L_Ax_1_z.append(Ax_1_z)
        Ax_1_z=np.array(L_Ax_1_z)
        Sum_z_Ax_1_z=np.sum(Ax_1_z,axis=0)
        # print(GM.GammaCube.flat[0])
            
        ii=0
        J=np.zeros((NParms,),np.float64)
        if Slice is None:
            Sl=slice(None)
        else:
            Sl=slice(Slice,Slice+1)
            
        
        for iSlice in range(self.NSlice)[Sl]:
            ThisNParms=L_NParms[iSlice]
            iPar=ii
            jPar=iPar+ThisNParms
            
            GammaSlice=GM.GammaCube[iSlice]
            SqrtCov=L_SqrtCov[iSlice]
            SqrtCov_xy=SqrtCov[self.IndexCube_xy_Slice,:]

            # #######################
            dNx_0_dg=n_z[iSlice]*SqrtCov[:,:]*GammaSlice.reshape((-1,1))*self.CellRad_0**2
            Sum_dNx_0_dg=np.sum(dNx_0_dg[self.IndexCube_Mask_Slice,:],axis=0)
            J[iPar:jPar]+= -Sum_dNx_0_dg

            # #########################
            dNx_1_dg=n_z[iSlice]*SqrtCov_xy[:,:]*GammaSlice.flat[self.IndexCube_xy_Slice].reshape((-1,1))*self.CellRad_1**2
            Sum_dNx_1_dg=np.sum(dNx_1_dg,axis=0)
            J[iPar:jPar]+= Sum_dNx_1_dg

            # #########################
            dAx_dg_0 = n_zt[:,iSlice].reshape((-1,1))*SqrtCov_xy[:,:]*GammaSlice.flat[self.IndexCube_xy_Slice].reshape((-1,1))#*np.log(10)
            dAx_dg_1 = Sum_z_Ax_1_z
            dAx_dg   = dAx_dg_0/dAx_dg_1.reshape((-1,1))*self.CellRad_1**2
            Sum_dAx_dg=np.sum(dAx_dg,axis=0)
            J[iPar:jPar]+= + Sum_dAx_dg
                
            ii+=ThisNParms

        return J

    # #################################
    # HESSIANs

    def d2logLdg2(self,g0):

        T=ClassTimeIt.ClassTimeIt()
        T.disable()
        GM=self.MassFunction.GammaMachine
        GM.computeGammaCube(g0)
        g=g0.reshape((-1,1))
        T.timeit("Compute Gamma")
        L_SqrtCov=GM.L_SqrtCov
        L_NParms=GM.L_NParms
        NParms=GM.NParms
        Ns=self.CM.Cat_s.shape[0]
        n_z=self.CM.DicoDATA["DicoSelFunc"]["n_z"]
        n_zt=self.CM.Cat_s.n_zt

        # Sum_z_Ax_1_z=np.sum(self.Ax_1_z,axis=0)
        L_Ax_1_z=[]
        for iSlice in range(self.NSlice):
            Gamma_i=GM.GammaCube[iSlice].flat[self.IndexCube_xy_Slice]
            Ax_1_z=n_zt[:,iSlice]*Gamma_i*self.CellRad_1**2
            L_Ax_1_z.append(Ax_1_z)
        Ax_1_z=np.array(L_Ax_1_z)
        Sum_z_Ax_1_z=np.sum(Ax_1_z,axis=0)
            
        ii=0
        H=np.zeros((NParms,),np.float32)
        for iSlice in range(self.NSlice):
            ThisNParms=L_NParms[iSlice]
            iPar=ii
            jPar=iPar+ThisNParms
            
            GammaSlice=GM.GammaCube[iSlice]
            SqrtCov=L_SqrtCov[iSlice]
            SqrtCov_xy=SqrtCov[self.IndexCube_xy_Slice,:]

            # ##################################"
            dNx_0_dg=n_z[iSlice] * (SqrtCov[:,:])**2 * GammaSlice.reshape((-1,1))*self.CellRad_0**2*self.fNormLog
            Sum_dNx_0_dg=np.sum(dNx_0_dg[self.IndexCube_Mask_Slice,:],axis=0)
            H[iPar:jPar]+= -Sum_dNx_0_dg

            
            # ##################################"
            dNx_1_dg=n_z[iSlice]*(SqrtCov_xy[:,:])**2 * GammaSlice.flat[self.IndexCube_xy_Slice].reshape((-1,1))*self.CellRad_1**2
            Sum_dNx_1_dg=np.sum(dNx_1_dg,axis=0)
            H[iPar:jPar]+= + Sum_dNx_1_dg

            
            # ##################################"
            dAx_dg_0 = n_zt[:,iSlice].reshape((-1,1))* (SqrtCov_xy[:,:]) \
                       * GammaSlice.flat[self.IndexCube_xy_Slice].reshape((-1,1))#*np.log(10)
            dAx_dg_1 = Sum_z_Ax_1_z
            dAx_dg_A   = SqrtCov_xy[:,:]*dAx_dg_0 / dAx_dg_1.reshape((-1,1)) * self.CellRad_1**2
            dAx_dg_B   = - ( (dAx_dg_0 * dAx_dg_0) * self.CellRad_1**4 / (dAx_dg_1.reshape((-1,1)))**2 )
            dAx_dg = dAx_dg_A + dAx_dg_B
            Sum_dAx_dg=np.sum(dAx_dg,axis=0)
            H[iPar:jPar]+= + Sum_dAx_dg
            
            ii+=ThisNParms

        
        return H

    # #############################
    # FULLLLL
    
    def d2logLdg2_Full(self,g0):

        T=ClassTimeIt.ClassTimeIt()
        T.disable()
        GM=self.MassFunction.GammaMachine
        GM.computeGammaCube(g0)
        g=g0.reshape((-1,1))
        T.timeit("Compute Gamma")
        L_SqrtCov=GM.L_SqrtCov
        L_NParms=GM.L_NParms
        NParms=GM.NParms
        Ns=self.CM.Cat_s.shape[0]
        n_z=self.CM.DicoDATA["DicoSelFunc"]["n_z"]
        n_zt=self.CM.Cat_s.n_zt

        # Sum_z_Ax_1_z=np.sum(self.Ax_1_z,axis=0)
        L_Ax_1_z=[]
        for iSlice in range(self.NSlice):
            Gamma_i=GM.GammaCube[iSlice].flat[self.IndexCube_xy_Slice]
            Ax_1_z=n_zt[:,iSlice]*Gamma_i*self.CellRad_1**2
            L_Ax_1_z.append(Ax_1_z)
        Ax_1_z=np.array(L_Ax_1_z)
        Sum_z_Ax_1_z=np.sum(Ax_1_z,axis=0)

        DicoIndex={}
        ii=0
        for iSlice in range(self.NSlice):
            ThisNParms=L_NParms[iSlice]
            iPar=ii
            jPar=iPar+ThisNParms
            DicoIndex[iSlice]=(iPar,jPar)
            ii+=ThisNParms
            
        ii=0
        H=np.zeros((NParms,NParms),np.float32)
        
        dAidg=np.zeros((Ns,NParms),np.float32)
        for iSlice in range(self.NSlice):
            GammaSlice=GM.GammaCube[iSlice]
            i0,i1=DicoIndex[iSlice]
            ThisNParms=i1-i0
            SqrtCov=L_SqrtCov[iSlice]
            SqrtCov_xy=SqrtCov[self.IndexCube_xy_Slice,:]
            iPar,jPar=i0,i1
            ng=n_zt[:,iSlice].reshape((-1,1))* GammaSlice.flat[self.IndexCube_xy_Slice].reshape((-1,1))
            dAidg[:,i0:i1] = ng * SqrtCov_xy.reshape((Ns,ThisNParms)) 
        
        for iSlice in range(self.NSlice):
            GammaSlice=GM.GammaCube[iSlice]
            i0,i1=DicoIndex[iSlice]
            ThisNParms=i1-i0
            SqrtCov=L_SqrtCov[iSlice]
            SqrtCov_xy=SqrtCov[self.IndexCube_xy_Slice,:]
            iPar,jPar=i0,i1
            
            
            # ##################################
            dNx_0_dg=n_z[iSlice]\
                      * SqrtCov.reshape((self.NPix**2,ThisNParms,1))\
                      * SqrtCov.reshape((self.NPix**2,1,ThisNParms))\
                      * GammaSlice.reshape((-1,1,1))*self.CellRad_0**2*self.fNormLog
            Sum_dNx_0_dg=np.sum(dNx_0_dg[self.IndexCube_Mask_Slice,...],axis=0)
            H[iPar:jPar,iPar:jPar]+= - Sum_dNx_0_dg
            
            # ##################################
            dNx_1_dg=n_z[iSlice].reshape(-1,1,1)\
                      * SqrtCov_xy.reshape((Ns,ThisNParms,1))\
                      * SqrtCov_xy.reshape((Ns,1,ThisNParms))\
                      * GammaSlice.flat[self.IndexCube_xy_Slice].reshape((-1,1,1))*self.CellRad_1**2
            Sum_dNx_1_dg=np.sum(dNx_1_dg,axis=0)
            H[iPar:jPar,iPar:jPar]+= Sum_dNx_1_dg

            ng=n_zt[:,iSlice].reshape((-1,1))* GammaSlice.flat[self.IndexCube_xy_Slice].reshape((-1,1))
            dAx_dg_0 = ng.reshape(-1,1,1) * SqrtCov_xy.reshape((Ns,ThisNParms,1)) * SqrtCov_xy.reshape((Ns,1,ThisNParms))
            H[iPar:jPar,iPar:jPar]+= np.sum(dAx_dg_0 * (Sum_z_Ax_1_z.reshape((-1,1,1)))**(-1),axis=0)*self.CellRad_1**2

            for jSlice in range(self.NSlice):
                GammaSlice_j=GM.GammaCube[jSlice]
                j0,j1=DicoIndex[jSlice]
                ThisNParms_j=j1-j0
                SqrtCov_j=L_SqrtCov[jSlice]
                SqrtCov_xy_j=SqrtCov_j[self.IndexCube_xy_Slice,:]



                # ##################################
                Js=dAidg[:,i0:i1]
                Js_j=dAidg[:,j0:j1]
                H[i0:i1,j0:j1]+= - np.sum(Js.reshape((Ns,ThisNParms,1))*Js_j.reshape((Ns,1,ThisNParms_j)) \
                                          * (Sum_z_Ax_1_z.reshape((-1,1,1)))**(-2),axis=0)*self.CellRad_1**4
            

        
        return H

    
    def recenterNorm(self,X):
        return X
        return self.CSW.recenterNorm(X)
        
