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
            
            CAD=ClassAndersonDarlingMachine()
            CAD.generatePA2(100,NTry=10000)
            for iSlice in range(self.NSlice):
                self.LCAD.append(CAD)
            # for iSlice in range(self.NSlice):
            #     CAD=ClassAndersonDarlingMachine()
            #     CAD.generatePA2(GM.L_NParms[iSlice],NTry=2000)
            #     self.LCAD.append(CAD)
                
    def measure_dLdg(self,g0,DoPlot=0):
        g=g0.copy()
        L0=self.L(g,DoPlot=DoPlot)

        dL_dg0=self.dLdg(g)
        NN=g.size
        dg=.001
        dL_dg1=np.zeros((NN,),np.float64)
        Parm_id=np.arange(NN)
        
        for i in range(NN):
            g1=g.copy()
            g1[Parm_id[i]]+=dg
            L1=self.L(g1,DoPlot=DoPlot)
            dL_dg1[i]=(L0-L1)/((g-g1)[Parm_id[i]])
            if (L0-L1)==0:
                print("  0 diff for iParm = %i"%i)

        pylab.figure("Jacob")
        pylab.clf()
        # pylab.plot(dL_dg0[Parm_id]/dL_dg1)
        pylab.plot(dL_dg0[Parm_id])
        pylab.plot(dL_dg1)
        pylab.draw()
        pylab.show(block=False)
        pylab.pause(0.1)


    def L(self,g0,DoPlot=0):
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
        # SumNx_0=np.sum(TypeSum((n_z.reshape(-1,1,1)*self.CellRad_0**2)*GM.GammaCube))
        SumNx_0=np.sum(TypeSum(n_z.reshape(-1,1,1)*GM.GammaCube).flat[self.IndexCube_Mask])*self.CellRad_0**2/self.fNormLog
        # SumNx_0=np.sum([TypeSum(n_z[i]*GM.GammaCube[i]).flat[self.IndexCube_Mask_Slice] for i in range(self.NSlice)])*self.CellRad_0**2
        #print("nz", (n_z*self.CellRad_0**2), SumNx_0)
        #print(GM.GammaCube)
        
        Nx_1=np.zeros((Ns,),np.float32)
        # Ax_1=np.zeros((Ns,),np.float32)
        L_Ax_1_z=[]
        for iSlice in range(self.NSlice):
            Gamma_i=GM.GammaCube[iSlice].flat[self.IndexCube_xy_Slice]
            Nx_1[:]+=n_z[iSlice]*Gamma_i*self.CellRad_1**2
            Ax_1_z=n_zt[:,iSlice]*Gamma_i*self.CellRad_1**2
            #Ax_1[:]+=Ax_1_z
            L_Ax_1_z.append(Ax_1_z)
        SumNx_1=np.sum(TypeSum(Nx_1))
        self.Ax_1_z=np.array(L_Ax_1_z)
        self.Ax_1=np.sum(TypeSum(self.Ax_1_z),axis=0)
        
        SumAx_1=np.sum(self.funcNormLog(TypeSum(self.Ax_1)))
        #print(SumNx_0, SumNx_1, SumAx_1)
        L=-SumNx_0 + SumNx_1 + SumAx_1
        if self.MAP:
            for CAD in self.LCAD:
                L+=CAD.logP_x(g.flatten())
            # k=g.size
            # gTg=np.dot(g.T,g).flat[0]+1e-10
            # L+= - (1/2.)*gTg + (k/2-1)*np.log(gTg)
            # #L+= - (1/2.)*gTg
            
        #L=  (k/2-1)*np.log(gTg)
        # L=-SumNx_0 + SumAx_1# - (1/2.)*np.dot(g.T,g).flat[0]
        #L=-SumNx_0# - (1/2.)*np.dot(g.T,g).flat[0]
        #L= + SumNx_1# - (1/2.)*np.dot(g.T,g).flat[0]
        #L= SumAx_1
        self.gCurrentL=g0.copy()
        return L

    def dLdg(self,g0):

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
        J=np.zeros((NParms,),np.float32)
        for iSlice in range(self.NSlice):
            ThisNParms=L_NParms[iSlice]
            iPar=ii
            jPar=iPar+ThisNParms
            
            GammaSlice=GM.GammaCube[iSlice]
            SqrtCov=L_SqrtCov[iSlice]
            
            dNx_0_dg=n_z[iSlice]*SqrtCov[:,:]*GammaSlice.reshape((-1,1))*self.CellRad_0**2
            Sum_dNx_0_dg=np.sum(dNx_0_dg[self.IndexCube_Mask_Slice,:],axis=0)#*np.log(10)
            #Sum_dNx_0_dg=np.sum(dNx_0_dg,axis=0)*np.log(10)
            
            SqrtCov_xy=SqrtCov[self.IndexCube_xy_Slice,:]
            dNx_1_dg=n_z[iSlice]*SqrtCov_xy[:,:]*GammaSlice.flat[self.IndexCube_xy_Slice].reshape((-1,1))*self.CellRad_1**2
            Sum_dNx_1_dg=np.sum(dNx_1_dg,axis=0)
            
            dAx_dg_0 = n_zt[:,iSlice].reshape((-1,1))*SqrtCov_xy[:,:]*GammaSlice.flat[self.IndexCube_xy_Slice].reshape((-1,1))#*np.log(10)
            dAx_dg_1 = Sum_z_Ax_1_z
            dAx_dg   = dAx_dg_0/dAx_dg_1.reshape((-1,1))*self.CellRad_1**2
            #print("A",dAx_dg_0.flat[0],dAx_dg_1.flat[0])
            Sum_dAx_dg=np.sum(dAx_dg,axis=0)
            #print(Sum_dNx_0_dg.shape,Sum_dNx_1_dg.shape,Sum_dAx_dg.shape)
            J[iPar:jPar]= -Sum_dNx_0_dg + Sum_dNx_1_dg + Sum_dAx_dg# - g0.flat[iPar:jPar] 
            #J[iPar:jPar]=  + Sum_dNx_1_dg
            #J[iPar:jPar]= -Sum_dNx_0_dg

            #J[iPar:jPar]= g0.flat[iPar:jPar]
            #J[iPar:jPar]=  Sum_dAx_dg# - g0.flat[iPar:jPar]
            #J[iPar:jPar]= -Sum_dNx_0_dg + Sum_dNx_1_dg + Sum_dAx_dg
            #J[iPar:jPar]=  + Sum_dAx_dg
            #J[iPar:jPar]= -Sum_dNx_0_dg
            #print(iSlice,np.abs(Sum_dNx_0_dg).max(),np.abs(Sum_dNx_1_dg).max(),np.abs(Sum_dAx_dg).max())

            if self.MAP:
                J[iPar:jPar]+=self.LCAD[iSlice].dlogPdx(g[iPar:jPar].flatten())
            ii+=ThisNParms
            
        # if self.MAP:
        #     k=g0.size
        #     gTg=np.sum(g0**2)+1e-10
        #     J[:]+= -g0.flat[:] + 2*(k/2-1)*g0.flat[:]/gTg
        #     #J[:]+= -g0.flat[:]# + 2*(k/2-1)*g0.flat[:]/gTg
        # #J[:]+=  2*(k/2-1)*g0.flat[:]/gTg
        return J

    # #############################################

    def measure_dJdg(self,g0,DoPlot=0):
        g=g0.copy()
        dJdg0=self.dJdg(g)

        NN=g.size
        dg=1e-2
        dJdg1=np.zeros((NN,),np.float64)
        Parm_id=np.arange(NN)

        pylab.figure("Hessian")
        pylab.clf()
        pylab.plot(dJdg0[Parm_id])
        pylab.draw()
        pylab.show(block=False)
        pylab.pause(0.1)
        
        dLdg_0=self.dLdg(g)
        for i in range(NN):
            print("%i/%i"%(i,NN))
            g1=g.copy()
            g1[Parm_id[i]]+=dg
            dLdg_1=self.dLdg(g1)
            print(dLdg_0[Parm_id[i]],dLdg_1[Parm_id[i]])
            dJdg1[i]=(dLdg_0[Parm_id[i]]-dLdg_1[Parm_id[i]]) / ((g-g1)[Parm_id[i]])

        # pylab.plot(dL_dg0[Parm_id]/dL_dg1)
        pylab.plot(dJdg1)
        pylab.draw()
        pylab.show(block=False)
        pylab.pause(0.1)

        
                
    def dJdg(self,g0):

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
        J=np.zeros((NParms,),np.float32)
        for iSlice in range(self.NSlice):
            ThisNParms=L_NParms[iSlice]
            iPar=ii
            jPar=iPar+ThisNParms
            
            GammaSlice=GM.GammaCube[iSlice]
            SqrtCov=L_SqrtCov[iSlice]
            
            dNx_0_dg=n_z[iSlice] * (SqrtCov[:,:])**2 * GammaSlice.reshape((-1,1))*self.CellRad_0**2*self.fNormLog
            Sum_dNx_0_dg=np.sum(dNx_0_dg[self.IndexCube_Mask_Slice,:],axis=0)#*np.log(10)
            #Sum_dNx_0_dg=np.sum(dNx_0_dg,axis=0)*np.log(10)
            
            SqrtCov_xy=SqrtCov[self.IndexCube_xy_Slice,:]
            dNx_1_dg=n_z[iSlice]*(SqrtCov_xy[:,:])**2 * GammaSlice.flat[self.IndexCube_xy_Slice].reshape((-1,1))*self.CellRad_1**2
            Sum_dNx_1_dg=np.sum(dNx_1_dg,axis=0)
            
            dAx_dg_0 = n_zt[:,iSlice].reshape((-1,1))* (SqrtCov_xy[:,:]) * GammaSlice.flat[self.IndexCube_xy_Slice].reshape((-1,1))#*np.log(10)
            dAx_dg_1 = Sum_z_Ax_1_z
            
            dAx_dg_A   = SqrtCov_xy[:,:]*dAx_dg_0 / dAx_dg_1.reshape((-1,1)) * self.CellRad_1**2

            dAx_dg_B   = - ( (dAx_dg_0 * dAx_dg_0) * self.CellRad_1**4 / (dAx_dg_1.reshape((-1,1)))**2 )
            
            dAx_dg = dAx_dg_A + dAx_dg_B

            Sum_dAx_dg=np.sum(dAx_dg,axis=0)
            #print(Sum_dNx_0_dg.shape,Sum_dNx_1_dg.shape,Sum_dAx_dg.shape)
            J[iPar:jPar]= -Sum_dNx_0_dg + Sum_dNx_1_dg + Sum_dAx_dg#g0.flat[iPar:jPar]
            #J[iPar:jPar]= 0
            #print(iSlice,np.abs(Sum_dNx_0_dg).max(),np.abs(Sum_dNx_1_dg).max(),np.abs(Sum_dAx_dg).max())
            
            if self.MAP:
                J[iPar:jPar]+=np.abs(self.LCAD[iSlice].d2logPdx2(g[iPar:jPar].flatten()))
            ii+=ThisNParms
            
        # k=g0.size
        # gTg=np.sum(g0**2)+1e-10
        # #J[:]+= -g0.flat[:] + 2*(k/2-1)*g0.flat[:]/gTg
        #     #J[:]+=self.CAD.d2logPdx2(g.flatten())
        # # if self.MAP:
        # #     J[:]+= -1 + 2*(k/2-1)*(1./gTg-2*g.flat[:]**2/(gTg)**2)
        # #     #J[:]+= -1 # + 2*(k/2-1)*(1./gTg-2*g.flat[:]**2/(gTg)**2)
        
        return J
