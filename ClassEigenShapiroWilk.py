
import numpy as np
import ClassShapiroWilk
from DDFacet.Other import logger
log = logger.getLogger("ClassEigenSW")
import ClassAndersonDarling

class ClassEigenShapiroWilk():
    def __init__(self,L_NParms,NTry=2000,NPerTest=20):
        self.L_NParms=L_NParms
        self.DicoCSW_N={}

        self.NSlice=len(self.L_NParms)
        DicoCSW={}
        DicoSlice={}
        for iSlice in range(self.NSlice):
            N=self.L_NParms[iSlice]
            NBin=N//NPerTest+1
            NBin=np.max([2,NBin])
            BinX=np.int32(np.linspace(0,N,NBin*2))
            Li0=BinX[0:-2].tolist()
            Li1=BinX[2:].tolist()

            BinX=np.int32(np.linspace(0,N,NBin))
            Li0=BinX[0:-1].tolist()
            Li1=BinX[1:].tolist()
            Ln=[]
            Li0i1=[]
            for iSub,(i0,i1) in enumerate(zip(Li0,Li1)):
                n=i1-i0
                log.print("zSlice #%i, subStat #%i, using samples of %i-size"%(iSlice,iSub,n))
                if n not in DicoCSW.keys():
                    log.print("Initialising ClassSW with n=%i"%n)
                    #CSW=ClassShapiroWilk.ClassShapiroWilk()
                    CSW=ClassAndersonDarling.ClassAndersonDarlingMachine()
                    CSW.Init(n,NTry=2000)
                    DicoCSW[n]=CSW
                Ln.append(n)
                Li0i1.append((i0,i1))
            DicoSlice[iSlice]={"ListN":Ln,
                               "List_i0i1":Li0i1}
        self.DicoSlice=DicoSlice
        self.DicoCSW=DicoCSW

    def logP_x(self,g):
        L=0
        ii=0
        for iSlice in range(self.NSlice):
            ThisNParms=self.L_NParms[iSlice]
            iPar=ii
            jPar=iPar+ThisNParms
            gg=g[iPar:jPar]
            iii=0
            for n,(i0,i1) in zip(self.DicoSlice[iSlice]["ListN"],self.DicoSlice[iSlice]["List_i0i1"]):
                ggg=gg[i0:i1]
                L+=self.DicoCSW[n].logP_x(ggg)
            ii+=ThisNParms
        return L
    
    
    def dlogPdx(self,g):
        dLdx=np.zeros((g.size,),np.float64)
        ii=0
        for iSlice in range(self.NSlice):
            ThisNParms=self.L_NParms[iSlice]
            iPar=ii
            jPar=iPar+ThisNParms
            gg=g[iPar:jPar]
            dLdxs=dLdx[iPar:jPar]
            iii=0
#            for n in self.DicoSlice[iSlice]["ListN"]:
            for n,(i0,i1) in zip(self.DicoSlice[iSlice]["ListN"],self.DicoSlice[iSlice]["List_i0i1"]):
                ggg=gg[i0:i1]
                dLdxs[i0:i1]+=self.DicoCSW[n].dlogPdx(ggg)
                iii+=n
            ii+=ThisNParms
        return dLdx

    def d2logPdx2(self,g):
        dJdx=np.zeros((g.size,),np.float64)
        ii=0
        for iSlice in range(self.NSlice):
            ThisNParms=self.L_NParms[iSlice]
            iPar=ii
            jPar=iPar+ThisNParms
            gg=g[iPar:jPar]
            dJdxs=dJdx[iPar:jPar]
            iii=0
#            for n in self.DicoSlice[iSlice]["ListN"]:
            for n,(i0,i1) in zip(self.DicoSlice[iSlice]["ListN"],self.DicoSlice[iSlice]["List_i0i1"]):
                ggg=gg[i0:i1]
                dJdxs[i0:i1]+=self.DicoCSW[n].d2logPdx2(ggg)
                iii+=n
            ii+=ThisNParms
        return dJdx



    def recenterNorm(self,g):
        L=0
        ii=0
        for iSlice in range(self.NSlice):
            ThisNParms=self.L_NParms[iSlice]
            iPar=ii
            jPar=iPar+ThisNParms
            gg=g[iPar:jPar]
            iii=0
            for n,(i0,i1) in zip(self.DicoSlice[iSlice]["ListN"],self.DicoSlice[iSlice]["List_i0i1"]):
                ggg=gg[i0:i1]
                mean=np.mean(ggg)
                sig=np.std(ggg)
                ggg[:]=(ggg[:]-mean)/sig
                iii+=n
            ii+=ThisNParms
        return g
    
    def meas_dlogP_dx(self,X,ParmId=None):
        if ParmId is None:
            ParmId=np.arange(x.size)
            
        LH=[]
        for iParm in ParmId:
            dx=np.linspace(-.001,.001,2)
            LJ=[]
            for ix in range(dx.size):
                Xix=X.copy()
                Xix[iParm]+=dx[ix]
                LJ.append(self.logP_x(Xix))
            H=(LJ[1]-LJ[0])/(dx[1]-dx[0])
            LH.append(H)

        return np.array(LH)

    def meas_d2logP_dx2(self,X,ParmId=None):
        if ParmId is None:
            ParmId=np.arange(x.size)
        LH=[]
        for iParm in ParmId:
            dx=np.linspace(-.00001,.00001,2)
            LJ=[]
            for ix in range(dx.size):
                Xix=X.copy()
                Xix[iParm]+=dx[ix]
                LJ.append(self.dlogPdx(Xix)[iParm])
            H=(LJ[1]-LJ[0])/(dx[1]-dx[0])
            LH.append(H)

