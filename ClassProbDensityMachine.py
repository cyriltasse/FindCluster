from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import pylab
from scipy import ndimage
from DDFacet.Other import logger
log = logger.getLogger("ClassProbDensityMachine")

def test():
    import ClassCatalogMachine2
    CM=ClassCatalogMachine2.ClassCatalogMachine()
    CM.Init()
    PDM=ClassProbDensityMachine(CM)
    PDM.computePDF_All()
    

    
class ClassProbDensityMachine():
    def __init__(self,CatalogMachine,zg_Pars=[0.01,1.5,10],logM_Pars=[8.,12.,13]):
        self.CM=CatalogMachine
        self.zg=np.linspace(*zg_Pars)
        self.zg_Pars=zg_Pars
        self.logM_Pars=logM_Pars
        FactorOverSample_logM=10
        self.logM_g=np.linspace(*logM_Pars)
        self.logM_gm=(self.logM_g[1::]+self.logM_g[0:-1])/2.
        dlogM = self.logM_g[1] - self.logM_g[0]
        self.logM_g_hr=np.linspace(self.logM_g[0],self.logM_g[-1],self.logM_g.size*FactorOverSample_logM)

        m_step = FactorOverSample_logM
        zg_meas=self.CM.DicoDATA["zgrid_pz"]
        self.ind_zg=np.where((zg_meas>=zg_Pars[0])&(zg_meas<zg_Pars[1]))
        zg_meas_sel=zg_meas[self.ind_zg]

        self.LabelsArray=np.zeros((zg_meas_sel.size,self.logM_g_hr.size),np.int64)
        iLabel=0
        self.IndexArray=[]
        self.NlogM_rebin=logM_Pars[-1]-1
        self.Nz_rebin=zg_Pars[-1]-1
        
        for zBlock in range(zg_Pars[-1]-1):
            z0=self.zg[zBlock]
            z1=self.zg[zBlock+1]
            ind_z=np.where((zg_meas_sel>=z0)&(zg_meas_sel<=z1))[0]
            for MBlock in range(logM_Pars[-1]-1):
                m0=self.logM_g[MBlock]
                m1=self.logM_g[MBlock+1]
                ind_m=np.where((self.logM_g_hr>=m0)&(self.logM_g_hr<=m1))[0]
                self.LabelsArray[ind_z[0]:ind_z[-1]+1,ind_m[0]:ind_m[-1]+1]=iLabel
                self.IndexArray.append(iLabel)
                iLabel+=1

        self.IndexArray=np.array(self.IndexArray)
        
        #         print 
        # pylab.clf()
        # pylab.imshow(self.LabelsArray,interpolation="nearest",aspect="auto")
        # pylab.draw()
        # pylab.show(False)
        # stop
        
        # Index_logM=np.zeros((self.logM_g_hr.size,),dtype=np.int64)
        # ii=0
        # iDone=0
        # iPow=0
        # for i in range(Index_logM.size):
        #     Index_logM[iDone]=ii
        #     iDone+=1
        #     if iDone%FactorOverSample_logM==0:
        #         ii+=2**iPow
        #         iPow+=1
        # for iz in range(self.zg.size-1):
        #     z0=self.zg[iz]
        #     z1=self.zg[iz+1]
        #     ind=np.where()
        
    def computePDF_All(self):
        log.print("Compute rebined p(z,m)...")
        
        indkeep=np.zeros((self.CM.Cat.shape[0],),dtype=np.bool8)
        for ID in range(self.CM.Cat.shape[0]):
            #print float(ID)/self.CM.Cat.shape[0]
            self.CM.Cat.Pzm[ID][:,:]=self.computePDF_ID(ID)
            if np.max(self.CM.Cat.Pzm[ID][:,:])>0.:
                self.CM.Cat.PosteriorOK[ID]=1

        


    def compute_n_zt(self):
        log.print("Compute n_zt...")
        n_zm=self.CM.DicoDATA["DicoSelFunc"]["n_zm"]
        for ID in range(self.CM.Cat.shape[0]):
            self.CM.Cat.n_zt[ID][:]=np.sum(self.CM.Cat.Pzm[ID]*n_zm,axis=1)
    
    def computePDF_ID(self,ID):
        if self.CM.Cat.chi_best[ID]==0. or np.isnan(self.CM.Cat.chi_best[ID]): return 0.
        Median_m=self.CM.Cat.Mass_median[ID]
        #if np.isnan(Median_m) or Median_m==0.: return 0.
        
        sig0_m=abs(Median_m-self.CM.Cat.Mass_l68[ID])
        sig1_m=abs(Median_m-self.CM.Cat.Mass_u68[ID])
        if np.isnan(sig0_m) or sig0_m==0.: return 0.
        if np.isnan(sig1_m) or sig1_m==0.: return 0.
        
        Pm=np.zeros((self.logM_g_hr.size,),np.float32)
        ind_l=np.where(self.logM_g_hr<Median_m)[0]
        ind_u=np.where(self.logM_g_hr>=Median_m)[0]
        if ind_l.size>0: Pm[ind_l]=np.exp(-(Median_m-self.logM_g_hr[ind_l])**2/(2.*sig0_m**2))
        if ind_u.size>0: Pm[ind_u]=np.exp(-(Median_m-self.logM_g_hr[ind_u])**2/(2.*sig1_m**2))
        
        
        Pz=self.CM.Cat.pz[ID][self.ind_zg]
        P=Pz.reshape((-1,1))*Pm.reshape((1,-1))
        if np.max(P)==0:
            return 0.
        Pr=ndimage.mean(P, labels=self.LabelsArray, index=self.IndexArray)
        Pr=Pr.reshape((self.Nz_rebin,self.NlogM_rebin))
        Pr/=np.sum(Pr)

        # Pr.fill(0.)
        # Pr[self.Nz_rebin//2,self.NlogM_rebin//2]=1.
        
        return Pr
    
        # pylab.clf()
        # pylab.subplot(1,2,1)
        # pylab.imshow(P,aspect="auto")
        # pylab.subplot(1,2,2)
        # pylab.imshow(Pr,aspect="auto")
        # pylab.draw()
        # pylab.show(False)
        # pylab.pause(0.1)
