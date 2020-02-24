from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from astropy.cosmology import WMAP9 as cosmo
import FieldsToFiles

import ClassMassFunction
import os
from DDFacet.Other import MyPickle
from DDFacet.Other import logger
log = logger.getLogger("ClassSelectionFunction")
# # ##############################
# # Catch numpy warning
# np.seterr(all='raise')
# import warnings
# warnings.filterwarnings('error')
# #with warnings.catch_warnings():
# #    warnings.filterwarnings('error')
# # ##############################

    
    
class ClassSelectionFunction():
    def __init__(self,CM):
        self.CM=CM
        self.zg=np.linspace(*self.CM.zg_Pars)
        self.logM_g=np.linspace(*self.CM.logM_Pars)
        if "DicoSelFunc" in CM.DicoDATA.keys():
            log.print("Getting precomputed selection function...")
            self.DicoSelFunc=CM.DicoDATA["DicoSelFunc"]
            
        self.ModelMassFunc=ClassMassFunction.ClassMassFunction()

    def ComputeMassFunction(self):
        Z=self.CM.Cat.z
        logM=self.CM.Cat.Mass_best
        OmegaSr=self.CM.OmegaTotal
        Phi=np.zeros((self.zg.size-1,self.logM_g.size-1),np.float32)
        Phi1=np.zeros((self.zg.size-1,self.logM_g.size-1),np.float32)
        Selfunc=np.zeros((self.zg.size-1,self.logM_g.size-1),np.float32)
        CMF=ClassMassFunction.ClassMassFunction()
        NObsPerSr=np.zeros((self.zg.size-1,self.logM_g.size-1),np.float32)
        for izbin in range(self.zg.size-1):
            z0,z1=self.zg[izbin],self.zg[izbin+1]
            zm=(z0+z1)/2.
            dz=z1-z0
            dV_dz=cosmo.differential_comoving_volume(zm).to_value()
            V=dz*dV_dz*OmegaSr

            for iMbin in range(self.logM_g.size-1):
                logM0,logM1=self.logM_g[iMbin],self.logM_g[iMbin+1]
                C0=((Z>z0)&(Z<z1))
                C1=((logM>logM0)&(logM<logM1))
                ind=np.where(C0&C1)[0]
                Phi1[izbin,iMbin]=ind.size/V/(logM1-logM0)
                Phi[izbin,iMbin]= np.sum(self.CM.Cat.Pzm[:,izbin,iMbin])/V/(logM1-logM0)
                Phi_model=CMF.givePhiM(zm,(logM1+logM0)/2.)
                Selfunc[izbin,iMbin]=Phi[izbin,iMbin]/Phi_model
                #Selfunc[izbin,iMbin]=np.min([Phi[izbin,iMbin]/Phi_model,1.])
                Selfunc[izbin,iMbin]=np.max([0.,Selfunc[izbin,iMbin]])
                
                NObsPerSr[izbin,iMbin]=Phi_model*(logM1-logM0)*dz*dV_dz*Selfunc[izbin,iMbin]
                
            logMm=((self.logM_g[0:-1]+self.logM_g[1::])/2.)

        self.PhiMeasured=Phi
        self.PhiMeasured1=Phi1
        self.Selfunc=Selfunc

        self.DicoSelFunc={"logM_g":self.logM_g,
                          "zg":self.zg,
                          "SelFunc":Selfunc,
                          "n_zm":NObsPerSr,
                          "n_z":np.sum(NObsPerSr,axis=1)}

        
        # FileName="%s.SelFunc.Dico"%self.CM.DicoDATA["FileNames"]['PhysCatName']
        # log.print("Saving selection function as: %s"%FileName)
        # MyPickle.DicoNPToFile(self.DicoSelFunc,FileName)

    def setSelectionFunction(self,FileName=None,DicoSelFunc=None):
        if FileName is not None:
            log.print("Loading selection function: %s"%FileName)
            self.DicoSelFunc=MyPickle.FileToDicoNP(FileName)
        elif Dico is not None:
            self.DicoSelFunc=DicoSelFunc
        self.logM_g=self.DicoSelFunc["logM_g"]
        self.zg=self.DicoSelFunc["zg"]
        self.Selfunc=self.DicoSelFunc["SelFunc"]

    def giveSelFunc(self,z,logM):
        zm=(self.zg[1::]+self.zg[0:-1])/2.
        logMm=(self.logM_g[1::]+self.logM_g[0:-1])/2.
        iz=np.argmin(np.abs(z-zm))
        #ilogM=np.argmin(np.abs(logM.reshape((-1,1))-logMm.reshape((1,-1))),axis=1)
        return np.interp(logM, logMm, self.Selfunc[iz])
        
    def PlotSelectionFunction(self):
        return
        Phi=self.PhiMeasured
        Phi1=self.PhiMeasured1
        import pylab
        pylab.clf()
        for izbin in range(self.zg.size-1):
            z0,z1=self.zg[izbin],self.zg[izbin+1]
            zm=(z0+z1)/2.
            logMm=((self.logM_g[0:-1]+self.logM_g[1::])/2.)
            
            CMF=ClassMassFunction.ClassMassFunction()
            Phi_model=CMF.givePhiM(zm,logMm)
            # CMF=ClassMassFunction.ClassMassFunction(Model="Fontana06")
            # Phi_model2=CMF.givePhiM(zm,logMm)
            pylab.subplot(3,3,izbin+1)
            pylab.scatter(logMm,np.log10(Phi[izbin]))
            pylab.scatter(logMm,np.log10(Phi1[izbin]),c="red")
            pylab.plot(logMm,np.log10(Phi_model),color="black",ls="--")
            # # #pylab.plot(logMm,np.log10(Phi_model2),color="black",ls=":")
            # pylab.scatter(logMm,self.Selfunc[izbin])
            # MM=np.linspace(logMm[0],logMm[-1],100)
            # s=self.giveSelFunc(zm,MM)
            # pylab.scatter(MM,s,color="red")
            # #pylab.plot(logMm,np.log10(Phi_model2),color="black",ls=":")
            pylab.title("%.2f < Z < %.2f"%(z0,z1))
            pylab.grid()
        pylab.tight_layout()
        pylab.draw()
        pylab.show(False)
        pylab.pause(0.5)

    # def ComputeSelectionFunction(self):
    #     Phi=self.PhiMeasured
    #     for izbin in range(self.zg.size-1):
    #         z0,z1=self.zg[izbin],self.zg[izbin+1]
    #         zm=(z0+z1)/2.
    #         logMm=((self.logM_g[0:-1]+self.logM_g[1::])/2.)
            
    #         CMF=ClassMassFunction.ClassMassFunction()
    #         Phi_model=CMF.givePhiM(zm,logMm)
            
    #         np.interp(x, xp, fp
