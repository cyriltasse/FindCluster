
import numpy as np
from astropy.cosmology import WMAP9 as cosmo
import FieldsToFiles
import ClassCatalogMachine2
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

def test():
    GSF=ClassSelectionFunction()
    GSF.LoadData()
    GSF.ComputeMassFunction()
    GSF.PlotSelectionFunction()
    
    
class ClassSelectionFunction():
    def __init__(self,z=[0.01,1.5,10],logMParms=[8.,12.5,20]):
        self.zg=np.linspace(*z)
        self.logM_g=np.linspace(*logMParms)
        self.ModelMassFunc=ClassMassFunction.ClassMassFunction()
        
    def LoadData(self,Show=False,
                 DicoDataNames=FieldsToFiles.DicoDataNames_EN1,
                 ForceLoad=0):
        
        # CM=ClassCatalogMachine.ClassCatalogMachine(self.rac,self.decc,
        #                                            CellDeg=self.CellDeg,
        #                                            NPix=self.NPix,
        #                                            ScaleKpc=self.ScaleKpc)

        CM=ClassCatalogMachine2.ClassCatalogMachine()
        
        # if Show:
        #     CM.showRGB()
            
        if os.path.isfile(DicoDataNames["PickleSave"]) and not ForceLoad:
            CM.PickleLoad(DicoDataNames["PickleSave"])
        else:
            CM.setMask(DicoDataNames["MaskImage"])
            CM.setPhysCatalog(DicoDataNames["PhysCat"])
            CM.setCat(DicoDataNames["PhotoCat"])
            CM.setPz(DicoDataNames["PzCat"])
            CM.PickleSave(DicoDataNames["PickleSave"])
        self.CM=CM
        
    def ComputeMassFunction(self):
        Z=self.CM.DicoDATA["z"]
        logM=self.CM.DicoDATA["Mass"]
        OmegaSr=self.CM.OmegaTotal
        Phi=np.zeros((self.zg.size-1,self.logM_g.size-1),np.float32)
        Selfunc=np.zeros((self.zg.size-1,self.logM_g.size-1),np.float32)
        CMF=ClassMassFunction.ClassMassFunction()
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
                Phi[izbin,iMbin]=ind.size/V/(logM1-logM0)
                
                Phi_model=CMF.givePhiM(zm,(logM1+logM0)/2.)
                Selfunc[izbin,iMbin]=np.min([Phi[izbin,iMbin]/Phi_model,1.])
                Selfunc[izbin,iMbin]=np.max([0.,Selfunc[izbin,iMbin]])
                
            logMm=((self.logM_g[0:-1]+self.logM_g[1::])/2.)

        self.PhiMeasured=Phi
        self.Selfunc=Selfunc

        DicoSelFunc={"logM_g":self.logM_g,
                     "zg":self.zg,
                     "SelFunc":Selfunc}
        FileName="%s.SelFunc.Dico"%self.CM.DicoDATA["FileNames"]['PhysCatName']
        print>>log,"Saving selection function as: %s"%FileName
        MyPickle.DicoNPToFile(DicoSelFunc,FileName)

    def LoadSelectionFunction(self,FileName):
        print>>log,"Loading selection function: %s"%FileName
        self.DicoSelFunc=MyPickle.FileToDicoNP(FileName)
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
        
        Phi=self.PhiMeasured
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
            # pylab.scatter(logMm,np.log10(Phi[izbin]))
            # pylab.plot(logMm,np.log10(Phi_model),color="black",ls="--")
            # #pylab.plot(logMm,np.log10(Phi_model2),color="black",ls=":")
            pylab.scatter(logMm,self.Selfunc[izbin])
            MM=np.linspace(logMm[0],logMm[-1],100)
            s=self.giveSelFunc(zm,MM)
            pylab.scatter(MM,s,color="red")
            #pylab.plot(logMm,np.log10(Phi_model2),color="black",ls=":")
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
