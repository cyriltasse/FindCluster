
import numpy as np
from astropy.cosmology import WMAP9 as cosmo
import FieldsToFiles
import ClassCatalogMachine2
import ClassMassFunction
import os

def test():
    GSF=GetSelectionFunction()
    GSF.LoadData()
    
class GetSelectionFunction():
    def __init__(self,z=[0.01,2.,10],logMParms=[8.,12.,10]):
        
        self.zg=np.linspace(*z)

        self.logM_g=np.linspace(*logMParms)
        
        self.ModelMassFunc=ClassMassFunction.ClassMassFunction()
        
        
        
    def LoadData(self,Show=False,DicoDataNames=FieldsToFiles.DicoDataNames_EN1,ForceLoad=True):
        
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
        Z=self.CM.DicoData["z"]
        logM=self.CM.DicoData["Mass"]
        OmegaSr=self.CM.OmegaTotal
        for izbin in range(self.zg.size-1):
            z0,z1=self.zg[izbin],self.zg[izbin+1]
            zm=(z0+z1)/2.
            dz=z1-z0
            dV_dz=cosmo.differential_comoving_volume(zm).to_value()
        
            V=dz*dV_dz*OmegaSr

            for iMbin in range(self.logM_g.size-1):
                logM0,logM1=self.logM_g[iMbin],self.logM_g[iMbin+1]
                C0=np.where((Z>z0)&(Z<z1))[0]
                C1=np.where((logM>logM0)&(logM<logM1))[0]
                ind=np.where(C0&C1)[0]
                
                
