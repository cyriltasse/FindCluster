import os
os.environ["OMP_NUM_THREADS"] = "1"
from multiprocessing import Pool
#import emcee
import numpy as np
import matplotlib.pyplot as plt
import GeneDist
import ClassSimulCatalog
import matplotlib
import matplotlib.pyplot as pylab
import ClassMassFunction
from astropy.cosmology import WMAP9 as cosmo
from DDFacet.Other import logger
log = logger.getLogger("ClassRunMultiLM")

from DDFacet.Other import ModColor
from DDFacet.Other import ClassTimeIt
from DDFacet.Other.progressbar import ProgressBar
from DDFacet.Array import shared_dict
import ClassCatalogMachine
#import ClassDiffLikelihoodMachine as ClassLikelihoodMachine
import ClassLogDiffLikelihoodMachine as ClassLikelihoodMachine
import ClassInitGammaCube
import ClassDisplayRGB

from DDFacet.Other.AsyncProcessPool import APP, WorkerProcessError
from DDFacet.Other import Multiprocessing
from DDFacet.Other import AsyncProcessPool
from DDFacet.ToolsDir import ModCoord

# from DDFacet.Other.AsyncProcessPool import APP, WorkerProcessError
# from DDFacet.Other import Multiprocessing
# from DDFacet.Other import AsyncProcessPool
import scipy.optimize
import GeneDist
import ClassPlotMachine
from DDFacet.ToolsDir.GiveEdges import GiveEdges
import ClassCovMatrix_Sim3D_2 as ClassCovMatrixMachine
import ClassRunLM_Cov
import ClassSaveFITS

# # ##############################
# # Catch numpy warning
# np.seterr(all='raise')
# import warnings
# warnings.filterwarnings('error')
# #with warnings.catch_warnings():
# #    warnings.filterwarnings('error')
# # ##############################

def test():
    CLM=ClassRunMultiLM(mainFOV=0.3)
    CLM.run()



class ClassRunMultiLM():
    def __init__(self,
                 mainFOV=0.1,
                 CellDeg=0.002,
                 NCPU=0):
        DoPlot=False
        self.NCPU=NCPU
        pylab.close("all")
        self.CellDeg=CellDeg
        
        CM=ClassCatalogMachine.ClassCatalogMachine()
        CM.Init()
        CM.Recompute_nz_nzt()
        self.CM=CM

        #241.20678,55.59485 # cluster
        FacetFOV=0.2
        NPix=int(FacetFOV/CellDeg)
        if (NPix%2)==0:
            NPix+=1
        log.print("Choosing NPix=%i"%NPix)
        self.NPixFacet=NPixFacet=NPix
        FacetFOV=NPixFacet*CellDeg
        
        self.CellRad=CellRad=CellDeg*np.pi/180

        
        Dl=FacetFOV*np.pi/180/2
        mainFOV=(mainFOV//FacetFOV+1)*FacetFOV
        self.mainFOVrad=mainFOVrad=mainFOV*np.pi/180

        NPixMain=int(self.mainFOVrad/self.CellRad)
        if (NPixMain%2)==0:
            NPixMain+=1
        self.NPixMain=NPixMain
        
        NFacet=int(mainFOVrad/Dl+1)
        l0=-mainFOVrad/2.
        lFacet_g=l0+Dl*np.arange(NFacet)
        mFacet_g=lFacet_g

        self.rac_main,self.decc_main=self.CM.racdecc_main
        self.rac_main,self.decc_main=241.20678*np.pi/180,55.59485*np.pi/180
        #self.rac_main,self.decc_main = rac_main_rad*180/np.pi,decc_main_rad*180/np.pi

        self.CoordMachine = ModCoord.ClassCoordConv(self.rac_main, self.decc_main)

        self.DicoFacets={}
        lCorner=-mainFOVrad/2.
        for iFacet in range(NFacet):
            for jFacet in range(NFacet):
                FacetID=iFacet*NFacet+jFacet
                self.DicoFacets[FacetID]={}
                l0,m0=lFacet_g[iFacet], mFacet_g[jFacet]
                ra0,dec0=self.CoordMachine.lm2radec(np.array([l0]),np.array([m0]))
                ra0,dec0=ra0[0],dec0[0]
                self.DicoFacets[FacetID]["radec"]=(ra0,dec0)
                self.DicoFacets[FacetID]["radecDeg"]=(ra0*180/np.pi,dec0*180/np.pi)
                self.DicoFacets[FacetID]["lm"]=(l0,m0)
                self.DicoFacets[FacetID]["xy"]=(int(l0/CellRad)+self.NPixMain//2,int(m0/CellRad)+self.NPixMain//2)


        
        CACM=ClassCovMatrixMachine.ClassAngularCovMat(CellDeg,NPixFacet,CM.zg_Pars)
        CACM.initCovMatrices()
        self.DicoCov=CACM.DicoCov
        logger.setSilent(["ClassEigenSW",
                          "ClassAndersonDarling",
                          "ClassRunLM",
                          "PlotMachine",
                          "ClassAndersonDarling",
                          "ClassCatalogMachine",
                          "ClassSelectionFunction"])
        self.finaliseInit()
        
    def finaliseInit(self):
        APP.registerJobHandlers(self)
        AsyncProcessPool.init(ncpu=self.NCPU,
                              affinity=0)
        APP.startWorkers()

    def run(self):

        for FacetID in sorted(list(self.DicoFacets.keys())):
            rac_deg,decc_deg=self.DicoFacets[FacetID]["radecDeg"]
            SubField={"rac_deg":rac_deg,
                      "decc_deg":decc_deg,
                      "NPix":self.NPixFacet,
                      "CellDeg":self.CellDeg}
            APP.runJob("_giveGammaFacet:%i"%(FacetID), 
                       self._giveGammaFacet,
                       args=(SubField,FacetID))#,serial=True)
            
        results=APP.awaitJobResults("_giveGammaFacet:*", progress="Compute Gamma")
        

        for res in results:
            FacetID,g,MedianCube,SigmaCube=res
            self.DicoFacets[FacetID]["MedianCube"]=MedianCube
            self.DicoFacets[FacetID]["Sigma"]=SigmaCube

        Im=np.zeros((self.CM.NSlice,self.NPixMain,self.NPixMain),np.float32)
        ImSum=np.zeros((self.CM.NSlice,self.NPixMain,self.NPixMain),np.float32)
        xx,yy=np.mgrid[-3.:3:1j*self.NPixFacet,-3.:3:1j*self.NPixFacet]
        #WFacet=np.ones((self.CM.NSlice,self.NPixFacet,self.NPixFacet),np.float32)
        WFacet=np.exp(-(xx**2+yy**2)/2.)
        WFacet=WFacet.reshape((1,self.NPixFacet,self.NPixFacet))
        
        for FacetID in sorted(list(self.DicoFacets.keys())):
            x,y=self.DicoFacets[FacetID]["xy"]
            N0=self.NPixFacet
            N1=self.NPixMain
            Aedge,Bedge=GiveEdges(x,y,N1,
                                  N0//2,N0//2,N0)
            x0d,x1d,y0d,y1d=Bedge
            x0,x1,y0,y1=Aedge
            
            MedianCube=self.DicoFacets[FacetID]["MedianCube"]

            #Im=np.zeros((self.CM.NSlice,self.NPixMain,self.NPixMain),np.float32)
            Im[:,x0:x1,y0:y1]+=(MedianCube*WFacet)[:,x0d:x1d,y0d:y1d]
            ImSum[:,x0:x1,y0:y1]+=WFacet[:,x0d:x1d,y0d:y1d]

        pylab.clf()
        pylab.imshow((Im/ImSum)[2],interpolation="nearest",vmin=-1,vmax=1)
        pylab.draw()
        pylab.show(block=False)
        pylab.pause(0.5)
        
        self.MedianCube=Im/ImSum
        self.MedianCube[np.isnan(self.MedianCube)]=0.
        self.SaveFITS()
            
    def SaveFITS(self,Name="Test.fits"):

        im=ClassSaveFITS.ClassSaveFITS(Name,
                                       self.MedianCube.shape,
                                       self.CellDeg,
                                       (self.rac_main,self.decc_main))#, Freqs=np.linspace(1400e6,1500e6,20))


        Gd=np.zeros_like(self.MedianCube)
        for ich in range(self.CM.NSlice):
            Gd[ich]=self.MedianCube[ich].T[:,::-1]
        im.setdata(Gd.astype(np.float32))#,CorrT=True)
        im.ToFits()
        im.close()
    
            
    def _giveGammaFacet(self,SubField,FacetID):
        matplotlib.use('Agg')
        CLM=ClassRunLM_Cov.ClassRunLM_Cov(SubField,
                                          self.CM,
                                          self.DicoCov,
                                          DoPlot=False,
                                          PlotID=FacetID)
        
        
        g,MedianCube,SigmaCube=CLM.runLM()
        return FacetID,g,MedianCube,SigmaCube
        #np.save("gEst.npy",g)
    
