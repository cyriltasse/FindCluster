#!/usr/bin/env python
import os
os.environ["OMP_NUM_THREADS"] = "1"
from multiprocessing import Pool
#import emcee
import numpy as np
import matplotlib.pyplot as plt
import GeneDist
import ClassSimulCatalog
import matplotlib.pyplot as pylab
import ClassMassFunction
from astropy.cosmology import WMAP9 as cosmo
from DDFacet.Other import logger
log = logger.getLogger("RunMCMC")
from DDFacet.Other.AsyncProcessPool import APP, WorkerProcessError
from DDFacet.Other import Multiprocessing
from DDFacet.Other import ModColor
from DDFacet.Other import ClassTimeIt
from DDFacet.Other.progressbar import ProgressBar
from DDFacet.Other import AsyncProcessPool
from DDFacet.Array import shared_dict
import ClassCatalogMachine
import ClassLikelihoodMachine
import ClassInitGammaCube
import ClassGammaMachine_Wave
import ClassDisplayRGB
import ClassSaveFITS

def read_options():
    desc="""Questions and suggestions: cyril.tasse@obspm.fr"""
    global options
    opt = optparse.OptionParser(usage='',version='%prog version 1.0',description=desc)

    group = optparse.OptionGroup(opt, "* SM related options", "Won't work if not specified.")
    group.add_option('--SkyModel',help='Name of the bbs type skymodel',default='')
    group.add_option('--BaseImageName',help='List of targets [no default]',default='')
    opt.add_option_group(group)

    options, arguments = opt.parse_args()
    f = open(SaveName,"wb")
    pickle.dump(options,f)

def g_z(z):
    a=4.
    g=np.zeros_like(z)
    ind=np.where((z>(1./a))&(z<a))[0]
    g[ind]=1./z[ind]
    return g

def test(ComputeInitCube=False):
    rac_deg,decc_deg=241.20678,55.59485 # cluster
    FOV=0.15
    SubField={"rac_deg":rac_deg,
              "decc_deg":decc_deg,
              "FOV":FOV,
              "CellDeg":0.001}
    MCMCMachine=ClassRunMCMC(SubField,ComputeInitCube=ComputeInitCube)
    
    MCMCMachine.runMCMC()
    
class ClassMakeDensityImage():
    def __init__(self,
                 SubField,
                 NCPU=56,
                 ComputeInitCube=True,
                 randomiseCatalog=False):
        self.NCPU=NCPU
        self.CM=ClassCatalogMachine.ClassCatalogMachine()
        self.CM.Init()
        if SubField["rac_deg"] is None:
            SubField["rac_deg"]=self.CM.rac_main*180./np.pi
            SubField["decc_deg"]=self.CM.decc_main*180./np.pi
            
        self.rac_deg,self.decc_deg=SubField["rac_deg"],SubField["decc_deg"]
        
        self.racdecc=self.rac,self.decc=self.rac_deg*np.pi/180,self.decc_deg*np.pi/180
        self.FOV=SubField["FOV"]
        self.CellDeg=SubField["CellDeg"]
        self.CellRad=self.CellDeg*np.pi/180
        
        self.NPix=int(self.FOV/self.CellDeg)
        if (self.NPix%2)!=0:
            self.NPix+=1
            
        print>>log,"Choosing NPix=%i"%self.NPix

        self.CM.cutCat(self.rac,self.decc,self.NPix,self.CellRad)
        self.zParms=self.CM.zg_Pars
        self.NSlice=self.zParms[-1]-1
        self.logMParms=self.CM.logM_Pars
        self.logM_g=np.linspace(*self.logMParms)

        self.DistMachine=GeneDist.ClassDistMachine()
        z=np.linspace(-10,10,1000)
        g=g_z(z)
        G=np.cumsum(g)
        G/=G[-1]
        self.DistMachine.setCumulDist(z,G)

        self.GM=ClassGammaMachine_Wave.ClassGammaMachine(self.CM.racdecc_main,
                                                         self.racdecc,
                                                         self.CellDeg,
                                                         self.NPix,
                                                         zParms=self.zParms)
        
        self.CIGC=ClassInitGammaCube.ClassInitGammaCube(self.CM,self.GM,ScaleKpc=[200.])
        
       
    def finaliseInit(self):
        APP.registerJobHandlers(self)
        AsyncProcessPool.init(ncpu=self.NCPU,
                              affinity=0)
        APP.startWorkers()

    def showRGB(self,zLabels=False):
        DRGB=ClassDisplayRGB.ClassDisplayRGB()
        DRGB.setRGB_FITS(*self.CM.DicoDataNames["RGBNames"])
        DRGB.setRaDec(self.rac_deg,self.decc_deg)
        DRGB.setBoxArcMin(self.NPix*self.CellDeg*60.)
        DRGB.FitsToArray()
        DRGB.Display()#Scale="linear",vmin=,vmax=30)
        if zLabels:
            import pylab
            pylab.scatter(self.CM.Cat_s.l,self.CM.Cat_s.m)
            for i in range(self.CM.Cat_s.shape[0]):
                pylab.text(self.CM.Cat_s.l[i],self.CM.Cat_s.m[i],self.CM.Cat_s.z[i],color="red")
            pylab.draw()

    def killWorkers(self):
        print>>log, "Killing workers"
        APP.terminate()
        APP.shutdown()
        Multiprocessing.cleanupShm()
        
    def InitCube(self):
        if "EffectiveOmega" not in self.CIGC.DicoCube.keys():
            self.CIGC.computeEffectiveOmega()
        self.Cube=self.CIGC.InitGammaCube()
        #np.save("Cube",self.Cube)
        
    def randomiseCatalog(self):
        self.CM.randomiseCatalog()
        self.CM.coordToShared()

    def SaveFITS(self,Name="Test.fits"):
        G=self.Cube
        GG=G.reshape((self.NSlice,self.NPix,self.NPix))
        im=ClassSaveFITS.ClassSaveFITS(Name,
                                       GG.shape,
                                       self.CellDeg,
                                       (self.rac,self.decc))#, Freqs=np.linspace(1400e6,1500e6,20))


        Gd=np.zeros_like(G)
        for ich in range(self.NSlice):
            Gd[ich]=G[ich].T[:,::-1]
        im.setdata(Gd.astype(np.float32))#,CorrT=True)
        im.ToFits()
        im.close()


        # DRGB=ClassDisplayRGB.ClassDisplayRGB()
        # DRGB.setRGB_FITS(*NamesRGB)
        # radec=[self.rac_deg,self.dec_deg]
        # DRGB.setRaDec(*radec)
        # DRGB.setBoxArcMin(self.boxDeg*60)
        # DRGB.loadFitsCube(Name,Sig=1)
        # DRGB.Display(NamePNG="%s.png"%Name,Scale="linear",vmin=5,vmax=30)



        
def test():
    
    
    
    rac_deg,decc_deg=241.20678,55.59485 # cluster
    #FOV=0.15
    FOV=0.07
    #FOV=0.5
    FOV=1.5
    SubField={"rac_deg":rac_deg,
              "decc_deg":decc_deg,
              "FOV":FOV,
              "CellDeg":0.001}
    MDI=ClassMakeDensityImage(SubField,
                              randomiseCatalog=True)
    MDI.finaliseInit()
    # MDI.showRGB()
    MDI.InitCube()

    Name="Test35"
    MDI.SaveFITS(Name="/data/tasse/%s.fits"%Name)

    for iRandom in range(10):
        MDI.randomiseCatalog()
        MDI.InitCube()
        MDI.SaveFITS(Name="/data/tasse/%s_rand%2.2i.fits"%(Name,iRandom))
    
    MDI.killWorkers()
    
def test2():
    
    
    
    rac_deg,decc_deg=None,None
    FOV=3.5
    SubField={"rac_deg":rac_deg,
              "decc_deg":decc_deg,
              "FOV":FOV,
              "CellDeg":0.001}
    MDI=ClassMakeDensityImage(SubField,
                              randomiseCatalog=True)
    MDI.finaliseInit()
    # MDI.showRGB()
    MDI.InitCube()

    Name="Test35"
    MDI.SaveFITS(Name="/data/tasse/%s.fits"%Name)

    for iRandom in range(10):
        MDI.randomiseCatalog()
        MDI.InitCube()
        MDI.SaveFITS(Name="/data/tasse/%s_rand%2.2i.fits"%(Name,iRandom))
    
    MDI.killWorkers()
    
    
if __name__=="__main__":
    test2()
