#!/usr/bin/env python
import numpy as np
import pylab
import tables
import astropy.io.fits as pyfits
from astropy.cosmology import WMAP9 as cosmo
import tables
import matplotlib.pyplot as pylab
from astropy.wcs import WCS
from astropy.io import fits as fitsio
from pyrap.images import image as CasaImage
from DDFacet.Other import logger
log = logger.getLogger("ClassCatalogMachine")
from DDFacet.Other import ModColor
from DDFacet.Other.progressbar import ProgressBar
#from DDFacet.Other import AsyncProcessPool
import ClassDisplayRGB
import ClassSaveFITS
from DDFacet.ToolsDir import ModCoord
import RecArrayOps
from DDFacet.Other import MyPickle
import os
# # ##############################
# # Catch numpy warning
# np.seterr(all='raise')
# import warnings
# warnings.filterwarnings('error')
# #with warnings.catch_warnings():
# #    warnings.filterwarnings('error')
# # ##############################
import FieldsToFiles
import ClassProbDensityMachine
import ClassSelectionFunction
from DDFacet.Array import shared_dict

def AngDist(ra0,dec0,ra1,dec1):
    AC=np.arccos
    C=np.cos
    S=np.sin
    D=S(dec0)*S(dec1)+C(dec0)*C(dec1)*C(ra0-ra1)
    if type(D).__name__=="ndarray":
        D[D>1.]=1.
        D[D<-1.]=-1.
    else:
        if D>1.: D=1.
        if D<-1.: D=-1.
    return AC(D)

class ClassCatalogMachine():
    def __init__(self,zg_Pars=[0.1,2.5,25],logM_Pars=[8.,12.,13]):
        self.MaskFits=None
        self.DicoDATA={}
        self.PhysCat=None
        self.OmegaTotal=None
        self.zg_Pars=zg_Pars
        self.logM_Pars=logM_Pars
        self.DicoDATA["logM_Pars"]=self.logM_Pars
        self.DicoDATA["zg_Pars"]=self.zg_Pars
        self.DicoDATA_Shared=None

    def Init(self,Show=False,FieldName="EN1",ForceLoad=False):
        if FieldName=="EN1":
            self.DicoDataNames=DicoDataNames=FieldsToFiles.DicoDataNames_EN1
            
        if os.path.isfile(DicoDataNames["PickleSave"]) and not ForceLoad:
            self.PickleLoad(DicoDataNames["PickleSave"])
        else:
            self.setMask(DicoDataNames["MaskImage"])
            self.setPhysCatalog(DicoDataNames["PhysCat"])
            self.setPz(DicoDataNames["PzCat"])
            self.setCat(DicoDataNames["PhotoCat"])
            self.ComputeLM()
            self.ComputePzm()
            N0=self.Cat.shape[0]
            self.Cat=self.Cat[self.Cat.PosteriorOK==1]
            N1=self.Cat.shape[0]
            print>>log,"  have kept %.4f%% of objects (others have bad fit?)"%(100*float(N1)/N0)
            self.ComputeSelFunc()
            self.PDM.compute_n_zt()
            
            self.PickleSave(DicoDataNames["PickleSave"])
        self.RA_orig,self.DEC_orig=self.Cat.RA.copy(),self.Cat.DEC.copy()
        self.coordToShared()
        
    def ComputeLM(self):
        print>>log,"Compute (ra,dec)->(l,m)..."
        ra,dec=self.Cat.RA,self.Cat.DEC
        if type(ra) is not np.ndarray:
            ra=np.array([ra])
            dec=np.array([dec])
        l,m=self.CoordMachine.radec2lm(ra,dec)
        self.Cat.l[:]=l
        self.Cat.m[:]=m

        # x=np.int32(np.around(l/self.CellRad))+self.NPix//2
        # y=np.int32(np.around(m/self.CellRad))+self.NPix//2
        # x,y=self.give_xy(ra,dec)

    def randomiseCatalog(self):
        RA,DEC=self.RA_orig,self.DEC_orig
        ra0,ra1=RA.min(),RA.max()
        dec0,dec1=DEC.min(),DEC.max()
        ram=RA
        decm=DEC
        ras=np.array([],np.float32)
        decs=np.array([],np.float32)
        print>>log,"Generating randomised catalog..."
        while True:
            ra=ra0+np.random.rand(ram.size)*(ra1-ra0)
            dec=dec0+np.random.rand(decm.size)*(dec1-dec0)
            #print ram.size
            FLAGMASK=np.bool8(np.array([self.GiveMaskFlag(ra[iS],dec[iS]) for iS in range(ra.size)]))
            #print "f"
            if FLAGMASK.any():
                ras=np.concatenate([ras,ra[FLAGMASK]])
                decs=np.concatenate([decs,dec[FLAGMASK]])
            if (FLAGMASK==0).any():
                ram=ra[FLAGMASK==0]
                decm=dec[FLAGMASK==0]
            else:
                break
            
            
        self.Cat.RA[:]=ras.flat[:]
        self.Cat.DEC[:]=decs.flat[:]

        self.ComputeLM()
        
        print>>log,"  done..."

    def coordToShared(self):
        print>>log, "Putting coordinates in shared array..."
        if not self.DicoDATA_Shared:
            self.DicoDATA_Shared = shared_dict.create("DATA_Shared")
        self.DicoDATA_Shared["RA"]=self.Cat.RA[:]
        self.DicoDATA_Shared["DEC"]=self.Cat.DEC[:]
        self.DicoDATA_Shared["l"]=self.Cat.l[:]
        self.DicoDATA_Shared["m"]=self.Cat.m[:]
    
        
    def ComputeSelFunc(self):
        print>>log,"Computig selection function..."
        CSF=ClassSelectionFunction.ClassSelectionFunction(self)
        CSF.ComputeMassFunction()
        CSF.PlotSelectionFunction()
        self.DicoDATA["DicoSelFunc"]=CSF.DicoSelFunc


    def ComputePzm(self):
        self.PDM=ClassProbDensityMachine.ClassProbDensityMachine(self,zg_Pars=self.zg_Pars,logM_Pars=self.logM_Pars)
        return self.PDM.computePDF_All()

        
    def setPz(self,PzFile):
        print>>log,"Opening p-z hdf5 file: %s"%PzFile
        H=tables.open_file(PzFile)
        self.DicoDATA["zgrid_pz"]=H.root.zgrid[:].copy()
        self.DicoDATA["pz"]=H.root.Pz[:].copy()
        H.close()
        
    def setPhysCatalog(self,CatName):
        self.PhysCatName=CatName
        print>>log,"Opening mass/sfr catalog fits file: %s"%CatName
        self.PhysCat=fitsio.open(CatName)[1].data
        self.PhysCat=self.PhysCat.view(np.recarray)


        
        

    def setCat(self,CatName):
        print>>log,"Opening catalog fits file: %s"%CatName
        self.PhotoCatName=CatName
        self.PhotoCat=pyfits.open(CatName)[1].data
        self.PhotoCat=self.PhotoCat.view(np.recarray)

        if self.PhysCat is not  None:
            print>>log,"  Create augmented photocat ..."
            dID=self.PhotoCat.id[1::]-self.PhotoCat.id[0:-1]
            if np.max(dID)!=1: stop
            FIELDS=self.PhotoCat.dtype.descr+self.PhysCat.dtype.descr
            if "pz" in self.DicoDATA.keys():
                nr,nz=self.DicoDATA["pz"].shape
                FIELDS+=[("pz",self.DicoDATA["pz"].dtype,(nz,))]
                FIELDS+=[("Pzm",np.float32,(self.zg_Pars[-1]-1,self.logM_Pars[-1]-1))]
                FIELDS+=[("n_zt",np.float32,(self.zg_Pars[-1]-1,))]
            FIELDS+=[("l",np.float32),("m",np.float32)]
            FIELDS+=[("xCube",np.int16),("yCube",np.int16)]
            FIELDS+=[("PosteriorOK",np.bool)]
            PhotoCat=np.zeros((self.PhotoCat.shape[0],),dtype=FIELDS)
            print>>log,"  Copy photo fields ..."
            for field in self.PhotoCat.dtype.fields.keys():
                PhotoCat[field][:]=self.PhotoCat[field][:]
            ID=self.PhysCat["ID"][:]
            print>>log,"  Copy phys fields ..."
            for field in self.PhysCat.dtype.fields.keys():
                PhotoCat[field][ID]=self.PhysCat[field][:]

            print>>log,"  Copy p(z) ..."
            PhotoCat["pz"][:,:]=self.DicoDATA["pz"][:,:]
            del(self.DicoDATA["pz"])
            self.Cat=PhotoCat
            self.Cat=self.Cat.view(np.recarray)
            
        # if self.PhysCat is not  None:
        #     print>>log,"  Append fields..."
        #     dID=self.PhotoCat.id[1::]-self.PhotoCat.id[0:-1]
        #     if np.max(dID)!=1: stop
        #     self.PhotoCat=RecArrayOps.AppendField(self.PhotoCat,("IDPhys",np.int32))
        #     self.PhotoCat.IDPhys=-1.
        #     self.PhotoCat.IDPhys[self.PhysCat.ID]=np.arange(self.PhysCat.shape[0])
            
        
        # if self.PhysCat is not  None:
        #     print>>log,"  Append fields..."
        #     self.Cat=RecArrayOps.AppendField(self.Cat,("Mass",np.float32))
        #     self.Cat=RecArrayOps.AppendField(self.Cat,("SFR",np.float32))
        #     self.Cat=RecArrayOps.AppendField(self.Cat,("z",np.float32))
        #     self.Cat.Mass.fill(-1)
        #     self.Cat.SFR.fill(-1)
        #     self.Cat.z.fill(-1)
        #     dID=self.Cat.id[1::]-self.Cat.id[0:-1]
        #     if np.max(dID)!=1: stop
        #     print>>log,"  Append physical information to photometric catalog..."
        #     self.Cat.Mass[self.PhysCat.ID]=self.PhysCat.Mass_median[:]  
        #     self.Cat.SFR[self.PhysCat.ID]=self.PhysCat.SFR_best[:]
        #     self.Cat.z[self.PhysCat.ID]=self.PhysCat.z[:]

        
          
        #self.CatRange=np.arange(self.Cat.FLAG_CLEAN.size)
            
        print>>log,"Remove spurious objects..."
        ind=np.where((self.Cat.FLAG_CLEAN == 1)&
                     (self.Cat.i_fluxerr > 0)&
                     (self.Cat.K_flux > 0)&
                     (self.Cat.FLAG_OVERLAP==7)&
                     (self.Cat.ch2_swire_fluxerr > 0))[0]
        
        ind=np.where((self.Cat.FLAG_CLEAN == 1)&
                     (self.Cat.i_fluxerr > 0)&
                     (self.Cat.K_flux > 0)&
                     (np.logical_not(np.isnan(self.Cat.Mass_best)))&
                     #(np.logical_not(np.isnan(self.Cat.Mass)))&
                     (self.Cat.FLAG_OVERLAP==7)&
                     (self.Cat.ch2_swire_fluxerr > 0))[0]
        
        # ind=np.where((self.Cat.FLAG_CLEAN == 1)&
        #              (self.Cat.i_fluxerr > 0)&
        #              #(self.Cat.K_flux > 0)&
        #              (self.Cat.ch2_swire_fluxerr > 0))[0]
        
        self.Cat=self.Cat[ind]
        #self.CatRange=self.CatRange[ind]
        # K=self.Cat.K_flux-self.Cat.K_flux.min()+1.
        # pylab.hist(np.log10(K),bins=100)
        # pylab.show()
        self.indFLAG=ind

        self.Cat.RA*=np.pi/180
        self.Cat.DEC*=np.pi/180

        RA=self.Cat.RA
        DEC=self.Cat.DEC

        # MASS.fill(1.)
        
        if self.MaskFits:
            print>>log, "Flagging in-mask sources..."
            FLAGMASK=np.bool8(np.array([self.GiveMaskFlag(RA[iS],DEC[iS]) for iS in range(self.Cat.RA.size)]))
            #FLAGMASK.fill(1)
            self.Cat=self.Cat[FLAGMASK]
            #self.CatRange=self.CatRange[FLAGMASK]
            print>>log, "  done ..."
            
        # ind=np.where(self.Cat.IDPhys!=-1)[0]
        # self.Cat=self.Cat[ind]
        # #self.CatRange=self.CatRange[ind]
        # self.DicoDATA["Cat"]=self.Cat
        # #self.DicoDATA["CatRange"]=self.CatRange
        # self.DicoDATA["PhysCat"]=self.PhysCat

        self.DicoDATA["Cat"]=self.Cat
        
        # self.DicoDATA["ID"]=np.int32(self.Cat.id[:])
        # self.DicoDATA["RA"]=self.Cat.RA[:]
        # self.DicoDATA["DEC"]=self.Cat.DEC[:]
        # self.DicoDATA["K"]=self.Cat.K_flux[:]
        # self.DicoDATA["IDPhys"]=self.Cat.K_flux[:]
        # #self.DicoDATA["Mass"]=self.Cat.Mass[:]
        # #self.DicoDATA["SFR"]=self.Cat.SFR[:]
        # #self.DicoDATA["z"]=self.Cat.z[:]
        # self.RA_orig,self.DEC_orig=self.DicoDATA["RA"].copy(),self.DicoDATA["DEC"].copy()

        
    def PickleSave(self,FileName):
        print>>log, "Saving catalog as: %s"%FileName
        FileNames={"MaskFitsName":self.MaskFitsName,
                  "PhotoCatName":self.PhotoCatName,
                  "PhysCatName":self.PhysCatName}
        self.DicoDATA["Cat"]=self.Cat
        self.DicoDATA["FileNames"]=FileNames
        self.DicoDATA["OmegaTotal"]=self.OmegaTotal
        MyPickle.DicoNPToFile(self.DicoDATA,FileName)
        
    def PickleLoad(self,FileName):
        print>>log, "Loading catalog from: %s"%FileName
        self.DicoDATA=MyPickle.FileToDicoNP(FileName)
        self.Cat=self.DicoDATA["Cat"].view(np.recarray)
        #self.CatRange=self.DicoDATA["CatRange"]
        #self.PhysCat=self.DicoDATA["PhysCat"].view(np.recarray)
        self.OmegaTotal=self.DicoDATA["OmegaTotal"]
        self.setMask(self.DicoDATA["FileNames"]["MaskFitsName"])
        
    def RaDecToMaskPix(self,ra,dec):
        if abs(ra)>2*np.pi: stop
        if abs(dec)>2*np.pi: stop
        f,p=self.fp
        _,_,xc,yc=self.MaskCasaImage.topixel([f,p,dec,ra])
        xc,yc=int(xc),int(yc)
        return xc,yc
        
    def giveEffectiveOmega(self,ra,dec,WtRadiusRad):
        xc,yc=self.RaDecToMaskPix(ra,dec)
        WtRDeg=WtRadiusRad*180/np.pi
        WtRpix3Sig=int(3*WtRDeg/self.CDELT)

        x0,x1=xc-WtRpix3Sig,xc+WtRpix3Sig+1
        y0,y1=yc-WtRpix3Sig,yc+WtRpix3Sig+1
        ThisMask=np.bool8(self.MaskArray[0,0,x0:x1,y0:y1]).copy()
        nx,ny=ThisMask.shape
        CDeltRad=self.CDELT*np.pi/180
        dx,dy=np.mgrid[-3*WtRadiusRad:3*WtRadiusRad:1j*nx,-3*WtRadiusRad:3*WtRadiusRad:1j*ny]
        r=np.sqrt(dx**2+dy**2)
        wt=np.exp(-r**2/(2.*WtRadiusRad**2))
        wt[ThisMask]=0
        
        wt=wt[r<3.*WtRadiusRad].flatten()
        
        
        return np.sum(wt)*CDeltRad**2


        
    def giveFracMasked(self,ra,dec,R):
        xc,yc=self.RaDecToMaskPix(ra,dec)
        Rpix=int(R/self.CDELT)

        
        # x0,x1=xc-Rpix,xc+Rpix+1
        # y0,y1=yc-Rpix,yc+Rpix+1
        
        # _,_,nx,ny=self.MaskArray.shape
        # x0=np.max([0,x0])
        # y0=np.max([0,y0])
        # x1=np.min([nx,x0])
        # y1=np.min([ny,y0])
        # print "============="
        # print x0,x1,nx
        # print y0,y1,ny

        
        
        ThisMask=(self.MaskArray[0,0,xc-Rpix:xc+Rpix+1,yc-Rpix:yc+Rpix+1]).copy()
        x,y=np.mgrid[-Rpix:Rpix+1,-Rpix:Rpix+1]
        r=np.sqrt(x**2+y**2)
        ThisMask[r>Rpix]=-1
        UsedArea_pix=(np.where(ThisMask==0)[0]).size
        Area_pix=(np.where(ThisMask!=-1)[0]).size
        if Area_pix!=0:
            frac=UsedArea_pix/float(Area_pix)
        else:
            frac=1.
        
        if frac==0: frac=-1.
        if frac<0.5: frac=-1
        # pylab.clf()
        # pylab.imshow(ThisMask,interpolation="nearest")
        # pylab.draw()
        # pylab.show(False)
        # pylab.pause(0.1)
        # #stop
        # print frac
        return frac

    def GiveMaskFlag(self,ra,dec):
        xc,yc=self.RaDecToMaskPix(ra,dec)
        FLAG=self.MaskArray[0,0,xc,yc]
        return 1-FLAG
    
    def setMask(self,MaskImage):
        print>>log,"Opening mask image: %s"%MaskImage
        self.MaskFitsName=MaskImage
        self.MaskFits=pyfits.open(MaskImage)[0]
        self.MaskArray=self.MaskFits.data
        self.MaskCasaImage=CasaImage(MaskImage)
        f,p,_,_=self.MaskCasaImage.toworld([0,0,0,0])
        self.fp=f,p
        self.CDELT=abs(self.MaskFits.header["CDELT1"])
        rac=self.MaskFits.header["CRVAL1"]*np.pi/180
        decc=self.MaskFits.header["CRVAL2"]*np.pi/180
        self.racdecc_main=rac,decc
        self.rac_main,self.decc_main=rac,decc
        
        self.CoordMachine = ModCoord.ClassCoordConv(rac, decc)
        if self.OmegaTotal is None:
            NPixZero=self.MaskArray.size-np.count_nonzero(self.MaskArray)
            self.OmegaTotal=NPixZero*(self.CDELT*np.pi/180)**2
            
    def cutCat(self,rac,decc,NPix,CellRad):
        print>>log,"Selection objects in window..."
        lc,mc=self.CoordMachine.radec2lm(rac,decc)
        # r=((NPix//2)+0.5)*CellRad
        # l0,l1=lc-r,lc+r
        # m0,m1=mc-r,mc+r
        #print>>log, (lc,mc)

        N0=(NPix)//2
        N1=NPix-1-N0
        #print>>log,(N0,N1)
        lg,mg=np.mgrid[-N0*CellRad+lc:N1*CellRad+lc:1j*NPix,-N0*CellRad+mc:N1*CellRad+mc:1j*NPix]
        #print>>log,(np.mgrid[-N0:N1:1j*NPix])
        l0=lg.min()-0.5*CellRad
        l1=lg.max()+0.5*CellRad
        m0=mg.min()-0.5*CellRad
        m1=mg.max()+0.5*CellRad

        Cl=((self.Cat.l>l0)&(self.Cat.l<l1))
        Cm=((self.Cat.m>m0)&(self.Cat.m<m1))
        CP=(self.Cat.PosteriorOK==1)
        ind=np.where(Cl&Cm&CP)[0]

        NN0=self.Cat.shape[0]
        self.Cat_s=self.Cat[ind]
        self.Cat_s=self.Cat_s.view(np.recarray)
        self.Cat_s.xCube=np.int32(np.around((self.Cat_s.l-lc)/CellRad))+NPix//2
        self.Cat_s.yCube=np.int32(np.around((self.Cat_s.m-mc)/CellRad))+NPix//2
        
        #print>>log,(self.Cat_s.xCube.max(),self.Cat_s.yCube.max())
        #print>>log,(self.Cat_s.xCube.min(),self.Cat_s.yCube.min())

        # Cx=((self.Cat_s.xCube>=0)&(self.Cat_s.xCube<NPix))
        # Cy=((self.Cat_s.yCube>=0)&(self.Cat_s.yCube<NPix))
        # ind=np.where(Cx&Cy)[0]
        # self.Cat_s=self.Cat_s[ind]

        # # ##############################
        # nn=self.Cat_s.shape[0]
        # self.Cat_s=self.Cat_s[nn//10:nn//10+1]
        # n_zm=self.DicoDATA["DicoSelFunc"]["n_zm"]
        # self.Cat_s.Pzm.fill(0.)
        # for ID in range(self.Cat_s.shape[0]):
        #     n,m=self.Cat_s.Pzm[ID].shape
        #     self.Cat_s.Pzm[ID][n//5,m//3]=1.
        #     self.Cat_s.n_zt[ID][:]=np.sum(self.Cat_s.Pzm[ID]*n_zm,axis=1)
        # # ##############################
        
        N1=self.Cat_s.shape[0]
        print>>log, "Selected %i objects [out of %i - that is %.3f%% of the main catalog]"%(N1,NN0,100*float(N1)/NN0)
        
def test(Show=True,NameOut="Test100kpc.fits"):

    Cat="/data/tasse/DataDeepFields/EN1/EN1_opt_spitzer_merged_vac_opt3as_irac4as_all_hpx_public.fits"
    Pz="/data/tasse/DataDeepFields/EN1/EN1_opt_spitzer_merged_vac_opt3as_irac4as_all_public_pz.hdf"
    MaskImage="/data/tasse/DataDeepFields/EN1/optical_images/iband/EL_EN1_iband.fits.mask.fits"
    rac,decc=241.20678,55.59485 # cluster
    #rac,decc=240.36069,55.522467 # star
    #rac,decc=radec
    COM=ClassOverdensityMap(rac,decc,
                            CellDeg=0.001,
                            NPix=501,
                            ScaleKpc=500)
    if Show:
        COM.showRGB()
    COM.setMask(MaskImage)
    COM.setCat(Cat)
    COM.setPz(Pz)
    COM.finaliseInit()
    # COM.giveDensityAtRaDec(rac,decc)
    COM.giveDensityGrid(Randomize=True)
    COM.giveDensityGrid(Randomize=False)
    COM.normaliseCube()
    COM.killWorkers()
    COM.SaveFITS(Name=NameOut)

# if __name__=="__main__":
#     try:
#         test()
#     except Exception,e:
#         print>>log, "Got an exception... : %s"%str(e)    
#         print>>log, "killing workers..."
#         APP.terminate()
#         APP.shutdown()
#         Multiprocessing.cleanupShm()

def main():
    CM=ClassCatalogMachine()
    CM.Init(ForceLoad=True)


if __name__=="__main__":
    main()
