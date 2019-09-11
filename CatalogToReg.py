#!/usr/bin/env python

import numpy as np
from rad2hmsdms import rad2hmsdms
import astropy.io.fits as pyfits
import sys
from astroquery.gaia import Gaia
import ClassOverdensityMap

def getGaiaCat():
    tables = Gaia.load_tables(only_names=True)
    ra=15.*(16+10./60+55.650/3600)
    dec=54+58/60.+18.36/3600.
    R=1.64274
    job = Gaia.launch_job_async("SELECT * \
    FROM gaiadr1.gaia_source \
    WHERE CONTAINS(POINT('ICRS',gaiadr1.gaia_source.ra,gaiadr1.gaia_source.dec),CIRCLE('ICRS',%f,%f,%f))=1 \
    AND phot_g_mean_mag < 18;"%(ra,dec,R) , dump_to_file=True)
    r=job.get_results()
    #np.savez("/data/tasse/DataDeepFields/EN1/GAIA_EN1.npz")
    return np.array(r["ra"])*np.pi/180,np.array(r["dec"])*np.pi/180,np.array(r["phot_g_mean_mag"])
    
def CatToReg(CatName=None,Type="GAIA",OutReg=None,Color="green"):



    if Type=="GAIA":
        Cat=pyfits.open(CatName)[1].data
        Cat=Cat.view(np.recarray)
        ra=Cat["ra"][:]*np.pi/180
        dec=Cat["dec"][:]*np.pi/180
    elif Type=="GAIARequest":
        ra,dec,Mag=getGaiaCat()
    elif Type=="npz":
        RAName="ra"
        DECName="dec"
        D=np.load(CatName)
        ra=D["ra"]*np.pi/180
        dec=D["dec"]*np.pi/180
    elif Type=="LoTSS":        
        Cat=pyfits.open(CatName)[1].data
        Cat=Cat.view(np.recarray)
        ind=np.where((Cat.FLAG_CLEAN == 1)&(Cat.i_fluxerr > 0)&
                     (Cat.ch2_swire_fluxerr > 0)&(Cat.CLASS_STAR < 0.2))[0]
        Cat=Cat[ind]
        ra=Cat["RA"][:]*np.pi/180
        dec=Cat["DEC"][:]*np.pi/180 
        ra=ra[::11]
        dec=dec[::11]
   
    z=np.array([ -1.70694363e-06,   1.06015353e-04,  -2.25874420e-03, 1.76217309e-02,  -2.37432427e-02])
    p = np.poly1d(z)
    
    if OutReg:
        f=open(OutReg,"w")
    else:
        f=open("%s.reg"%CatName,"w")
        
    f.write("""# Region file format: DS9 version 4.1\n""")
    f.write("""global color=%s dashlist=8 3 width=2 font="helvetica 10 normal roman" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1\n"""%Color)
    f.write("""fk5\n""")



    for i in range(ra.size):
        print ("%i/%i"%(i,ra.size))
        tra=ra[i]
        tdec=dec[i]
        sra=rad2hmsdms(tra,Type="ra",deg=False)
        sdec=rad2hmsdms(tdec,Type="dec",deg=False)
        sdec1=rad2hmsdms(tdec+6e-3*np.pi/180,Type="dec",deg=False)
        sra=sra.replace(" ",":")
        sdec=sdec.replace(" ",":")
        sdec1=sdec1.replace(" ",":")
        if Type=="LoTSS":
            f.write("""point(%s,%s) # point=cross\n"""%(sra,sdec))
        elif Type=="GAIARequest":
            RDeg=np.max([p(Mag[i]),0])
            f.write("""circle(%s,%s,%f"\n"""%(sra,sdec,1.1*RDeg*3600.))
        #f.write("""# text(%s,%s) text={%4.2f}\n"""%(sra,sdec1,Mag[i]))
    f.close()

def testGAIA():
    # CatName="/data/tasse/DataDeepFields/EN1/1567424183117O-result.fits"
    # CatToReg(CatName)
    # CatToReg(Type="GAIARequest",OutReg="/data/tasse/DataDeepFields/EN1/GAIAMags.reg",Color="red")
    CatToReg(CatName="/data/tasse/DataDeepFields/EN1/EN1_opt_spitzer_merged_vac_opt3as_irac4as_all_hpx_public.fits",Type="LoTSS")
    



def PlotDistCat():
    CatName="/data/tasse/DataDeepFields/EN1/EN1_opt_spitzer_merged_vac_opt3as_irac4as_all_hpx_public.fits"
    Cat=pyfits.open(CatName)[1].data
    Cat=Cat.view(np.recarray)
    ind=np.where((Cat.FLAG_CLEAN == 1)&(Cat.i_fluxerr > 0)&
                 (Cat.ch2_swire_fluxerr > 0))[0]
    Cat=Cat[ind]
    ra0=Cat["RA"][:]*np.pi/180
    dec0=Cat["DEC"][:]*np.pi/180
    ra1,dec1,M1=getGaiaCat()
    D1=[]
    D2=[]
    D3=[]
    M=[]
    R=30./60
    for i in range(ra1.size)[::3]:
        print "%i/%i"%(i,ra1.size)
        #d=ClassOverdensityMap.AngDist(ra0,dec0,ra1[i],dec1[i])
        #Cx=np.where((ra0<ra1[i]+R)&(ra0>ra1[i]-R))[0]
        #Cy=np.where((dec0<dec1[i]+R)&(dec0>dec1[i]-R))[0]
        ra0s=ra0#[Cx&Cy]
        dec0s=dec0#[Cx&Cy]
        d=np.sort(np.sqrt((ra0s-ra1[i])**2+(dec0s-dec1[i])**2)*180/np.pi)
        #ind=np.where(d<R)[0]
        D1.append(d[1])
        D2.append(d[2])
        D3.append(d[3])
        M.append(M1[i])

    import pylab
    pylab.clf()
    pylab.scatter(M,D1,c="blue")
    pylab.scatter(M,D2,c="green")
    pylab.scatter(M,D3,c="red")
    pylab.draw()
    pylab.show(False)
        
if __name__=="__main__":
    CatToReg(sys.argv[1])
