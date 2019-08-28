#!/usr/bin/env python
from astropy.io import fits
from astropy import wcs
import optparse
import numpy as np
import os
import pickle


def read_options():
    desc=""" """
    
    opt = optparse.OptionParser(usage='Usage: %prog <options>',version='%prog version 1.0',description=desc)
    
    group = optparse.OptionGroup(opt, "* Data selection options")
    group.add_option('--InFile',type=str,help='',default=None)
    group.add_option('--OutFile',type=str,help='',default=None)
    group.add_option('--Ra',type=float,help='',default=None)
    group.add_option('--Dec',type=float,help='',default=None)
    group.add_option('--N',type=int,help='',default=None)
    group.add_option('--dPix',type=int,help='',default=1)
    opt.add_option_group(group)
    
    options, arguments = opt.parse_args()
    f = open("last_param.obj","wb")
    pickle.dump(options,f)
    return options

def CutFits(InFile=None,OutFile=None,Ra=None,Dec=None,N=100,Overwrite=True,dPix=1,boxArcMin=None):
    if OutFile is None:
        OutFile="%s.cut.fits"%InFile
    if not Overwrite:
        if os.path.isfile(OutFile):
            return

    f = fits.open(InFile)
    h=f[0].header
    for key in h.keys():
        if "PC"==key[0:2]: del(h[key])
    w = wcs.WCS(h)
    newf = fits.PrimaryHDU()
    N=int(N)
    if boxArcMin is not None:
        dPixDeg=abs(w.wcs.cd.flat[0])
        N=int((boxArcMin/60.)/dPixDeg)
        
    if Ra is None:
        Ra=w.wcs.crval[0]
    if Dec is None:
        Dec=w.wcs.crval[1]

    print Ra,Dec
    print "Centering on (Ra, Dec) = %f, %f"%(Ra,Dec)
    xc,yc=w.all_world2pix(Ra,Dec,1)
    xc=int(xc[()])
    yc=int(yc[()])
    print xc,yc,N
    
    newf.data = f[0].data[yc-N:yc+N:dPix,xc-N:xc+N:dPix]
    newf.header = f[0].header
    newf.header.update(w[yc-N:yc+N:dPix,xc-N:xc+N:dPix].to_header())

    print newf.data.shape

    # newf.data = f[0].data[0,0,yc-N:yc+N,xc-N:xc+N]
    # newf.header = f[0].header
    # newf.header.update(w[:,:,yc-N:yc+N,xc-N:xc+N].to_header())
    # newf.header["NAXIS"]=2
    # for key in newf.header.keys():
    #     if len(key)==0: continue
    #     if key[-1]=="3" or key[-1]=="4":
    #         del(newf.header[key])
    
    
    if os.path.isfile(OutFile):
        os.system("rm %s"%OutFile)
    print "Writting image %s"%OutFile
    newf.writeto(OutFile)
    return newf.data

def main(options=None):

    if options is None:
        f = open("last_param.obj",'rb')
        options = pickle.load(f)
    

    CutFits(**options.__dict__)

if __name__=="__main__":
    OP=read_options()
    main(OP)
    #postage(fitsim,postfits,ra,dec,s)



# def postage(fitsim,postfits,ra,dec):

#     head = pf.getheader(fitsim)

#     hdulist = pf.open(fitsim)
#     # Parse the WCS keywords in the primary HDU
#     wcs = pw.WCS(hdulist[0].header)

#     # Some pixel coordinates of interest.
#     skycrd = np.array([ra,dec])
#     skycrd = np.array([[ra,dec,0,0]], np.float_)

#     # Convert pixel coordinates to world coordinates
#     # The second argument is "origin" -- in this case we're declaring we
#     # have 1-based (Fortran-like) coordinates.
#     pixel = wcs.wcs_sky2pix(skycrd, 1)
#     # Some pixel coordinates of interest.

#     x = pixel[0][0]
#     y = pixel[0][1]
#     pixsize = abs(wcs.wcs.cdelt[0])
#     if pl.isnan(s):
#         s = 25.
#     N = (s/pixsize)
#     print 'x=%.5f, y=%.5f, N=%i' %(x,y,N)

#     ximgsize = head.get('NAXIS1')
#     yimgsize = head.get('NAXIS2')

#     if x ==0:
#         x = ximgsize/2
#     if y ==0:
#         y = yimgsize/2

#     offcentre = False
#     # subimage limits: check if runs over edges
#     xlim1 =  x - (N/2)
#     if(xlim1<1):
#         xlim1=1
#         offcentre=True
#     xlim2 =  x + (N/2)
#     if(xlim2>ximgsize):
#         xlim2=ximgsize
#         offcentre=True
#     ylim1 =  y - (N/2)
#     if(ylim1<1):
#         ylim1=1
#         offcentre=True
#     ylim2 =  y + (N/2)
#     if(ylim2>yimgsize):
#         offcentre=True
#         ylim2=yimgsize

#     xl = int(xlim1)
#     yl = int(ylim1)
#     xu = int(xlim2)
#     yu = int(ylim2)
#     print 'postage stamp is %i x %i pixels' %(xu-xl,yu-yl)

#     # make fits cutout
#     inps = fitsim + '[%0.0f:%0.0f,%0.0f:%0.0f]' %(xl,xu,yl,yu)

#     if os.path.isfile(postfits): os.system('rm '+postfits)
#     os.system( 'fitscopy %s %s' %(inps,postfits) )
#     print  'fitscopy %s %s' %(inps,postfits) 

#     return postfits



