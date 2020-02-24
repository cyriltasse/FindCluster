from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import os.path

import numpy as np
from DDFacet.Other import logger
log= logger.getLogger("ClassSaveFITS")
from astropy.io import fits
from astropy.wcs import WCS


class ClassSaveFITS():

    def __init__(self,ImageName,ImShape,Cell,radec,Freqs=None,KeepCasa=False,Stokes=["I"],header_dict=None,history=None):
        """ Create internal data structures, then call CreateScratch to
        make the image itself. 
        header_dict is a dict of FITS keywords to add.
        history is a list of strings to put in the history
        """
        self.Cell=Cell
        self.radec=radec
        self.Freqs = Freqs
        self.header_dict=header_dict
        
        self.ImShape=ImShape
        self.nch,self.Npix,_ = ImShape

        self.ImageName=ImageName
        self.imageFlipped = False
        self.createScratch()
        # Fill in some standard keywords

        self.header['BTYPE'] = 'Intensity'
        self.header['BUNIT'] = 'Jy/beam'
        self.header['SPECSYS'] = 'TOPOCENT'
        if header_dict is not None:
            for k in header_dict:
                self.header[k]=header_dict[k]
        # if history is not None:
        #     if isinstance(history,str):
        #         history=[history]
        #     for h in history:
        #         self.header['HISTORY']=h

    def createScratch(self):
        """ Create the structures necessary to save the FITS image """

        self.w = WCS(naxis=3)
        self.w.wcs.ctype = ['RA---SIN','DEC--SIN','FREQ']
        self.w.wcs.cdelt[0] = -self.Cell
        self.w.wcs.cdelt[1] = self.Cell
        self.w.wcs.cunit = ['deg','deg','Hz']
        self.w.wcs.crval = [self.radec[0]*180.0/np.pi,self.radec[1]*180.0/np.pi,0]
        self.w.wcs.crpix = [1+(self.Npix-1)/2.0,1+(self.Npix-1)/2.0,1]

        self.fmean=None
        if self.Freqs is not None:

            self.w.wcs.crval[2] = self.Freqs[0]

            if self.Freqs.size>1:
                F=self.Freqs
                df=np.mean(self.Freqs[1::]-self.Freqs[0:-1])
                self.w.wcs.cdelt[2]=df
                self.fmean=np.mean(self.Freqs)
            else:
                self.fmean=self.Freqs[0]

        self.header = self.w.to_header()
        if self.fmean is not None:
            self.header['RESTFRQ'] = self.fmean

    def setdata(self, dataIn,CorrT=False):
        #log.print( "  ----> put data in casa image %s"%self.ImageName)

        data=dataIn.copy()
        if CorrT:
            nch,npol,_,_=dataIn.shape
            for ch in range(nch):
                data[ch,pol]=dataIn[ch][::-1].T
        self.imageFlipped = CorrT
        self.data = data

    def ToFits(self):
        FileOut=self.ImageName
        if FileOut[-5:]!=".fits": FileOut="%s.fits"%FileOut
        
        hdu = fits.PrimaryHDU(header=self.header,data=self.data)
        if os.path.exists(FileOut):
            os.unlink(FileOut)
        log.print( "  ----> Save image data as FITS file %s"%FileOut)
        hdu.writeto(FileOut)

    def close(self):
        #log.print( "  ----> Closing %s"%self.ImageName)
        del(self.data)
        del(self.header)
        #log.print( "  ----> Closed %s"%self.ImageName)


def test():
    np.random.seed(0)
    name,imShape,Cell,radec="lala3.psf", (10, 1029, 1029), 20, (3.7146787856873478, 0.91111035090915093)
    im=ClassCasaimage(name,imShape,Cell,radec,
                      Freqs=np.linspace(1400e6,1500e6,20),
                      header_dict={'comment':'A test'},
                      history=['Here is a history line.','Here is another'])
    im.setdata(np.random.randn(*(imShape)).astype(np.float32))#,CorrT=True)
    im.ToFits()
    im.close()

if __name__=="__main__":
    test()
