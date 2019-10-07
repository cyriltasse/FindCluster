from astropy.cosmology import WMAP9 as cosmo
import numpy as np
import scipy.signal
import ClassFFT
import matplotlib.pyplot as pylab

class ClassGammaMachine():
    def __init__(self,CellDeg,NPix,z=[0.01,2.,40],Mode="ConvGaussNoise",ScaleKpc=100):
        self.CellDeg=CellDeg
        self.NPix=NPix
        self.CellRad=CellDeg*np.pi/180
        self.zg=np.linspace(*z)
        self.Mode=Mode
        self.ScaleKpc=ScaleKpc
        if self.Mode=="ConvGaussNoise":
            self.SliceFunction=self.giveSlice_ConvGaussNoise
            self.NParms=self.NPix**2
        elif self.Mode=="ConvPaddedFFT":
            self.SliceFunction=self.giveSlice_ConvPaddedFFT
            for iz in range(self.zg.size-1):
                z0,z1=self.zg[iz],self.zg[iz+1]
                zm=(z0+z1)/2.
                ScalePix=self.ScaleKpc*cosmo.arcsec_per_kpc_comoving(zm).to_value()/(self.CellDeg*3600.)
                ScaleRad=ScalePix*self.CellRad
                freq = np.fft.fftfreq(self.NPix, d=self.CellRad)
                fScaleRad=1./ScaleRad
                fPix=freq[1]-freq[0]
                fScalePix=int(fScaleRad/fPix)
                if fScalePix%2==0: fScalePix+=1
                fScalePix=3
                N=fScalePix
                self.fScalePix=fScalePix
                self.NParms=(N//2*N+N//2)*2+1
            
    def giveSlice_ConvPaddedFFT(self,x,z0,z1):
        
        # zm=(z0+z1)/2.
        # ScalePix=self.ScaleKpc*cosmo.arcsec_per_kpc_comoving(zm).to_value()/(self.CellDeg*3600.)
        # Nsup=int(5.*ScalePix)
        # if Nsup%2==0: Nsup+=1
        # xx,yy=np.mgrid[-Nsup:Nsup+1:,-Nsup:Nsup+1]
        # dd=np.sqrt(xx**2+yy**2)
        # G=np.exp(-dd**2/(2.*ScalePix**2))
        # R=x.reshape((self.NPix,self.NPix))
        
        F0=np.zeros((self.NPix,self.NPix),np.complex64)
        xc=self.NPix//2
        Sup=self.fScalePix//2
        F=F0[xc-Sup:xc+Sup+1,xc-Sup:xc+Sup+1]
        xc0=Sup
        F[xc0,xc0]=x[0]
        xr=x[1:x.size//2+1]
        xi=x[x.size//2+1:]
        xr0=xr[0:self.fScalePix//2]
        xr1=xr[self.fScalePix//2:]
        xi0=xr[0:self.fScalePix//2]
        xi1=xr[self.fScalePix//2:]
        F[xc0+1:,xc0]=xr0+xi0*1j
        F[:xc0,xc0][::-1]=xr0-xi0*1j
        F[:,xc0+1:]=(xr1+1j*xi1).reshape((self.fScalePix,self.fScalePix//2))
        F[:,:xc0]=F[:,xc0+1:][::-1,::-1].conj()
        
        #Slice=scipy.signal.fftconvolve(R, G, mode="same")
        
        Mean=.5
        Amplitude=1.
        FFTM=ClassFFT.FFTW_2Donly_np(F0.shape, F0.dtype)#, norm=True, ncores=1, FromSharedId=None)

        

        Slice=FFTM.ifft(F0)
        # pylab.clf()
        # pylab.imshow(F0.real,interpolation="nearest")#,vmin=0,vmax=2.)
        # pylab.draw()
        # pylab.show(False)
        # pylab.pause(0.1)
        
        # A=Slice
        # m0,m1=A.min(),A.max()
        # Slice=(A-(m0+m1)/2.)/(m1-m0)*Amplitude+Mean

        Slice=np.real(Slice)

        # pylab.clf()
        # pylab.imshow(GammaCube[iz,:,:],interpolation="nearest",vmin=0,vmax=2.)
        # pylab.draw()
        # pylab.show(False)
        # pylab.pause(0.1)
        Slice[Slice<0]=1e-10
        return Slice

    

    def giveGammaCube(self,X):
        GammaCube=np.zeros((self.zg.size-1,self.NPix,self.NPix),np.float32)
        for iz in range(self.zg.size-1):
            z0,z1=self.zg[iz],self.zg[iz+1]
            GammaCube[iz,:,:]=self.SliceFunction(X,z0,z1)
        return GammaCube
            
    def giveSlice_ConvGaussNoise(self,x,z0,z1):
        Mean=.5
        Amplitude=1.
        zm=(z0+z1)/2.
        R=x.reshape((self.NPix,self.NPix))
        ScalePix=self.ScaleKpc*cosmo.arcsec_per_kpc_comoving(zm).to_value()/(self.CellDeg*3600.)
        Nsup=int(5.*ScalePix)
        if Nsup%2==0: Nsup+=1
        xx,yy=np.mgrid[-Nsup:Nsup+1:,-Nsup:Nsup+1]
        dd=np.sqrt(xx**2+yy**2)
        G=np.exp(-dd**2/(2.*ScalePix**2))
        Slice=scipy.signal.fftconvolve(R, G, mode="same")
        
        A=Slice
        m0,m1=A.min(),A.max()
        Slice=(A-(m0+m1)/2.)/(m1-m0)*Amplitude+Mean
            
        # pylab.clf()
        # pylab.imshow(GammaCube[iz,:,:],interpolation="nearest",vmin=0,vmax=2.)
        # pylab.draw()
        # pylab.show(False)
        # pylab.pause(0.1)
        Slice[Slice<0]=0.
        return Slice


