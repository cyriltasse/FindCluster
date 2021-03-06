from astropy.cosmology import WMAP9 as cosmo
import numpy as np
import scipy.signal
import ClassFFT
import matplotlib.pyplot as pylab
from DDFacet.ToolsDir import ModCoord
from DDFacet.Other import ClassTimeIt
from DDFacet.Other import logger
log = logger.getLogger("ClassGammaMachine")

class ClassGammaMachine():
    def __init__(self,
                 radec,
                 CellDeg,
                 NPix,
                 zParms=[0.01,2.,40],
                 Mode="ConvGaussNoise",
                 ScaleKpc=100):
        self.radec=radec
        self.rac,self.decc=self.radec
        self.CellDeg=CellDeg
        self.NPix=NPix
        self.CellRad=CellDeg*np.pi/180
        self.zg=np.linspace(*zParms)
        self.Mode=Mode
        self.ScaleKpc=ScaleKpc
        
        self.CoordMachine = ModCoord.ClassCoordConv(self.rac, self.decc)
        nn=NPix//2
        lg,mg=np.mgrid[-nn*self.CellRad:nn*self.CellRad:1j*self.NPix,-nn*self.CellRad:nn*self.CellRad:1j*self.NPix]
        rag,decg=self.CoordMachine.lm2radec(lg.flatten(),mg.flatten())
        self.rag=rag.reshape(lg.shape)
        self.decg=decg.reshape(lg.shape)
        self.lg,self.mg=lg,mg
        self.GammaCube=None
        self.NSlice=self.zg.size-1
        
        if self.Mode=="ConvGaussNoise":
            self.SliceFunction=self.giveSlice_ConvGaussNoise
            self.NParms=self.NPix**2
        elif self.Mode=="ConvPaddedFFT":
            self.SliceFunction=self.giveSlice_ConvPaddedFFT
            self.L_NParms=[]
            self.L_fScalePix=[]
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
                #fScalePix=7
                N=fScalePix
                self.L_fScalePix.append(fScalePix)
                NParms=(N//2*N+N//2)*2+1
                self.L_NParms.append(NParms)
                print>>log,"  - Slice #%i [z=%.2f -> z=%.2f]: %i parameters"%(iz,z0,z1,NParms)

            self.NParms=np.sum(self.L_NParms)
        print>>log,"Total number of free parameters: %i"%self.NParms

    def PlotGammaCube(self):
        import pylab
        pylab.figure("Gamma Cube")
        pylab.clf()
        for iPlot in range(9):
            pylab.subplot(3,3,iPlot+1)
            pylab.imshow(self.GammaCube[iPlot],interpolation="nearest")
            pylab.draw()
            pylab.show(False)
            pylab.pause(0.1)

    def giveGamma(self,z,ra,dec):
        
        if self.GammaCube is None:
            return np.ones((ra.size,),np.float32)
        
        if "float" not in str(type(z)): stop
        if type(ra) is not np.ndarray:
            ra=np.array([ra])
        if type(dec) is not np.ndarray:
            dec=np.array([dec])

        l,m=self.CoordMachine.radec2lm(ra,dec)
        x=np.int32(np.around(l/self.CellRad))+self.NPix//2
        y=np.int32(np.around(m/self.CellRad))+self.NPix//2
        zm=(self.zg[0:-1]+self.zg[1:])/2.
        iz=np.argmin(np.abs(zm-z))
        nx,ny=self.GammaCube[iz].shape
        # print "giveGamma",x,y
        ind=x*ny+y
        return self.GammaCube[iz].flat[ind]

    

    def giveSlice_ConvPaddedFFT(self,LX,iSlice):
        
        # zm=(z0+z1)/2.
        # ScalePix=self.ScaleKpc*cosmo.arcsec_per_kpc_comoving(zm).to_value()/(self.CellDeg*3600.)
        # Nsup=int(5.*ScalePix)
        # if Nsup%2==0: Nsup+=1
        # xx,yy=np.mgrid[-Nsup:Nsup+1:,-Nsup:Nsup+1]
        # dd=np.sqrt(xx**2+yy**2)
        # G=np.exp(-dd**2/(2.*ScalePix**2))
        # R=x.reshape((self.NPix,self.NPix))

        T=ClassTimeIt.ClassTimeIt("Slice")
        x=LX[iSlice]
        z0=self.zg[iSlice]
        z1=self.zg[iSlice+1]
        fScalePix=self.L_fScalePix[iSlice]
        
        F0=np.zeros((self.NPix,self.NPix),np.complex64)
        xc=self.NPix//2
        Sup=fScalePix//2
        F=F0[xc-Sup:xc+Sup+1,xc-Sup:xc+Sup+1]
        xc0=Sup
        F[xc0,xc0]=x[0]
        xr=x[1:x.size//2+1]
        xi=x[x.size//2+1:]
        xr0=xr[0:fScalePix//2]
        xr1=xr[fScalePix//2:]
        xi0=xr[0:fScalePix//2]
        xi1=xr[fScalePix//2:]
        F[xc0+1:,xc0]=xr0+xi0*1j
        F[:xc0,xc0][::-1]=xr0-xi0*1j
        F[:,xc0+1:]=(xr1+1j*xi1).reshape((fScalePix,fScalePix//2))
        F[:,:xc0]=F[:,xc0+1:][::-1,::-1].conj()
        
        #Slice=scipy.signal.fftconvolve(R, G, mode="same")
        
        Mean=.5
        Amplitude=1.
        T.timeit("unpack")
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
        T.timeit("iFT")
        return Slice

    

    def computeGammaCube(self,X):
        LX=[]
        ii=0
        T=ClassTimeIt.ClassTimeIt("Gamma")
        for iSlice in range(self.NSlice):
            LX.append(X[ii:ii+self.L_NParms[iSlice]])
            ii+=self.L_NParms[iSlice]
        T.timeit("unpack")
        GammaCube=np.zeros((self.zg.size-1,self.NPix,self.NPix),np.float32)
        for iz in range(self.zg.size-1):
            z0,z1=self.zg[iz],self.zg[iz+1]
            GammaCube[iz,:,:]=self.SliceFunction(LX,iz)
        T.timeit("Slices")
        self.GammaCube=GammaCube

            
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


