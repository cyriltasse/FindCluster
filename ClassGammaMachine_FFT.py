

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
                 radec_main,
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
        self.rac_main,self.decc_main=self.radec_main=radec_main

        self.CoordMachine = ModCoord.ClassCoordConv(self.rac_main, self.decc_main)
        l0,m0=self.CoordMachine.radec2lm(self.rac, self.decc)

        nn=NPix//2
        lg,mg=np.mgrid[-nn*self.CellRad+l0:nn*self.CellRad+l0:1j*self.NPix,-nn*self.CellRad+m0:nn*self.CellRad+m0:1j*self.NPix]
        rag,decg=self.CoordMachine.lm2radec(lg.flatten(),mg.flatten())
        self.rag=rag.reshape(lg.shape)
        self.decg=decg.reshape(lg.shape)
        self.lg,self.mg=lg,mg

        self.GammaCube=None
        self.NSlice=self.zg.size-1
        
        self.L_NParms=[]
        self.L_fScalePix=[]
        #self.Build_Sampling_DiscreteRing()
        self.Build_Sampling_Gaussian()
        
    def Build_Sampling_ContinuousRing(self):
        freq = np.fft.fftfreq(self.NPix, d=self.CellRad)
        fPix=freq[1]-freq[0]
        self.L_Ind=[]
        fBox=0.01
        for iz in range(self.zg.size-1):
            z0,z1=self.zg[iz],self.zg[iz+1]
            zm=(z0+z1)/2.
            Mask=np.zeros((self.NPix,self.NPix),dtype=np.bool)
            for ScaleKpc in self.ScaleKpc:
                S0,S1=ScaleKpc-ScaleKpc*fBox,ScaleKpc+ScaleKpc*fBox
                print>>log,"    - Parameter mapping %.1f->%.1f kpc"%(iz,z0)
                ScalePix0=S0*cosmo.arcsec_per_kpc_comoving(zm).to_value()/(self.CellDeg*3600.)
                ScaleRad0=ScalePix0*self.CellRad
                fScaleRad0=1./ScaleRad0
                fScalePix0=int(fScaleRad0/fPix)
                
                ScalePix1=S1*cosmo.arcsec_per_kpc_comoving(zm).to_value()/(self.CellDeg*3600.)
                ScaleRad1=ScalePix1*self.CellRad
                fScaleRad1=1./ScaleRad1
                fScalePix1=int(fScaleRad1/fPix)

                f0,f1=freq.min(),freq.max()
                fx,fy=np.mgrid[f0:f1:1j*self.NPix,f0:f1:1j*self.NPix]
                r=np.sqrt(fx**2+fy**2)
                #Ind=np.arange(self.NPix**2).reshape((self.NPix,self.NPix))
                Mask[(r>fScaleRad1)&(r<fScaleRad0)]=1
                
            Mask[0:self.NPix//2,:]=0
            Mask[self.NPix//2,0:self.NPix//2]=0
            Mask[self.NPix//2,self.NPix//2]=1
            
            ind=np.where(Mask.flatten())[0]
            self.L_Ind.append(ind)
            NParms=ind.size
            self.L_NParms.append(NParms)
            print>>log,"  -- Slice #%i [z=%.2f -> z=%.2f]: %i parameters"%(iz,z0,z1,NParms)

            # import pylab
            # pylab.clf()
            # pylab.imshow(Mask,interpolation="nearest",cmap="gray")
            # pylab.draw()
            # pylab.show(False)
            # pylab.pause(0.1)
            
        self.NParms=np.sum(self.L_NParms)
        print>>log,"Total number of free parameters: %i"%self.NParms

    def Build_Sampling_DiscreteRing(self):
        self.SliceFunction=self.giveSlice_ConvPaddedFFT
        freq = np.fft.fftfreq(self.NPix, d=self.CellRad)
        fPix=freq[1]-freq[0]
        self.L_Ind=[]
        for iz in range(self.zg.size-1):
            z0,z1=self.zg[iz],self.zg[iz+1]
            zm=(z0+z1)/2.
            Mask=np.zeros((self.NPix,self.NPix),dtype=np.bool)
            for ScaleKpc in self.ScaleKpc:
                S0=ScaleKpc
                ScalePix0=S0*cosmo.arcsec_per_kpc_comoving(zm).to_value()/(self.CellDeg*3600.)
                print 
                ScaleRad0=ScalePix0*self.CellRad
                fScaleRad0=1./ScaleRad0
                fScalePix0=int(fScaleRad0/fPix)
                
                f0,f1=freq.min(),freq.max()
                fx,fy=np.mgrid[f0:f1:1j*self.NPix,f0:f1:1j*self.NPix]
                r=np.sqrt(fx**2+fy**2)
                #Ind=np.arange(self.NPix**2).reshape((self.NPix,self.NPix))
                Mask[(r>fScaleRad0-1.4*fPix/2.)&(r<fScaleRad0+1.4*fPix/2.)]=1
                
            Mask[0:self.NPix//2,:]=0
            Mask[self.NPix//2,0:self.NPix//2]=0
            Mask[self.NPix//2,self.NPix//2]=0
            
            ind=np.where(Mask.flatten())[0]
            self.L_Ind.append(ind)
            NParms=ind.size*2+1
            self.L_NParms.append(NParms)
            print>>log,"  -- Slice #%i zm=%.2f [z=%.2f -> z=%.2f]: %i parameters"%(iz,zm,z0,z1,NParms)

            # import pylab
            # pylab.clf()
            # pylab.imshow(Mask,interpolation="nearest",cmap="gray")
            # pylab.draw()
            # pylab.show(False)
            # pylab.pause(0.1)
            
        self.NParms=np.sum(self.L_NParms)
        print>>log,"Total number of free parameters: %i"%self.NParms

    def Build_Sampling_Gaussian(self):
        self.SliceFunction=self.giveSlice_ConvPaddedFFT
        freq = np.fft.fftfreq(self.NPix, d=self.CellRad)
        fPix=freq[1]-freq[0]
        self.L_Ind=[]

        for iz in range(self.zg.size-1):
            z0,z1=self.zg[iz],self.zg[iz+1]
            zm=(z0+z1)/2.
            Mask=np.zeros((self.NPix,self.NPix),dtype=np.bool)
            S0=self.ScaleKpc[-1]
            
            ScalePix0=S0*cosmo.arcsec_per_kpc_comoving(zm).to_value()/(self.CellDeg*3600.)
            ScaleRad0=ScalePix0*self.CellRad
            fScaleRad0=1./ScaleRad0
            fScalePix0=int(fScaleRad0/fPix)

            f0,f1=freq.min(),freq.max()
            fx,fy=np.mgrid[f0:f1:1j*self.NPix,f0:f1:1j*self.NPix]
            r=np.sqrt(fx**2+fy**2)
            G=np.exp(-r**2/(2.*fScaleRad0**2))
            Gs=np.sum(G)
            Nr=int(Gs/2.)#int(Gs/4.)
            Nr=int(Gs)
            #Nr=int(Gs/4.)
            Nr=np.max([100,Nr])
            def giveRand2D():
                fx_r=np.int32(np.random.randn(Nr)*fScalePix0+self.NPix//2)
                fy_r=np.int32(np.random.randn(Nr)*fScalePix0+self.NPix//2)
                cx=((fx_r>=0)&(fx_r<self.NPix))
                cy=((fy_r>=0)&(fy_r<self.NPix))
                C=(cx&cy)

                # dx=(np.random.rand(Nr)-0.5)*2*fScalePix0*2
                # dy=(np.random.rand(Nr)-0.5)*2*fScalePix0*2
                
                dx,dy=np.mgrid[-fScalePix0:fScalePix0+1,-fScalePix0:fScalePix0+1]
                dx=dx.flatten()
                dy=dy.flatten()
                C=(np.sqrt(dx**2+dy**2)<fScalePix0)
                fx_r=np.int32(dx+self.NPix//2)
                fy_r=np.int32(dy+self.NPix//2)
                
                return fx_r[C],fy_r[C]

            fx,fy=giveRand2D()
            Mask.flat[fy*self.NPix+fx]=1
            #Mask.fill(1)
            Mask[0:self.NPix//2,:]=0
            Mask[self.NPix//2,0:self.NPix//2]=0
            Mask[self.NPix//2,self.NPix//2]=0
            
            ind=np.where(Mask.flatten())[0]
            self.L_Ind.append(ind)
            NParms=ind.size*2+1
            self.L_NParms.append(NParms)
            print>>log,"  -- Slice #%i zm=%.2f [z=%.2f -> z=%.2f]: %i parameters"%(iz,zm,z0,z1,NParms)
            
            
            Mask+=Mask[::-1,::-1].conj()
            Mask[self.NPix//2,self.NPix//2]=1
            
            # import pylab
            # pylab.figure("Sampling")
            # pylab.clf()
            # pylab.imshow(Mask,interpolation="nearest",cmap="gray")
            # #pylab.imshow(G,interpolation="nearest",cmap="gray")
            # pylab.draw()
            # pylab.show(False)
            # pylab.pause(1)
            
        self.NParms=np.sum(self.L_NParms)
        print>>log,"Total number of free parameters: %i"%self.NParms


    def CubeToVec(self,Cube):
        ii=0
        Cube=np.complex64(Cube.copy())
        XOut=np.zeros((self.NParms,),np.float32)
        for iSlice in range(self.NSlice):
            FFTM=ClassFFT.FFTW_2Donly_np((self.NPix,self.NPix), Cube.dtype)#, norm=True, ncores=1, FromSharedId=None)
            N=self.L_NParms[iSlice]
            n=(N-1)/2
            Slice=FFTM.fft(Cube[iSlice])
            x=Slice.flat[self.L_Ind[iSlice]]
            XOut[ii]=Slice[self.NPix//2,self.NPix//2].real
            XOut[ii+1:ii+n+1]=x.real
            XOut[ii+n+1:ii+N]=x.imag
            ii+=N
        return XOut
            
            
    def PlotGammaCube(self,Cube=None,FigName="Gamma Cube"):
        if Cube is None:
            Cube=self.GammaCube
        import pylab
        pylab.figure(FigName)
        pylab.clf()
        for iPlot in range(9):
            S=Cube[iPlot]
            pylab.subplot(3,3,iPlot+1)
            pylab.imshow(S,interpolation="nearest")#,vmin=0.,vmax=2.)
            pylab.title("[%f - %f]"%(S.min(),S.max()))
            pylab.draw()
            pylab.show(False)
            pylab.pause(0.1)


    def computeGammaCube(self,X):
        LX=[]
        ii=0
        T=ClassTimeIt.ClassTimeIt("Gamma")
        T.disable()
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
        T.disable()
        x=LX[iSlice]
        z0=self.zg[iSlice]
        z1=self.zg[iSlice+1]
        
        F=np.zeros((self.NPix,self.NPix),np.complex64)
        xc=self.NPix//2
        
        xr=x[1:x.size//2+1]
        xi=x[x.size//2+1:]
        #print xr,xi
        
        F.flat[self.L_Ind[iSlice]]+=xr[:]
        F.flat[self.L_Ind[iSlice]]+=xi[:]*1j
        
        
        F+=F[::-1,::-1].conj()
        F[xc,xc]=x[0]
        
        #Slice=scipy.signal.fftconvolve(R, G, mode="same")
        
        Mean=.5
        Amplitude=1.
        T.timeit("unpack")
        FFTM=ClassFFT.FFTW_2Donly_np(F.shape, F.dtype)#, norm=True, ncores=1, FromSharedId=None)

        

        
        
        Slice=FFTM.ifft(F)#/(x.size-2)

        # pylab.clf()
        # pylab.subplot(1,2,1)
        # pylab.imshow(F.real,interpolation="nearest")#,vmin=0,vmax=2.)
        # pylab.subplot(1,2,2)
        # pylab.imshow(F.imag,interpolation="nearest")#,vmin=0,vmax=2.)
        # pylab.draw()
        # pylab.show(False)
        # pylab.pause(1)

        # A=Slice
        # m0,m1=A.min(),A.max()
        # Slice=(A-(m0+m1)/2.)/(m1-m0)*Amplitude+Mean

        Slice=np.real(Slice)
        #print np.mean(Slice)

        # pylab.clf()
        # pylab.imshow(GammaCube[iz,:,:],interpolation="nearest",vmin=0,vmax=2.)
        # pylab.draw()
        # pylab.show(False)
        # pylab.pause(0.1)
        Slice[Slice<0]=1e-10
        T.timeit("iFT")
        return Slice

    

