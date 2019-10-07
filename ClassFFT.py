import scipy
import numpy as np
import pyfftw
import psutil
import numexpr
from DDFacet.Other import ClassTimeIt
Fs=scipy.fftpack.fftshift
iFs=scipy.fftpack.ifftshift

class FFTW_2Donly():
    def __init__(self, shape, dtype, norm=True, ncores=1, FromSharedId=None):
        # if FromSharedId is None:
        #     self.A = pyfftw.n_byte_align_empty( shape[-2::], 16, dtype=dtype)
        # else:
        #     self.A = NpShared.GiveArray(FromSharedId)

        #pyfftw.interfaces.cache.enable()
        #pyfftw.interfaces.cache.set_keepalive_time(3000)
        self.ncores=ncores or NCPU_global
        #print "plan"
        T= ClassTimeIt.ClassTimeIt("ModFFTW")
        T.disable()

        #self.A = pyfftw.interfaces.numpy_fft.fft2(self.A, axes=(-1,-2),overwrite_input=True, planner_effort='FFTW_MEASURE',  threads=self.ncores)
        T.timeit("planF")
        #self.A = pyfftw.interfaces.numpy_fft.ifft2(self.A, axes=(-1,-2),overwrite_input=True, planner_effort='FFTW_MEASURE',  threads=self.ncores)
        T.timeit("planB")
        #print "done"
        self.ThisType=dtype
        self.norm = norm

    def fft(self, Ain):
        axes=(-1,-2)

        T= ClassTimeIt.ClassTimeIt("ModFFTW")
        T.disable()

        sin=Ain.shape
        if len(Ain.shape)==2:
            s=(1,1,Ain.shape[0],Ain.shape[1])
            A=Ain.reshape(s)
        else:
            A=Ain

        nch,npol,_,_=A.shape
        for ich in range(nch):
            for ipol in range(npol):
                A_2D = iFs(A[ich,ipol].astype(self.ThisType),axes=axes)
                T.timeit("shift and copy")
                A_2D[...] = pyfftw.interfaces.numpy_fft.fft2(A_2D, axes=(-1,-2),overwrite_input=True, planner_effort='FFTW_MEASURE', threads=self.ncores)
                T.timeit("fft")
                A[ich,ipol]=Fs(A_2D,axes=axes)
                T.timeit("shift")
        if self.norm:
            A /= (A.shape[-1] * A.shape[-2])

        return A.reshape(sin)

    def ifft(self, A, norm=True):
        axes=(-1,-2)
        sin=A.shape
        if len(A.shape)==2:
            s=(1,1,A.shape[0],A.shape[1])
            A=A.reshape(s)
        #log=MyLogger.getLogger("ModToolBox.FFTM2.ifft")
        nch,npol,_, _ = A.shape
        for ich in range(nch):
            for ipol in range(npol):
                A_2D = iFs(A[ich,ipol].astype(self.ThisType),axes=axes)
                A_2D[...] = pyfftw.interfaces.numpy_fft.ifft2(A_2D, axes=(-1,-2),overwrite_input=True, planner_effort='FFTW_MEASURE', threads=self.ncores)
                A[ich,ipol]=Fs(A_2D,axes=axes)
        if self.norm:
            A *= (A.shape[-1] * A.shape[-2])
        return A.reshape(sin)


# LAPACK (or ATLAS???) version of the FFT engine
class FFTW_2Donly_np():
    def __init__(self, shape=None, dtype=None, ncores = 1):

        return

    def fft(self,A,ChanList=None):
        axes=(-1,-2)

        T= ClassTimeIt.ClassTimeIt("ModFFTW")
        T.disable()

        n,n=A.shape



        B = iFs(A.astype(A.dtype),axes=axes)
        T.timeit("shift and copy")
        B = np.fft.fft2(B,axes=axes)
        T.timeit("fft")
        A=Fs(B,axes=axes)/(A.shape[-1]*A.shape[-2])
        T.timeit("shift")

        return A

    def ifft(self,A,ChanList=None):
        axes=(-1,-2)
        #log=MyLogger.getLogger("ModToolBox.FFTM2.ifft")


        B = iFs(A.astype(A.dtype),axes=axes)
        B = np.fft.ifft2(B,axes=axes)
        A=Fs(B,axes=axes)*(A.shape[-1]*A.shape[-2])

        return A

