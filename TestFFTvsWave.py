import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as pylab

import pywt
import pywt.data

import ClassFFT
from DDFacet.Other import ClassTimeIt

def test():
    # Load image
    original = pywt.data.camera()
    Cube=np.load("Cube.npy")
    Slice= Cube[-1].copy()
    plt.figure(2)
    plt.imshow(Slice)
    plt.draw()
    plt.show(False)
    
    NPix=Slice.shape[-1]
    # Wavelet transform of image, and plot approximation and details
    titles = ['Approximation', ' Horizontal detail',
              'Vertical detail', 'Diagonal detail']

    N=1000
    T=ClassTimeIt.ClassTimeIt()
    FFTM=ClassFFT.FFTW_2Donly_np((NPix,NPix), Cube.dtype)#, norm=True, ncores=1, FromSharedId=None)
    for i in range(N):
        FFTM.fft(Slice.copy())
    T.timeit("fft")

    level=3
    for i in range(N):
        #coeffs2 = pywt.dwt2(np.real(Slice), 'haar')
        coeffs2 = pywt.wavedec2(np.real(Slice), 'haar', mode='periodization', level=level)
    
    T.timeit("wave")

    LL, (LH, HL, HH) = coeffs2
    fig = plt.figure(figsize=(12, 3))
    for i, a in enumerate([LL, LH, HL, HH]):
        ax = fig.add_subplot(1, 4, i + 1)
        ax.imshow(a, interpolation="nearest", cmap=plt.cm.gray)
        ax.set_title(titles[i], fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    stop
    fig.tight_layout()
    plt.show(False)
    stop

def test2():
    # Load image
    original = pywt.data.camera()
    Cube=np.load("Cube.npy")
    Slice= Cube[6].copy()
    plt.figure(2)
    plt.imshow(Slice)
    plt.draw()
    plt.show(False)
    
    NPix=Slice.shape[-1]
    # Wavelet transform of image, and plot approximation and details
    titles = ['Approximation', ' Horizontal detail',
              'Vertical detail', 'Diagonal detail']

    N=1000
    T=ClassTimeIt.ClassTimeIt()
    FFTM=ClassFFT.FFTW_2Donly_np((NPix,NPix), Cube.dtype)#, norm=True, ncores=1, FromSharedId=None)
    for i in range(N):
        FFTM.fft(Slice.copy())
    T.timeit("fft")
 
    Type="bior1.3"
    Type="db10"
    # Type="coif10"

    # #Type="sym20"
    # Type="sym2"
    # Type="rbio1.3"
    
    # Type="haar"
    level=7
    c = pywt.wavedec2(np.real(Slice),
                      Type,
                      mode='periodization',
                      level=level)
    arr, slices = pywt.coeffs_to_array(c)

    abs_arr=np.abs(arr).flatten()
    ind=np.where(abs_arr<1e-2*abs_arr.max())[0]
    arr.flat[ind]=0.
    N1=(arr.size-float(ind.size))
    print N1,N1/arr.size

    for i in range(N):
        c=pywt.array_to_coeffs(arr,slices,output_format='wavedec2')
        Slice1=pywt.waverec2(c, Type, mode='periodization')

    T.timeit("wave")

    pylab.clf()
    pylab.subplot(1,2,1)
    pylab.imshow(Slice)
    pylab.subplot(1,2,2)
    pylab.imshow(Slice1)
    pylab.tight_layout()
    pylab.show(False)

