import numpy as np
import scipy.sparse
import pylab
import scipy.sparse.linalg
import DDFacet.Other.ClassTimeIt as ClassTimeIt

def sqrtSVD(A):
    u,s,v=np.linalg.svd(A)
    s[s<0.]=0.
    ssq=np.sqrt(np.abs(s))
    v0=v.T*ssq.reshape(1,ssq.size)
    Asq=np.conj(np.dot(v0,u.T))
    return Asq

def toSparse(C,Th=1e-2):
    Ca=np.abs(C)
    indi,indj=np.where(Ca>Th*Ca.max())
    data=C[indi,indj]
    M=C.shape[0]
    Cs=scipy.sparse.coo_matrix((data, (indi, indj)), shape=(M, M))
    S=Cs.size/M**2
    print("Sparcity %f [%i]"%(S,int(S*M)))
    return Cs

def testSVDTimes():

    LN=[11,21,31,41,51,81,101,131,151]
    LT=[]

    fig=pylab.figure(0)
    
    for iN,N in enumerate(LN):
        T=ClassTimeIt.ClassTimeIt()
        dd=N#10.
        x,y=np.mgrid[-dd:dd:1j*N,-dd:dd:1j*N]
        dx=x.reshape((-1,1))-x.reshape((1,-1))
        dy=y.reshape((-1,1))-y.reshape((1,-1))
        d=dx**2+dy**2
        Sig=2.

        T.timeit("Init")
        C=np.exp(-(d)/(2.*Sig**2))
        T.timeit("C")
        Cs=toSparse(C)
        T.timeit("ToSparse")

    
        CellSize=dd/N
        A=(CellSize*N)**2
        M=C.shape[0]
        k=np.max([A/Sig**2,1.])
        k=int(k*7)
        k=np.min([M-1,k])
        print("k=%i"%k)
        # k=100 # M-1
        T.reinit()
        U,s,V=scipy.sparse.linalg.svds(Cs,k=k)
        LT.append(T.timeit("SVD"))
        print(U.shape)
        scipy.sparse.linalg.eigsh(Cs,k=k)
        T.timeit("Eig")
        for ik in range(k)[0:10]:
            u=U[:,ik].reshape(N,N)
            pylab.clf()
            pylab.imshow(np.abs(u),interpolation="nearest")
            pylab.title("%i/%i"%(ik,k))
            pylab.draw()
            pylab.show(block=False)
            pylab.pause(0.1)
            
        print(LT)
        s[s<0.]=0.
        ssq=np.sqrt(np.abs(s))
        v0=V.T*ssq.reshape(1,ssq.size)
        sqrtCs=np.conj(np.dot(v0,U.T))

        pylab.clf()
        pylab.imshow(sqrtCs)
        pylab.draw()
        pylab.show(False)
        pylab.pause(0.1)
        fig.savefig("SVD%3.3i.png"%iN)
        np.save("SVD%3.3i"%iN,sqrtCs)
        
    pylab.clf()
    pylab.plot(LN,LT)
    pylab.draw()
    pylab.show(False)

    

def test():
    T=ClassTimeIt.ClassTimeIt()
    N=21
    dd=10.
    x,y=np.mgrid[-dd:dd:1j*N,-dd:dd:1j*N]
    dx=x.reshape((-1,1))-x.reshape((1,-1))
    dy=y.reshape((-1,1))-y.reshape((1,-1))
    d=dx**2+dy**2
    Sig=1.

    T.timeit("Init")
    C=np.exp(-(d)/(2.*Sig**2))
    T.timeit("C")

    # pylab.clf()
    # pylab.imshow(C,interpolation="nearest")
    # pylab.draw()
    # pylab.show(False)
    Cs=toSparse(C)
    T.timeit("ToSparse")

    
    CellSize=dd/N
    A=(CellSize*N)**2
    M=C.shape[0]
    k=np.max([A/Sig**2,1.])
    k=int(k*7)
    k=int(np.min([M-1,k]))
    print("k=%i"%k)
    # k=100 # M-1
    Us,ss,Vs=scipy.sparse.linalg.svds(Cs,k=k)
    U,s,V=np.linalg.svd(C)
    print(U.shape,V.shape)
    T.timeit("SVDs0")

    
    # U,s,V=scipy.sparse.linalg.svds(Cs,k=k,return_singular_vectors="u")
    # T.timeit("SVDs")

    # u,s,v=np.linalg.svd(C)
    # T.timeit("SVD")

    # u,s,v=np.linalg.svd(C,hermitian=True)
    # T.timeit("SVDh")

    # pylab.clf()
    # pylab.plot(s)
    # pylab.draw()
    # pylab.show(block=False)
    
    #s[s<0.]=0.
    #ssq=np.sqrt(np.abs(s))
    ssq=np.sqrt(s)
    
    ssqs=np.sqrt(ss)
    sqrtCs=Us*ssqs.reshape(1,ssqs.size)
    #sqrtCs=U.dot(ssq.reshape((-1,1)))

    
    #sqrtCs=U*ssq.reshape(1,ssq.size)


    # V=V.T
    # v0,v1=U.min(),U.max()
    # pylab.clf()
    # ax=pylab.subplot(1,3,1)
    # pylab.imshow(U,interpolation="nearest",vmin=v0,vmax=v1)
    # pylab.subplot(1,3,2,sharex=ax,sharey=ax)
    # pylab.imshow(V,interpolation="nearest",vmin=v0,vmax=v1)
    # pylab.subplot(1,3,3,sharex=ax,sharey=ax)
    # pylab.imshow(U-V,interpolation="nearest")
    # pylab.draw()
    # pylab.show(block=False)
    # return

    Nr=10000
    X=np.random.randn(sqrtCs.shape[1],Nr)
    Y=sqrtCs.dot(X)
    Cm=np.dot(Y,Y.T)/Y.shape[1]

    v0,v1=C.min(),C.max()
    pylab.clf()
    ax=pylab.subplot(1,3,1)
    pylab.imshow(C,interpolation="nearest",vmin=v0,vmax=v1)
    pylab.subplot(1,3,2,sharex=ax,sharey=ax)
    pylab.imshow(Cm,interpolation="nearest",vmin=v0,vmax=v1)
    pylab.subplot(1,3,3,sharex=ax,sharey=ax)
    pylab.imshow(C-Cm,interpolation="nearest")
    pylab.draw()
    pylab.show(block=False)
    return

    # pylab.clf()
    # pylab.imshow(sqrtCs,interpolation="nearest")
    # pylab.draw()
    # pylab.show(False)
    
    sqrtCss=toSparse(sqrtCs)
    sC=sqrtCss.toarray()
    T.timeit(0)

    for i in range(10):
        v=np.random.randn(M).reshape((-1,1))
        Im=sqrtCss.dot(v).reshape((N,N))


        pylab.clf()
        pylab.imshow(Im,interpolation="nearest")
        pylab.draw()
        pylab.show(False)
        pylab.pause(0.5)
    T.timeit("s")
    
    # np.dot(sC,v)
    # T.timeit("nos")
    

    return sqrtCss
    
