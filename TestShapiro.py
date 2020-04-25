# import scipy.special
# import numpy as np
# import pylab
# import scipy.stats
# import GeneDist
# import scipy.optimize
# from ModIrregular import *
# from scipy.special import gamma
# from scipy.special import loggamma

# erf=scipy.special.erf
# G=scipy.special.gamma
# logG=scipy.special.loggamma
# Phi=lambda x: 0.5*(1+erf(x/np.sqrt(2)))
# def Sigmoid(x,a=1):
#     return 1./(1+np.exp(-x/a))

from ClassAndersonDarling import *

# # ##############################
# # Catch numpy warning
# np.seterr(all='raise')
# import warnings
# warnings.filterwarnings('error')
# #with warnings.catch_warnings():
# #    warnings.filterwarnings('error')
# # ##############################
from ClassShapiroWilk import *


def testDist():
    pylab.close("all")
    np.random.seed(43)
    Ln=np.int64(10**(np.linspace(1,3,5)))
    #Ln=np.int64(10**(np.linspace(1,3,2)))
    Ln=[7]
    Ln=[17]
    Ln=[200]
    cmap=pylab.cm.jet
    Lc=cmap(np.linspace(0.1,0.9,len(Ln)))[::-1]
    for i_n,n in enumerate(Ln):
        print("doing n=%i"%n)
        CAD=ClassShapiroWilk()
        # CAD.generateOrderStats(n)
        CAD.Init(n,NTry=1000)
        #CAD.generatePW(n,NTry=30000,UseScipy=True)



    
def testJacob():
    pylab.close("all")
    np.random.seed(43)
    X0=np.random.randn(10)#*2+1
    X=X0.copy()

    CAD=ClassShapiroWilk()
    CAD.Init(X.size)
    
    J2a=CAD.meas_dW_dx(X)
    #J2b,J2b1=CAD.give_dW_dx(X)
    J2b=CAD.dW_dx(X)


    # ########################################
    # ##################### W ###############
    fig=pylab.figure("Jacob W")
    pylab.clf()
    ax=pylab.subplot(2,2,1)
    pylab.scatter(X,J2a)
    pylab.title("Meas")
    pylab.subplot(2,2,2)
    pylab.scatter(X,J2b)
    pylab.title("Calc")
    pylab.subplot(2,2,3)
    pylab.scatter(X,J2a,c="black")
    pylab.scatter(X,J2b,c="blue")
    #pylab.scatter(X,J2b1,c="green")
    pylab.subplot(2,2,4)
    pylab.scatter(X,J2a/J2b)
    pylab.draw()
    pylab.show(block=False)
    pylab.pause(0.1)

    Ha=CAD.meas_d2W_dx2(X)
    print(Ha)
    Hb=CAD.d2W_dx2(X)

    fig=pylab.figure("Hessian W")
    pylab.clf()

    ax0=pylab.subplot(2,2,1)
    ax0.scatter(X,np.abs(Ha))
    #ax0.set_yscale("log")
    ax0.set_title("Meas")

    ax1=pylab.subplot(2,2,2,sharex=ax0,sharey=ax0)
    ax1.scatter(X,np.abs(Hb))
    #ax1.set_yscale("log")
    ax1.set_title("Calc")

    ax2=pylab.subplot(2,2,3,sharex=ax0,sharey=ax0)
    ax2.scatter(X,np.abs(Ha),c="black")
    ax2.scatter(X,np.abs(Hb),c="blue")
    #ax2.set_yscale("log")
    ax3=pylab.subplot(2,2,4,sharex=ax0)
    ax3.scatter(X,np.abs(Ha)/np.abs(Hb),c="black")
    pylab.draw()
    pylab.show(block=False)
    pylab.pause(0.1)

    
    # ########################################
    # ##################### logP #############
    CAD.Init(X.size,NTry=1000)
    
    Ha=CAD.meas_dlogP_dx(X)
    Hb=CAD.dlogPdx(X)
    
    fig=pylab.figure("Jacob logP")
    pylab.clf()
    ax0=pylab.subplot(2,2,1)
    ax0.scatter(X,np.abs(Ha))
    #ax0.set_yscale("log")
    ax0.set_title("Meas")
    ax1=pylab.subplot(2,2,2,sharex=ax0,sharey=ax0)
    ax1.scatter(X,np.abs(Hb))
    ax1.set_yscale("log")
    ax1.set_title("Calc")
    ax2=pylab.subplot(2,2,3,sharex=ax0,sharey=ax0)
    ax2.scatter(X,np.abs(Ha),c="black")
    ax2.scatter(X,np.abs(Hb),c="blue")
    ax2.set_yscale("log")
    ax3=pylab.subplot(2,2,4,sharex=ax0)
    ax3.scatter(X,np.abs(Ha)/np.abs(Hb),c="black")
    pylab.draw()
    pylab.show(block=False)
    pylab.pause(0.1)

    Ha=CAD.meas_d2logP_dx2(X)
    Hb=CAD.d2logPdx2(X)
    fig=pylab.figure("Hessian logP")
    pylab.clf()
    ax0=pylab.subplot(2,2,1)
    ax0.scatter(X,np.abs(Ha))
    #ax0.set_yscale("log")
    ax0.set_title("Meas")
    ax1=pylab.subplot(2,2,2,sharex=ax0,sharey=ax0)
    ax1.scatter(X,np.abs(Hb))
    ax1.set_yscale("log")
    ax1.set_title("Calc")
    ax2=pylab.subplot(2,2,3,sharex=ax0,sharey=ax0)
    ax2.scatter(X,np.abs(Ha),c="black")
    ax2.scatter(X,np.abs(Hb),c="blue")
    ax2.set_yscale("log")
    ax3=pylab.subplot(2,2,4,sharex=ax0)
    ax3.scatter(X,np.abs(Ha)/np.abs(Hb),c="black")
    pylab.draw()
    pylab.show(block=False)
    pylab.pause(0.1)


    
def testMin():
    np.random.seed(43)
    X0=np.float64(np.random.rand(50)-1)*2
    X0=np.random.rand(5)*0.01
    X0=np.random.randn(100)*2.+1#*0.1#+1#*0.1+2
    #X0=np.linspace(-0.2,0.2,3)
    #X0=np.concatenate([X0,np.random.rand(50)-3])
    # X0=np.float64(np.random.rand(50))
    # X0*=4
    # X0[0]=-4
    # X0[1]=-3
    X=X0.copy()

    CAD=ClassAndersonDarlingMachine()
    
    # pylab.clf()
    # ax=pylab.subplot(1,2,1)
    # pylab.plot(CAD.logP.x,CAD.logP.y)
    # pylab.subplot(1,2,2,sharex=ax)
    # pylab.plot(CAD.dlogPdW.x,CAD.dlogPdW.y)
    # pylab.draw()
    # pylab.show(block=False)
    # pylab.pause(0.1)
    # return

    Alpha=np.float64(np.array([1]))
    L_A=[]
    L_W=[]
    L_logP=[]
    iStep=0
    pylab.clf()
    
    CAD.Init(X.size,NTry=10000)
    
    while True:

        print("================ %i =============="%iStep)
        C=GeneDist.ClassDistMachine()
        W=CAD.giveW(X)
        logP=CAD.logP_W(W)
        L_W.append(W)
        L_logP.append(logP)
        print("W   = %f"%W)
        print("logP = %f"%logP)

        #J=CAD.give_dW_dx(X)[1]
        #J=CAD.dW_dx(X)
        J=-CAD.dlogPdx(X)

        if iStep%10==0:
            x,y=C.giveCumulDist(X,Ns=1000,Norm=True,xmm=(-10,10))
            xJ,yJ=C.giveCumulDist(J,Ns=1000,Norm=True)
            x0,y0=C.giveCumulDist(X0,Ns=1000,Norm=True,xmm=(-10,10))
            pylab.clf()
            pylab.subplot(2,2,1)
            pylab.plot(x,y,color="black")
            pylab.plot(x,Phi(x),color="black",ls=":")
            pylab.plot(x0,y0,color="black",ls="--")
            if len(L_W)>2:
                ax=pylab.subplot(2,2,2)
                #pylab.plot(np.log(L_W),color="black")
                pylab.plot(L_W,color="black")
                ax.set_yscale("log")
            pylab.subplot(2,2,3)
            pylab.plot(xJ,yJ,color="black")
            pylab.subplot(2,2,4)
            #pylab.plot(np.exp(L_logP),color="black")
            pylab.plot(CAD.empirical_PW.x,CAD.empirical_PW.y,color="black")
            pylab.plot(CAD.empirical_PW.x,np.exp(CAD.logP_W(CAD.empirical_PW.x)),color="black",ls="--")
            pylab.scatter([W],[np.exp(CAD.logP_W(W))])
            pylab.draw()
            pylab.show(block=False)
            pylab.pause(0.01)

        
        X1=X-Alpha*J
        if CAD.logP_W(CAD.giveW(X1))<CAD.logP_W(CAD.giveW(X)):
#        if CAD.giveW(X1)>CAD.giveW(X):
            Alpha/=1.5
        else:
            X=X1
            print(CAD.logP_W(CAD.giveW(X)))

            
        if iStep%10==0:
            Alpha*=1.5
        print("Alpha=%f"%Alpha)
            
        iStep+=1
    return
