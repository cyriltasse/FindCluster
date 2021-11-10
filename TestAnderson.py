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



def testDist():
    pylab.close("all")
    np.random.seed(43)
    Ln=np.int64(10**(np.linspace(1,4,5)))
    Ln=np.int64(10**(np.linspace(1,3,2)))
    #Ln=[10,100]
    cmap=pylab.cm.jet
    Lc=cmap(np.linspace(0.1,0.9,len(Ln)))[::-1]
    n=50
    for i_n,n in enumerate(Ln):
        print("doing n=%i"%n)
        CAD=ClassAndersonDarlingMachine()
        CAD.Init(n,NTry=3000)
import ModIrregular
def testDiffTerm():
    X=np.random.randn(20)+0.3#*2+1
    CAD=ClassAndersonDarlingMachine()
    CAD.Init(X.size,NTry=1000)

    def D(x):
        F_X=ModIrregular.giveIrregularCumulDist(x)
        Diff=(CAD.w*(F_X-CAD.f_Phi)**2)#.int()
        Diff=(F_X)
        #Diff=(F_X)
        return Diff

    pylab.close("all")
    fig=pylab.figure("Diff Term")
    dx=1e-3
    Diff0=D(X)
    for iParm in range(X.size):
        Xix=X.copy()
        Xix[iParm]+=dx
        Diff=D(Xix)
        dDiff=Diff-Diff0
        #pylab.plot(dDiff.x,X.size*dDiff.y/dx)
        pylab.plot(dDiff.x,dDiff.y/dx,marker=".")
        pylab.draw()
        pylab.show(False)
        pylab.pause(0.1)
        return
        
import ClassEigenShapiroWilk

def testJacob():
    np.random.seed(43)
    X0=np.random.randn(20)+0.3#*2+1
    X=X0.copy()

    CAD=ClassAndersonDarlingMachine()
    CAD.Init(X.size,NTry=1000)
    #CAD=ClassEigenShapiroWilk.ClassEigenShapiroWilk()

    fig=pylab.figure("Jacob")
    J2a=CAD.meas_dA2_dx(X)
    #J2b,J2b1=CAD.give_dA2_dx(X)
    J2b=CAD.dA2_dx(X)

    pylab.close("all")

    # ########################################
    # ##################### A2 ###############
    fig=pylab.figure("Jacob A2")
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

    #Ha=H_Diag_Meas=CAD.meas_d2A2_dx2(X)
    H_Meas_Full=CAD.meas_d2A2_dx2_full(X)
    Ha=H_Meas_Full_Diag=np.diag(H_Meas_Full)
    
    Hb0=CAD.d2A2_dx2(X,Diag=True)
    H_Calc_Full=CAD.d2A2_dx2(X,Diag=False)
    Hb1=H_Calc_Full_Diag=np.diag(H_Calc_Full)
    
    fig=pylab.figure("Hessian A2")
    pylab.clf()

    ax0=pylab.subplot(2,2,1)
    ax0.scatter(X,np.abs(Ha))
    ax0.set_yscale("log")
    ax0.set_title("Meas")

    ax1=pylab.subplot(2,2,2,sharex=ax0,sharey=ax0)
    ax1.scatter(X,np.abs(Hb0))
    ax1.set_yscale("log")
    ax1.set_title("Calc")

    ax2=pylab.subplot(2,2,3,sharex=ax0,sharey=ax0)
    ax2.scatter(X,np.abs(Ha),c="blue")
    ax2.scatter(X,np.abs(Hb0),c="black")
    ax2.scatter(X,np.abs(Hb1),c="gray")
    ax2.set_yscale("log")
    ax3=pylab.subplot(2,2,4,sharex=ax0)
    ax3.scatter(X,(Ha)/(Hb0),c="black")
    ax3.scatter(X,(Ha)/(Hb1),c="gray")
    pylab.draw()
    pylab.show(block=False)
    pylab.pause(0.1)

    fig=pylab.figure("HessianA2 Full")
    fig.clf()
    pylab.subplot(1,2,1)
    pylab.imshow(np.log10(np.abs(H_Calc_Full)),interpolation="nearest")
    pylab.subplot(1,2,2)
    pylab.imshow(np.log10(np.abs(H_Meas_Full)),interpolation="nearest")
    pylab.draw()
    pylab.show(block=False)
    pylab.pause(0.1)
    return
    # ########################################
    # ##################### logP #############
    
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
    ax3.scatter(X,(Ha)/(Hb),c="black")
    pylab.draw()
    pylab.show(block=False)
    pylab.pause(0.1)

    Ha=CAD.meas_d2logP_dx2(X)
    Hb0=CAD.d2logPdx2(X)
    Hbb=CAD.d2logPdx2_full(X)
    Hb1=np.diag(Hbb)
    
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
    ax2.scatter(X,np.abs(Ha),c="blue")
    ax2.scatter(X,np.abs(Hb0),c="black")
    ax2.scatter(X,np.abs(Hb1),c="gray")
    ax2.set_yscale("log")
    ax3=pylab.subplot(2,2,4,sharex=ax0)
    ax3.scatter(X,(Ha)/(Hb0),c="black")
    ax3.scatter(X,(Ha)/(Hb1),c="gray")
    pylab.draw()
    pylab.show(block=False)
    pylab.pause(0.1)


    H_Meas_Full=CAD.meas_d2logP_dx2_full(X)
    fig=pylab.figure("Hessian Full")
    fig.clf()
    pylab.subplot(1,2,1)
    pylab.imshow(np.log10(np.abs(Hbb)),interpolation="nearest")
    pylab.subplot(1,2,2)
    pylab.imshow(np.log10(np.abs(H_Meas_Full)),interpolation="nearest")
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
    # pylab.plot(CAD.dlogPdA2.x,CAD.dlogPdA2.y)
    # pylab.draw()
    # pylab.show(block=False)
    # pylab.pause(0.1)
    # return

    Alpha=np.float64(np.array([1]))
    L_A=[]
    L_A2=[]
    L_logP=[]
    iStep=0
    pylab.clf()
    
    CAD.Init(X.size,NTry=10000)
    
    while True:

        print("================ %i =============="%iStep)
        C=GeneDist.ClassDistMachine()
        A2=CAD.giveA2(X)
        logP=CAD.logP_A2(A2)
        L_A2.append(A2)
        L_logP.append(logP)
        print("A2   = %f"%A2)
        print("logP = %f"%logP)

        #J=CAD.give_dA2_dx(X)[1]
        #J=CAD.dA2_dx(X)
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
            if len(L_A2)>2:
                ax=pylab.subplot(2,2,2)
                #pylab.plot(np.log(L_A2),color="black")
                pylab.plot(L_A2,color="black")
                ax.set_yscale("log")
            pylab.subplot(2,2,3)
            pylab.plot(xJ,yJ,color="black")
            pylab.subplot(2,2,4)
            #pylab.plot(np.exp(L_logP),color="black")
            pylab.plot(CAD.empirical_PA2.x,CAD.empirical_PA2.y,color="black")
            pylab.plot(CAD.empirical_PA2.x,np.exp(CAD.logP_A2(CAD.empirical_PA2.x)),color="black",ls="--")
            pylab.scatter([A2],[np.exp(CAD.logP_A2(A2))])
            pylab.draw()
            pylab.show(block=False)
            pylab.pause(0.01)

        
        X1=X-Alpha*J
        if CAD.logP_A2(CAD.giveA2(X1))<CAD.logP_A2(CAD.giveA2(X)):
#        if CAD.giveA2(X1)>CAD.giveA2(X):
            Alpha/=1.5
        else:
            X=X1
            print(CAD.logP_A2(CAD.giveA2(X)))

            
        if iStep%10==0:
            Alpha*=1.5
        print("Alpha=%f"%Alpha)
            
        iStep+=1
    return
