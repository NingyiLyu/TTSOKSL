#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 18 14:46:51 2021

@author: ningyi
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from numpy import *
import pylab
import math
import tt
from numpy import linalg as LA
import time
start_time=time.time()

def parameters():
    global nstates,n,nx,EYE,tau,eps,rma,dim,rd,ro,po
    global dx,dp,nsc,gam,d,ddx
    global sig,m,om,eshift
    global wj,cj,qmodes,wfflag,Vb1,Vb2,aflag

    aflag=0                # flag for visualization of adiabatic populations
    qmodes=0              # qmodes = 0 for TDSCF simulation
    wfflag = 0             # wfflag = 1 to visualize wavepacket (if qmodes > 1, edit lines with ttpsi2[:,:,nsl])
    tau = 2.5              # propagation time step
    nsc = 17001            # number of propagation steps
    gam = 0.001            # phenomeological dumping factor for simulating homogeneous broadening
    eps = 1e-14            # tt approx error
    rma = 30                # max tt rank
    eshift = 2.48          # energy shift for PES
    dim = 2+qmodes         # number of coords
    nstates=2              # number of surfaces
    Vb1 = 0                # classical bath energy PES1
    Vb2 = 0                # classical bath energy PES2
    d = 8
    n = 2**d               # number or grid points
    Lx = 2*np.pi           # box size x
    Ly = 10.0              # box size y
    L = np.ones((dim))*Ly  # Box width
    L[0] = Lx
    ro=np.zeros(dim,dtype=float)       # initial wavepacket position
    po=np.ones(dim,dtype=float)*0.     # initial wavepacket momentum
    nx=np.zeros(dim,dtype=int)         # number of grid points per dimension
    dx=np.zeros(dim,dtype=float)       # grid point spacing
    dp=np.zeros(dim,dtype=float)       # momentum grid point spacing
    for i in range(dim):
        nx[i] = n                      # number of grid points
        dx[i] = L[i]/nx[i]             # coord grid spacing
        dp[i] = 2.0*np.pi/L[i]         # momenta grid spacing
    ddx=1.0
    for i in range(dim):
        ddx=ddx*dx[i]
    EYE = complex(0,1)                 # imaginary number
    m = np.ones((dim))                 # masses
    om = np.ones((dim))                # frequencies
    sig = np.ones((dim))*np.sqrt(2.0)  # Gaussian widths

    # Parameters for the first 2 modes (large amplitude modes of reaction surface, theta and x_str) 
    sig[0] = 0.15228275
    m[0] = 56198.347
    om[0] = 2.0/sig[0]**2/m[0]
    m[1] = 143.158
    om[1] = 1.0/m[1]
    return()

def bath_parameters():
    # Harmonic bath parameters for rhodopsin
    global wj,cj,qmodes,zem,cmodes
    global xc,pc,SACT,xxi,xxio,om,m
    cmodes=23
    if qmodes > 0:
        cmodes=0
    wj=[792.8,842.8,866.2,882.4,970.3,976.0,997.0,1017.1,1089.6,1189.0,\
            1214.7,1238.1,1267.9,1317.0,1359.0,1389.0,1428.4,1434.9,1451.8,\
            1572.8,1612.1,1629.2,1659.1]
    cj=[0.175,0.2,0.175,0.225,0.55,0.3,0.33,0.45,0.125,0.175,0.44,0.5,0.475,\
            0.238,0.25,0.25,0.25,0.225,0.225,0.25,0.225,0.125,0.225]

    xc=np.zeros(cmodes,dtype=float)  # classical bath coords
    pc=np.zeros(cmodes,dtype=float)  # classical bath momenta
    SACT=0                           # classical action
    xxi=1                            # survival amplitude for classical harmonic bath
    xxio=xxi                         # copy
    for j in range(23):
        wj[j]=wj[j]*4.5563353E-6
        cj[j]=cj[j]*wj[j]
    for j in range(qmodes):          # Parameters for bath harmonic modes
        om[j+2]=wj[j]
        m[j+2]=1.0/wj[j]
    zem=0                            # quantum zero point energy
    for j in range(qmodes):
        zem=zem+0.5*wj[j]

    return ()

def bath_propagation(tt_ps):
    # classical bath
    global wj,cj,zem,cmodes,tau
    global xc,pc,ddx,Vb1,Vb2,xxi,xxio,EYE,SACT
    Vb1=0.0
    Vb2=0.0
    xxio=xxi
    xxi=1
    aj = np.real(np.sum(np.conj(tt_ps)*tt_ps))*ddx
    # Velocity Verlet propagation of bath coords and momenta, action and classical bath survival amplitude
    for j in range(cmodes):
        Vb1=Vb1+0.5*wj[j]*(xc[j]**2+pc[j]**2)
        Vb2=Vb2+0.5*wj[j]*(xc[j]**2+pc[j]**2)+cj[j]*xc[j]
        force=-xc[j]*wj[j]-aj*cj[j]
        xc[j]=xc[j]+wj[j]*pc[j]*tau+0.5*tau**2*force*wj[j]
        force2=-xc[j]*wj[j]-aj*cj[j]
        pc[j]=pc[j]+tau*0.5*(force+force2)
        xxi=xxi*exp(-(pc[j]**2+xc[j]**2+2.0*EYE*pc[j]*xc[j])/4.0)
        SACT=SACT+tau*pc[j]**2*wj[j]
    xxi=xxi*exp(EYE*SACT)
    return ()

def expAg(A,e):
    # Taylor expansion of exp(A), with A an nstates x nstates matrix valued tensor train
    global nstates
    N=10
    w0=A*(1.0/2**N)
    tm=e
    k=N-1
    while k > 0:
        prod=e*0.0
        for j in range(nstates):
            for i in range(nstates):
                 for kk in range(nstates):
                     prod[j,i]=prod[j,i]+tm[j,kk]*w0[kk,i]*(1.0/k)
        tm=e+prod
        for j in range(nstates):
            for i in range(nstates):
                tm[j,i]=tm[j,i]
        k=k-1
    while k < N:
        prod=e*0.0
        for j in range(nstates):
            for i in range(nstates):
                 for kk in range(nstates):
                     prod[j,i]=prod[j,i]+tm[j,kk]*tm[kk,i]
        for j in range(nstates):
            for i in range(nstates):
                tm[j,i]=prod[j,i]
        k=k+1
    return tm

def mv22(emat,psi):
    # nstates x nstates matrix times vector valued full arrays
    global nstates
    out=[]
    for j in range(nstates):
        out.append(gridzeros)
        for k in range(nstates):
            out[j] = out[j] + emat[j][k]*psi[k]
    return out

def soft(fxy,emat,Pxy):
    # soft propagation
    global nstates,eps,rma
    out=mv22(emat,fxy)
    for j in range(nstates):
        fp=np.fft.fft2(out[j])*Pxy
        out[j]=np.fft.ifft2(fp)
    out=mv22(emat,out)
    return out

def v1(x,y): 
    # PES 1
    global wj,cj,qmodes,zem,Vb1
    V1=3.6/2.*(1.-np.cos(x))-eshift
    out=V1+0.19/2.*y**2
    out=out/27.2
    # add classical bath when cmodes > 0
    out = out + Vb1
    # substract quantum zero-point energy
    out=out-zem
    return out

def v2(x,y):
    # PES 2
    global wj,cj,qmodes,zem,Vb2
    V2= 2.48-1.09/2.0*(1.0-np.cos(x))-eshift
    out = V2+0.19/2.*y**2+0.1*y
    out=out/27.2
    # add classical bath
    out = out + Vb2
    # substract quantum zero-point energy
    out=out-zem
    return out

def vc(x,y):
    # Coupling of PESs
    out = 0.19*y
    out=out/27.2
    return out

def Up(px,py):
    global EYE,tau,m,dim
    out=0
    out=out+px**2/(2*m[0])
    out=out+py**2/(2*m[1])
    out=np.exp(-EYE*out*tau)
    return out

def psio(x,y):
    # initial Gaussian state
    global dim,ro,po,EYE,sig,eps
    out=0
    out = out + ((x-ro[0])/sig[0])**2 + ((y-ro[1])/sig[1])**2
    out=out+EYE*(x-ro[0])*po[0]+EYE*(y-ro[1])*po[1]
    out=np.exp(-out)*(2.0/np.pi)**(0.25*dim)
    for j in range(dim):
        out=out/np.sqrt(sig[j])
    return out


def htrans(x,y):
    # Heaviside function for trans population
    pi2=0.5*np.pi
    if (3*pi2 > np.abs(x) > pi2):
        out=1
    elif(7*pi2 > np.abs(x) > 5*pi2): 
        out=1
    else:
        out=0
    return out

def hcis(x,y):
    # Heaviside function for cis population
    pi2=0.5*np.pi
    if(np.abs(x) < pi2):
        out=1
    elif(5*pi2 > np.abs(x) > pi2*3):
        out=1
    elif(np.abs(x) > pi2*7):
        out=1
    else:
        out=0
    return out

def pop(psi1,heaviside):
    # Population defined by Heaviside function
    global ddx
    temp=heaviside*psi1*ddx
    out = np.sum(np.conj(psi1)*temp)
    return (np.real(out))

if __name__ == "__main__":
    global n,nx,EYE,tau,eps,rma,dim,ro,po,rd,wfflag,ddx,jind,aflag
    global dx,dp,m,om,nsc,Pxy,gam,d,xxio,qmodes,cmodes,eshift
    parameters()                      # read parameters
    bath_parameters()                 # initialize bath 
    xv=np.fft.fftfreq(nx[0],1.0/(nx[0]*dx[0]))  
    yv=np.fft.fftfreq(nx[1],1.0/(nx[1]*dx[1])) 
    pxv=np.fft.fftfreq(nx[0],1.0/(nx[0]*dp[0])) 
    pyv=np.fft.fftfreq(nx[1],1.0/(nx[1]*dp[1]))
    xs=np.fft.fftshift(xv)
    ys=np.fft.fftshift(yv)
    x,y=np.meshgrid(xs, ys, sparse=False, indexing='ij')
    px,py=np.meshgrid(pxv, pyv, sparse=False, indexing='ij')
    xgrid=x[:,0]
    ygrid=y[0,:]
    pxgrid=px[:,0]
    pygrid=py[0,:]
    v1grid=np.zeros((nx[0],nx[1]),dtype=complex)
    v2grid=np.zeros((nx[0],nx[1]),dtype=complex)
    vcgrid=np.zeros((nx[0],nx[1]),dtype=complex)
    Upgrid=np.zeros((nx[0],nx[1]),dtype=complex)
    psiogrid=np.zeros((nx[0],nx[1]),dtype=complex)
    transgrid=np.zeros((nx[0],nx[1]))
    cisgrid=np.zeros((nx[0],nx[1]))
    for i in range(nx[0]):
        for j in range(nx[1]):
            v1grid[i,j]=v1(xgrid[i],ygrid[j])
            v2grid[i,j]=v2(xgrid[i],ygrid[j])
            vcgrid[i,j]=vc(xgrid[i],ygrid[j])
            transgrid[i,j]=htrans(xgrid[i],ygrid[j])
            cisgrid[i,j]=hcis(xgrid[i],ygrid[j])
            psiogrid[i,j]=psio(xgrid[i],ygrid[j])
            Upgrid[i,j]=Up(pxgrid[i],pygrid[j])
    onem=[]
    gridVm=[]
    gridones=np.ones((nx[0],nx[1]),dtype=complex)
    gridzeros=gridones*0
    for i in range(nstates):
        for j in range(nstates):
            if i==j:
                onem.append(gridones)
                if i==1:
                    gridVm.append(v1grid)
                else:
                    gridVm.append(v2grid)
            else:
                onem.append(gridzeros)
                gridVm.append(vcgrid)
    onem=np.reshape(onem,(nstates,nstates,nx[0],nx[1]))
    gridVm=np.reshape(gridVm,(nstates,nstates,nx[0],nx[1]))
    gridUV=np.exp(gridVm*(-EYE*tau/2))
    psi1grid=psiogrid
    psi2grid=psiogrid*0
    psigrid=[]
    psigrid.append(psi1grid)
    psigrid.append(psi2grid)
    expUV=expAg(gridVm*(-EYE*tau/2),onem)
    ptransS0 = np.zeros((nsc))
    ptransS1 = np.zeros((nsc))  
    pcisS0 = np.zeros((nsc))  
    pcisS1 = np.zeros((nsc))
    population=np.zeros((nsc))
    rt = np.zeros((10*nsc),dtype=complex)  
    # array of times for visualization of survival amplitude
    au2ps=0.00002418884254 # conversion factor from au to ps
    t=np.linspace(0,10*nsc,10*nsc)*tau*au2ps
    # save a copy of initial state
    psi0grid=psigrid
    plt.figure(figsize=(7,5))
    
    if True:
    # Propagation loop
        for js in range(nsc):   
            if True:
                # compute survival amplitude rt=<psi0|psit>
                rr=0
#                for i in range(nstates):
                for i in range(1):
                    rr=rr+np.sum(psi0grid[i]*psigrid[i])*ddx
                rt[js]=rr
                ptransS1[js] = pop(psigrid[0],transgrid)
                ptransS0[js] = pop(psigrid[1],transgrid)
                pcisS1[js] = pop(psigrid[0],cisgrid)
                pcisS0[js] = pop(psigrid[1],cisgrid)
                ptot=ptransS1[js]+ptransS0[js]+pcisS1[js]+pcisS0[js]
                population[js]=ptot
                ptransS1[js]=ptransS1[js]/ptot
                ptransS0[js]=ptransS0[js]/ptot
                pcisS1[js]=pcisS1[js]/ptot
                pcisS0[js]=pcisS0[js]/ptot
                time1=t[js]
                print("time=",time1, "ptot=",ptot)
                rt[js]=rt[js]*xxio
                psigrid=soft(psigrid,expUV,Upgrid)
                        # plot wavepacket components              
                
        if True:
            # compute spectrum as the FT of rt
            dw=2.0*np.pi/(tau*10*nsc)
            w=np.fft.fftshift(np.fft.fftfreq(10*nsc,1.0/(10*nsc*dw)))
            rw=np.fft.fftshift(np.fft.ifft(rt*np.exp(-tau*t*gam))) 
            if wfflag == 1:
                ax= plt.subplot(3,2,5)
            else:
                ax= plt.subplot(3,1,3)
            ax.plot(w*27.2+eshift,np.real(rw),label='Absorption Spectrum')
            plt.legend()
            ax.set_xlim(2,3)
            ax.grid()
            plt.pause(.15)   

print("--- %s seconds ---" % (time.time() - start_time))
ptrans=ptransS0+ptransS1
pcis=pcisS0+pcisS1
pGS=ptransS0+pcisS0
t0=t[:nsc]
plt.xlim(0.,1.)    
plt.ylim(0.,.8)             
plt.xlabel('time(ps)')
plt.ylabel('Populations')
plt.plot(t0,ptrans,'r',label='trans')
plt.plot(t0,pGS,'b',label='GS')                      
plt.legend()           
                





















                
                
                
                
