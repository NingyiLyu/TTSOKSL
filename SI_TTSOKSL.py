#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 21:48:38 2021

@author: ningyi
"""
import numpy as np
import tt
import tt.ksl
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from time import process_time
t_start=process_time()
qmodes=0
nsc = 17001             # number of propagation steps
tau = 2.5              # propagation time step
gam = 0.001            # phenomeological dumping factor for simulating homogeneous broadening
eps = 1e-14            # tt approx error
rma = 2000000                # max tt rank
eshift = 2.48          # energy shift for PESs
dim = 2+qmodes         # number of coords
nstates=2              # number of surfaces
Vb1 = 0                # classical bath energy PES1
Vb2 = 0                # classical bath energy PES2
d = 8
n = [2**8,2**5,2**5]               # number or grid points
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
    if i==0:
        nx[i]=n[0]
        dx[i]=L[i]/nx[i]
        dp[i]=2.0*np.pi/L[i]
    elif i==dim-1:
        nx[i] = n[2]                      # number of grid points
        dx[i] = L[i]/nx[i]             # coord grid spacing
        dp[i] = 2.0*np.pi/L[i]         # momenta grid spacing
    else: 
        nx[i] = n[1]                      # number of grid points
        dx[i] = L[i]/nx[i]             # coord grid spacing
        dp[i] = 2.0*np.pi/L[i]  
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
x=np.zeros(nx[0])
y=np.zeros(nx[1])
z=np.zeros(nx[dim-1])
for j in range(nx[0]):
    x[j]=dx[0]*(j+1-nx[0]/2)
for j in range(nx[1]):
    y[j]=dx[1]*(j+1-nx[1]/2)
for j in range(nx[dim-1]):
    z[j]=dx[dim-1]*(j+1-nx[dim-1]/2)
nump=nx
px=np.zeros(nump[0])
py=np.zeros(nump[1])
pz=np.zeros(nump[dim-1])
for j in range(nump[0]):
    px[j]=dp[0]*(j+1-nump[0]/2)
for j in range(nump[1]):
    py[j]=dp[1]*(j+1-nump[1]/2)
for j in range(nump[dim-1]):
    pz[j]=dp[dim-1]*(j+1-nump[dim-1]/2)

wj=[792.8,842.8,866.2,882.4,970.3,976.0,997.0,1017.1,1089.6,1189.0,\
            1214.7,1238.1,1267.9,1317.0,1359.0,1389.0,1428.4,1434.9,1451.8,\
            1572.8,1612.1,1629.2,1659.1]
cj=[0.175,0.2,0.175,0.225,0.55,0.3,0.33,0.45,0.125,0.175,0.44,0.5,0.475,\
            0.238,0.25,0.25,0.25,0.225,0.225,0.25,0.225,0.125,0.225]
for j in range(qmodes):
    wj[j]=wj[j]*4.5563353E-6
    cj[j]=cj[j]*wj[j]
for j in range(qmodes):          # Parameters for bath harmonic modes
    om[j+2]=wj[j]
    m[j+2]=1.0/wj[j]
zem=0                            # quantum zero point energy
for j in range(qmodes):
    zem=zem+0.5*wj[j]

#TT-SOFT-KSL procedure. Essentially, this is changing exp(-i*V*tau/2)psi(x) to ksl(-i*V/2, psi(x), tau) 
#And changing exp(-i*T*tau)psi(p) to ksl(-i*T, psi(p), tau). 
def tt_soft_ksl(y0,Vm,Pxy):
    # soft propagation
    global nstates,eps,rma
    out=tt.ksl.ksl(Vm,y0,tau)
    fp=mfft(out,1)
    fp=tt.ksl.ksl(Pxy,fp,tau)
    out=mfft(fp,-1)
    out=tt.ksl.ksl(Vm,out,tau)
    return out

#mfft modified: no ft for the first core. 
def mfft(f,ind):
    # multidimensional fft of function f in tt format
    # ind=1 for fft, otherwise ifft
    global eps, rma
    y=f.to_list(f)                                 # get cores
    for k in range(1,len(y)):                        # core index
        for i in range(y[k].shape[0]):             # left index of core k
            for j in range(y[k].shape[2]):         # right index of core k
                if ind == 1:
                    y[k][i,:,j] = np.fft.fft(y[k][i,:,j])
#*4/n
                else:
                    y[k][i,:,j] = np.fft.ifft(y[k][i,:,j])
#*n/4
    out=f.from_list(y)                             # assemble tt from updated cores 
    out=out.round(eps,rma)
    return out

def genlist(e1, e2, e3, i,dim,xone,oney,onez):
    # generator of tt list of coordinates
    if i == dim-1:
        w = onez
        for j in range(dim-2):
            w=tt.kron(e2,w)
        w=tt.kron(e1,w)
    elif i==0:
        w = xone
        for j in range(dim-2):
            w=tt.kron(w,e2)
        w=tt.kron(w,e3)
    else:
        w = oney
        for j in range(dim-2-i):
            w = tt.kron(w,e2)
        for j in range(1,i):
            w = tt.kron(e2,w)
        for j in range(1):
            w = tt.kron(e1,w)
        w = tt.kron(w,e3)
            
    return w

xv=np.fft.fftfreq(nx[0],1.0/(nx[0]*dx[0]))  
yv=np.fft.fftfreq(nx[1],1.0/(nx[1]*dx[1]))
zv=np.fft.fftfreq(nx[dim-1],1.0/(nx[dim-1]*dx[dim-1])) 
pxv=np.fft.fftfreq(nx[0],1.0/(nx[0]*dp[0])) 
pyv=np.fft.fftfreq(nx[1],1.0/(nx[1]*dp[1]))
pzv=np.fft.fftfreq(nx[dim-1],1.0/(nx[dim-1]*dp[dim-1]))
xs=np.fft.fftshift(xv)
ys=np.fft.fftshift(yv)
zs=np.fft.fftshift(zv)
x,y=np.meshgrid(xs, ys, sparse=False, indexing='ij')
px,py=np.meshgrid(pxv, pyv, sparse=False, indexing='ij')

# grids for building tensor trains
xone =  np.zeros(nx[0]*nx[1],dtype=float)
oney =  np.zeros(nx[1]*nx[1],dtype=float)
onez =  np.zeros(nx[dim-1]*nx[dim-1],dtype=float)
xone=xs
oney=ys
onez=zs
pxone=pxv
onepy=pyv
onepz=pzv
ronesprim=tt.ones(nx[0],1)
ronesseco=tt.ones(nx[1],1)
ronesthir=tt.ones(nx[dim-1],1)
ttxone=tt.tensor(xone)
ttoney=tt.tensor(oney)
ttonez=tt.tensor(onez)
ttpxone=tt.tensor(pxone)
ttonepy=tt.tensor(onepy)
ttonepz=tt.tensor(onepz)
tt_x = [genlist(ronesprim,ronesseco,ronesthir,i,dim,ttxone,ttoney,ttonez) for i in range(dim)]
tt_p = [genlist(ronesprim, ronesseco,ronesthir,i,dim,ttpxone,ttonepy,ttonepz) for i in range(dim)]


#PESs
def v1(r): 
    # PES 1
    V1=3.6/2.*(1.-np.cos(r[:,0]))-eshift
    out=V1+0.19/2.*r[:,1]**2
    out=out/27.2
    # add quantum bath when qmodes > 0
    for j in range(qmodes):
        out = out + 0.5*wj[j]*r[:,j+2]**2
    # add classical bath when cmodes > 0
    out = out + Vb1
    # substract quantum zero-point energy
    out=out-zem
    return out

def v2(r):
    # PES 2
    V2= 2.48-1.09/2.0*(1.0-np.cos(r[:,0]))-eshift
    out = V2+0.19/2.*r[:,1]**2+0.1*r[:,1]
    out=out/27.2
    # add quantum bath
    for j in range(qmodes):
        out = out + 0.5*wj[j]*r[:,j+2]**2+cj[j]*r[:,j+2]
    # add classical bath
    out = out + Vb2
    # substract quantum zero-point energy
    out=out-zem
    return out

def vc(r):
    # Coupling of PESs
    out = 0.19*r[:,1]
    out=out/27.2
    return out

def Up(p):
    # KE part of Trotter expansion
    global EYE,tau,m,dim
    out=0
    for j in range(dim):
        out = out + p[:,j]**2/(2*m[j])
    return out

tt_v1=tt.multifuncrs2(tt_x,v1)
tt_v1=tt_v1.round(1e-14)
tt_v2=tt.multifuncrs2(tt_x,v2)
tt_v2=tt_v2.round(1e-14)
tt_vc=tt.multifuncrs2(tt_x,vc)
tt_vc=tt_vc.round(1e-14)
tt_Pxy=tt.multifuncrs2(tt_p,Up)
tt_Pxy=tt_Pxy.round(1e-14)


def htrans(r):
    # Heaviside function for trans population
    global eps
    pi2=0.5*np.pi
    nevals, dim = r.shape
    out = np.zeros((nevals,))
    for ii in range(nevals):
        if(3*pi2 > np.abs(r[ii,0]) > pi2):
            out[ii]=1
        elif(7*pi2 > np.abs(r[ii,0]) > 5*pi2):
            out[ii]=1
    return out

def hcis(r):
    # Heaviside function for cis population
    global eps
    pi2=0.5*np.pi
    nevals, dim = r.shape
    out = np.zeros((nevals,))
    for ii in range(nevals):
        if(np.abs(r[ii,0]) < pi2):
            out[ii]=1
        elif(5*pi2 > np.abs(r[ii,0]) > pi2*3):
            out[ii]=1
        elif(np.abs(r[ii,0]) > pi2*7):
            out[ii]=1
    return out

def psio(r):
    # initial Gaussian state
    global dim,ro,po,EYE,sig,eps
    out=0
    for j in range(dim):
        out = out + ((r[:,j]-ro[j])/sig[j])**2
    out=out+np.sum(EYE*(r-ro)*po,axis=1)
    out=np.exp(-out)*(2.0/np.pi)**(0.25*dim)
    for j in range(dim):
        out=out/np.sqrt(sig[j])
    return out
tt_psi1=tt.multifuncrs2(tt_x,psio)
tt_psi1=tt_psi1.round(1e-14)

#Construction of Vm
kin=tt.kron(tt.eye(nx[0],1),tt.eye(nx[1],dim-2))
kin=tt.kron(kin,tt.eye(nx[dim-1],1))*0
a11=np.array([[1,0],[0,0]])
tt_a11=tt.matrix(a11)
a21=np.array([[0,0],[1,0]])
tt_a21=tt.matrix(a21)
a12=np.array([[0,1],[0,0]])
tt_a12=tt.matrix(a12)
a22=np.array([[0,0],[0,1]])
tt_a22=tt.matrix(a22)
P1=tt.diag(tt_v1)
P2=tt.diag(tt_v2)
Pc=tt.diag(tt_vc)
V1=kin+P2
V2=kin+P1
V11=tt.kron(tt_a11,V1)
V12=tt.diag(tt_vc)
V12=tt.kron(tt_a12,V12)
V21=tt.diag(tt_vc)
V21=tt.kron(tt_a21,V21)
V22=V2
V22=tt.kron(tt_a22,V22)
V=V11+V12+V21+V22
A=-EYE*V
Vm=A*0.5
Vm=Vm.round(1e-10)

#Construction of T
Tp=tt.diag(tt_Pxy)
T11=tt.kron(tt_a11,Tp)
T22=tt.kron(tt_a22,Tp)
T=T11+T22
T=-EYE*T
T=T.round(1e-10)
su=tt.tensor(np.array([1,0]))
sd=tt.tensor(np.array([0,1]))
sm=tt.tensor(np.array([1,1]))

#Initial state
y0=tt.kron(su,tt_psi1)
yinit=y0
#All-zero arrays for output 
t= np.arange(0,tau*(nsc),tau)
p1 = np.zeros((nsc))
p2 = np.zeros((nsc))
pcisS0 = np.zeros((nsc))
pcisS1 = np.zeros((nsc))
ptransS0 = np.zeros((nsc))
ptransS1 = np.zeros((nsc))
population=np.zeros((nsc))
r2=np.zeros((nsc))
sa=np.zeros((nsc))
r1=np.zeros((nsc))
rt=np.zeros((nsc),dtype=complex)
pr = np.empty_like(t)
#heaviside functions for population curves
tt_heav1=tt.kron(su,tt.ones(nx[0],dim))
tt_heav2=tt.kron(sd,tt.ones(nx[0],dim))
tt_trans=tt.multifuncrs2(tt_x, htrans,verb=0)
tt_trans=tt_trans.round(1e-14)
tt_trans=tt.kron(sm,tt_trans)
tt_cis=tt.multifuncrs2(tt_x,hcis,verb=0)
tt_cis=tt_cis.round(1e-14)
tt_cis=tt.kron(sm,tt_cis)

# Rank-adaptive algorithm
overlap=0
for k in range(nsc):
    if k>0:
        if y0.r[2]<30:#cap max rank
            #rank adaptive
            yold=y0
            tt_rand=tt.rand(yold.n,yold.d,1)
            tt_rand=tt_rand*tt_rand.norm()**(-1)
            tt_rand=tt_rand*1e-10
            ynew=yold+tt_rand
            ynew=ynew*ynew.norm()**(-1)*np.sqrt(1/ddx)
            yold=tt_soft_ksl(yold,Vm,T)
            ynew=tt_soft_ksl(ynew,Vm,T)
            overlap=np.abs(tt.dot(yold,ynew))*ddx
            if np.abs(overlap-1)<2e-6:
                y0=yold
            else:
                y0=ynew
            print(y0.r)
            print(k*tau*0.00002418884254)
        else:
            y0=tt_soft_ksl(y0,Vm,T)
            print(k*tau*0.00002418884254)
    pcisS1[k]=np.abs(tt.dot(y0*tt_heav1*tt_cis,y0*tt_heav1*tt_cis))*ddx
    ptransS1[k]=np.abs(tt.dot(y0*tt_heav1*tt_trans,y0*tt_heav1*tt_trans))*ddx
    pcisS0[k]=np.abs(tt.dot(y0*tt_heav2*tt_cis,y0*tt_heav2*tt_cis))*ddx
    ptransS0[k]=np.abs(tt.dot(y0*tt_heav2*tt_trans,y0*tt_heav2*tt_trans))*ddx
    population[k]=pcisS1[k]+ptransS1[k]+pcisS0[k]+ptransS0[k]
    sa[k]=overlap
    r2[k]=y0.r[2]
    rt[k]=tt.dot(y0,yinit)*ddx
ptrans=ptransS0+ptransS1
pGS=ptransS0+pcisS0
t0=t*0.00002418884254
plt.xlim(0.,1.)    
plt.ylim(0.,.8)            
plt.xlabel('time(ps)')
plt.ylabel('Populations')
plt.plot(t0,ptrans,'r',label='trans')
plt.plot(t0,pGS,'b',label='GS')                     
plt.legend()    

t_stop=process_time()
print(t_stop-t_start)
            
            
            
            
            
            
            
            
            
            
            
            
            
            