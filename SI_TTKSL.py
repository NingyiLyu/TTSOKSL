#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 18 07:25:04 2021

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


# bath mode parameters
wj=[792.8,842.8,866.2,882.4,970.3,976.0,997.0,1017.1,1089.6,1189.0,\
            1214.7,1238.1,1267.9,1317.0,1359.0,1389.0,1428.4,1434.9,1451.8,\
            1572.8,1612.1,1629.2,1659.1,8.26551627,   16.70543968,   25.32728936,   34.13908143,\
43.14937377,   52.36731596,   61.8027052 ,   71.46604877,\
81.36863449,   91.52260991,  101.94107191,  112.63816832,\
123.6292134 ,  134.93081973,  146.56104925,  158.53958689,\
170.88794109,  183.62967617,  196.79068302,  210.39949584,\
224.48766465,  239.09019589,  254.24607645,  269.99890106,\
286.39762824,  303.49749807,  321.36115492,  340.06003291,\
359.67608134,  380.30393545,  402.05367795,  425.05439557,\
449.45882178,  475.44949029,  503.24703017,  533.12156457,\
565.40871997,  600.53268244,  639.04038535,  681.65397414,\
729.35469219,  783.52388141,  846.19522775,  920.54366198,\
1011.94029409, 1130.60409729, 1300.07027371, 1600.     ]
cj=[0.175,0.2,0.175,0.225,0.55,0.3,0.33,0.45,0.125,0.175,0.44,0.5,0.475,\
            0.238,0.25,0.25,0.25,0.225,0.225,0.25,0.225,0.125,0.225,2.24515007, 1.57925192, 1.28258457, 1.10472569, 0.98263702,\
0.89196992, 0.82106327, 0.76353769, 0.71556955, 0.67470848,\
0.63930157, 0.60818762, 0.58052351, 0.55568003, 0.53317661,\
0.51263884, 0.49376995, 0.4763311 , 0.46012741, 0.44499796,\
0.43080836, 0.41744521, 0.40481187, 0.39282523, 0.38141315,\
0.37051245, 0.3600673 , 0.35002783, 0.34034909, 0.33099007,\
0.32191288, 0.31308201, 0.30446362, 0.29602485, 0.28773306,\
0.27955499, 0.27145579, 0.26339774, 0.25533845, 0.24722841,\
0.2390072 , 0.23059732, 0.22189376, 0.21274442, 0.20290975,\
0.19196634, 0.1790181 , 0.1613691]
for j in range(71):
    wj[j]=wj[j]*4.5563353E-6
    cj[j]=cj[j]*wj[j]
for j in range(qmodes):          # Parameters for bath harmonic modes
    om[j+2]=wj[j]
    m[j+2]=1.0/wj[j]
zem=0                            # quantum zero point energy
for j in range(qmodes):
    zem=zem+0.5*wj[j]

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

xone =  np.zeros(nx[0]*nx[1],dtype=float)
oney =  np.zeros(nx[1]*nx[1],dtype=float)
onez =  np.zeros(nx[dim-1]*nx[dim-1],dtype=float)

xone=x
oney=y
onez=z
ronesprim=tt.ones(nx[0],1)
ronesseco=tt.ones(nx[1],1)
ronesthir=tt.ones(nx[dim-1],1)
ttxone=tt.tensor(xone)
ttoney=tt.tensor(oney)
ttonez=tt.tensor(onez)
tt_x = [genlist(ronesprim,ronesseco,ronesthir,i,dim,ttxone,ttoney,ttonez) for i in range(dim)]

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

#TT-PESs
tt_v1=tt.multifuncrs2(tt_x,v1)
tt_v1=tt_v1.round(1e-14)
tt_v2=tt.multifuncrs2(tt_x,v2)
tt_v2=tt_v2.round(1e-14)
tt_vc=tt.multifuncrs2(tt_x,vc)
tt_vc=tt_vc.round(1e-14)


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

#Fourier grid hamiltonian used to express kinetic operator in position space. 
#See page 24-25 of http://ursula.chem.yale.edu/~batista/classes/vvv/v570.pdf 
#for detailed derivation and examples of 
Txgrid=np.zeros((nx[0],nx[0]),dtype=complex)
Tygrid=np.zeros((nx[1],nx[1]),dtype=complex)
Tzgrid=np.zeros((nx[dim-1],nx[dim-1]),dtype=complex)
for l in range(nx[0]):
    for j in range(nx[0]):
        for k in range(nump[0]):
            Txgrid[l,j]=Txgrid[l,j]+np.exp(EYE*(x[j]-x[l])*px[k])*px[k]**2/2/m[0]*dx[0]*dp[0]/2/np.pi
for l in range(nx[1]):
    for j in range(nx[1]):
        for k in range(nump[1]):
            Tygrid[l,j]=Tygrid[l,j]+np.exp(EYE*(y[j]-y[l])*py[k])*py[k]**2/2/m[1]*dx[1]*dp[1]/2/np.pi
for l in range(nx[dim-1]):
    for j in range(nx[dim-1]):
        for k in range(nump[dim-1]):
            Tzgrid[l,j]=Tzgrid[l,j]+np.exp(EYE*(z[j]-z[l])*pz[k])*pz[k]**2/2/m[dim-1]*dx[dim-1]*dp[dim-1]/2/np.pi
Kinx=tt.matrix(Txgrid)
Kiny=tt.matrix(Tygrid)
Kinz=tt.matrix(Tzgrid)

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

#Build kinetic operator as TT-matrix
kin=tt.kron(tt.eye(nx[0],1),tt.eye(nx[1],dim-2))
kin=tt.kron(kin,tt.eye(nx[dim-1],1))*0

i=0
while i < dim:
    if i == 0:
        tmp=tt.kron(Kinx,tt.eye(nx[1],dim-2))
        tmp=tt.kron(tmp,tt.eye(nx[dim-1],1))
    else:
        if i == dim-1:
            tmp=tt.kron(tt.eye(nx[1],i-1),Kinz)
            tmp=tt.kron(tt.eye(nx[0],1),tmp)
        else:
            tmp=tt.kron(tt.eye(nx[1],i-1),Kiny*(m[1]/m[i]))
            tmp=tt.kron(tmp,tt.eye(nx[1],dim-i-2))
            tmp=tt.kron(tmp,tt.eye(nx[dim-1],1))
            tmp=tt.kron(tt.eye(nx[0],1),tmp)
    kin=kin+tmp
    kin=kin.round(eps)
    i=i+1
    
#Build hamiltonian as TT-matrix
#used to include two states in one tt array
a11=np.array([[1,0],[0,0]])
tt_a11=tt.matrix(a11)
a21=np.array([[0,0],[1,0]])
tt_a21=tt.matrix(a21)
a12=np.array([[0,1],[0,0]])
tt_a12=tt.matrix(a12)
a22=np.array([[0,0],[0,1]])
tt_a22=tt.matrix(a22)
#Potential operator as diagonal TT-matrix in position representation
P1=tt.diag(tt_v1)
P2=tt.diag(tt_v2)
Pc=tt.diag(tt_vc)
H1=kin+P2
H2=kin+P1
H11=tt.kron(tt_a11,H1)
H12=tt.diag(tt_vc)
H12=tt.kron(tt_a12,H12)
H21=tt.diag(tt_vc)
H21=tt.kron(tt_a21,H21)
H22=H2
H22=tt.kron(tt_a22,H22)
H=H11+H12+H21+H22
A=-EYE*H
A=A.round(1e-10)
#up and down vectors, used to extract state1 and state2 out of tt array 
su=tt.tensor(np.array([1,0]))
sd=tt.tensor(np.array([0,1]))
sm=tt.tensor(np.array([1,1]))
y0=tt.kron(su,tt_psi1)
yinit=y0
t= np.arange(0,tau*(nsc),tau)
#All-zero arrays for output 
p1 = np.zeros((nsc))
p2 = np.zeros((nsc))
pcisS0 = np.zeros((nsc))
pcisS1 = np.zeros((nsc))
ptransS0 = np.zeros((nsc))
ptransS1 = np.zeros((nsc))
population=np.zeros((nsc))
rt=np.zeros((nsc),dtype=complex)
r2=np.zeros((nsc))
sa=np.zeros((nsc))
r1=np.zeros((nsc))
pr = np.empty_like(t)

#tt-heaviside, used to extract population for GS/excited or cis/trans
tt_heav1=tt.kron(su,tt.ones(nx[0],dim))
tt_heav2=tt.kron(sd,tt.ones(nx[0],dim))
tt_trans=tt.multifuncrs2(tt_x, htrans,verb=0)
tt_trans=tt.kron(sm,tt_trans)
tt_trans=tt_trans.round(1e-14)
tt_cis=tt.multifuncrs2(tt_x,hcis,verb=0)
tt_cis=tt.kron(sm,tt_cis)
tt_cis=tt_cis.round(1e-14)

# Rank-adaptive algorithm
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
            yold=tt.ksl.ksl(A,yold,tau)
            ynew=tt.ksl.ksl(A,ynew,tau)
            overlap=np.abs(tt.dot(yold,ynew))*ddx
            if np.abs(overlap-1)<2e-6:
                y0=yold
            else:
                y0=ynew
            print(y0.r)
            print(k*tau*0.00002418884254)
        else:
            y0=tt.ksl.ksl(A,y0,tau)
            print(k*tau*0.00002418884254)
    pcisS1[k]=np.abs(tt.dot(y0*tt_heav1*tt_cis,y0*tt_heav1*tt_cis))*ddx
    ptransS1[k]=np.abs(tt.dot(y0*tt_heav1*tt_trans,y0*tt_heav1*tt_trans))*ddx
    pcisS0[k]=np.abs(tt.dot(y0*tt_heav2*tt_cis,y0*tt_heav2*tt_cis))*ddx
    ptransS0[k]=np.abs(tt.dot(y0*tt_heav2*tt_trans,y0*tt_heav2*tt_trans))*ddx
    population[k]=pcisS1[k]+ptransS1[k]+pcisS0[k]+ptransS0[k]
    r2[k]=y0.r[2]
    rt[k]=tt.dot(y0,yinit)*ddx


ptrans=ptransS0+ptransS1
pGS=ptransS0+pcisS0
t0=t*0.00002418884254

#population curves
plt.xlim(0.,1.)    
plt.ylim(0.,.8)            
plt.xlabel('time(ps)')
plt.ylabel('Populations')
plt.plot(t0,ptrans,'r',label='trans')
plt.plot(t0,pGS,'b',label='GS')                       
plt.legend()    


t_stop=process_time()
print(t_stop-t_start)
