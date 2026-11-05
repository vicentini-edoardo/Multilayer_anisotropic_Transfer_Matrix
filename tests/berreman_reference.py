"""Independent Berreman/Yeh 4x4 reference (scipy expm); shares no code with solver_fast.

Returns the Ex-amplitude ratio, which is -rpp of the solver's (Passler) convention.
Valid for an isotropic superstrate and moderate layer growth (k0*d*zeta < ~15).
"""
import numpy as np
from scipy.linalg import expm
C=299792458.0
def delta(eps, xi):
    e=eps; e33=e[2,2]
    D=np.zeros((4,4),complex)
    D[0,0]=-xi*e[2,0]/e33; D[0,1]=1-xi**2/e33; D[0,2]=-xi*e[2,1]/e33
    D[1,0]=e[0,0]-e[0,2]*e[2,0]/e33; D[1,1]=-xi*e[0,2]/e33; D[1,2]=e[0,1]-e[0,2]*e[2,1]/e33
    D[2,3]=1
    D[3,0]=e[1,0]-e[1,2]*e[2,0]/e33; D[3,1]=-xi*e[1,2]/e33; D[3,2]=e[1,1]-xi**2-e[1,2]*e[2,1]/e33
    return D
def modes(D, forward):
    q,V=np.linalg.eig(D)
    fw=(q.imag>1e-12*np.maximum(1,abs(q)))|((abs(q.imag)<=1e-12*np.maximum(1,abs(q)))&(q.real>0))
    idx=np.where(fw==forward)[0]
    if len(idx)!=2: raise ValueError("mode split %d"%len(idx))
    V=V[:,idx]
    # p first: larger |Ex| (row 0) relative to |Ey| (row 2)
    pw=abs(V[0])/(abs(V[0])+abs(V[2])+1e-300)
    order=np.argsort(-pw)
    return V[:,order]
def rpp_ref(system, f_hz, xi):
    """xi: complex kx/k0. Layers must have initialize_sys(f_hz) done. Isotropic superstrate required."""
    k0=2*np.pi*f_hz/C
    P=np.eye(4,dtype=complex)
    for L in system.layers:
        P=P@expm(-1j*k0*L.thick*delta(L.epsilon,xi))
    Ai=modes(delta(system.superstrate.epsilon,xi),True)
    Ar=modes(delta(system.superstrate.epsilon,xi),False)
    At=modes(delta(system.substrate.epsilon,xi),True)
    # Ai[:,0]*1 + Ar@[rp,rs] = P@At@[tp,ts]
    M=np.column_stack([Ar[:,0],Ar[:,1],-(P@At)[:,0],-(P@At)[:,1]])
    sol=np.linalg.solve(M,-Ai[:,0])
    return sol[0]*Ar[0,0]/Ai[0,0]  # ratio of Ex amplitudes
