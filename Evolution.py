import numpy as np
import scipy
import cmath 
import scipy.sparse as sp
import PauliStrings as pauli_strings

def QITE(n_qubits,H,H_trot,D,psi_0,N,dt,vervose=True,sparse=True):

    #checking whick method to obtain pauli strings is used
    if np.isreal(H.data).all() and np.isreal(psi_0.data).all():
        print("Using Real Pauli Strings") 
        num_paulis,PD,fail = pauli_strings.real(H_trot,D,n_qubits,sparse)
    else:
        print("Using General Pauli Strings")
        num_paulis,PD,fail = pauli_strings.general(H_trot,D,n_qubits,sparse)

    #chosing which routine of QITE will be used
    if fail:
        print("You need D>=T. Not running")
    else:
        if sparse:
            print("Sparse Routine")
            return QITE_sparse(n_qubits,H,H_trot,num_paulis,PD,psi_0,N,dt,vervose)
        else:
            print("Dense Routine, it will be slower and use more memory")
            return QITE_dense(n_qubits,H,H_trot,num_paulis,PD,psi_0,N,dt,vervose)

def QITE_sparse(n_qubits,H,H_trot,num_paulis,PD,psi_0,N,dt,vervose=True):
    psi_out=sp.lil_matrix((2**n_qubits,N+1),dtype=complex)
    psi_out[:,0]=psi_0.copy()
    psi_QITE=psi_0.copy()
    a=np.zeros((N,H_trot.Nk,num_paulis),dtype=complex)
    E_QITE=np.zeros(N+1)
    E_QITE[0]=np.real((psi_QITE.getH()@H@psi_QITE).trace())
    for i in range(0,N):
        #ara en aquest pas de temps anem a calcular els coeffs per a cada qubit
        for l in range(H_trot.Nk):
            #print('Time step',i+1,'/',N-1,'acting on qubit',l)
            #computing matrix S and coeffcient b (steps 1-3 in the algorithm)
            X=sp.lil_matrix((2**n_qubits,num_paulis),dtype=complex)
            b=np.zeros((num_paulis),dtype=complex)
            aux=np.real((psi_QITE.getH()@sp.linalg.expm(-2*H_trot.Hk[l]*dt)@psi_QITE).trace())
            c=cmath.sqrt(aux)
            expHdt=sp.linalg.expm(-H_trot.Hk[l]*dt)
            for j in range(num_paulis):
                b[j] = -1j*(psi_QITE.getH()@(expHdt@PD[l][j]-PD[l][j]@expHdt)@psi_QITE).trace()/c/dt
                X[:,j]=PD[l][j]@psi_QITE
            S=(X.getH()@X).todense()
           
            #obtencio coefficients a
            #invS_ex=np.linalg.pinv(S+S.T)
            #a[i,l]=np.real(invS_ex@b)

            #least square solution of equation (S+S^T)*a = b (step 4 in algorithm)
            a[i,l]=(scipy.linalg.lstsq(S+S.T,b,lapack_driver='gelsd'))[0]
            
            #construction of the evolution operator (steps 5 and 6 in the algorithm)
            operator=sp.csc_matrix((2**n_qubits,2**n_qubits),dtype=complex)
            for j in range(num_paulis):
                operator+=a[i,l,j]*PD[l][j]
            psi_QITE = sp.linalg.expm(-1j*operator*dt)@psi_QITE
            
        #Enegy of the state at time step i
        E_QITE[i+1] = np.real((psi_QITE.getH()@H@psi_QITE).trace())
        psi_out[:,i+1]=psi_QITE.copy()
        if vervose:
            print("Step",i+1,"/",N,"with energy",E_QITE[i+1])
    return E_QITE,psi_out,a

def QITE_dense(n_qubits,H,H_trot,num_paulis,PD,psi_0,N,dt,vervose=True):
    psi_out=np.zeros((2**n_qubits,N+1),dtype=complex)
    psi_out[:,0]=psi_0.copy()
    psi_QITE=psi_0
    a=np.zeros((N,n_qubits,num_paulis),dtype=complex)
    E_QITE=np.zeros(N+1)
    E_QITE[0]=np.real((psi_QITE.T.conj()@H@psi_QITE))

    for i in range(0,N):
        #ara en aquest pas de temps anem a calcular els coeffs per a cada qubit
        for l in range(H_trot.Nk):
            #obtencio matrius S
            P=np.zeros((num_paulis,2**n_qubits,2**n_qubits),dtype=complex)
            for j in range(num_paulis):
                P[j,:,:]=PD[l][j]
            X=np.matmul(P,psi_QITE)
            S=X@X.conj().T
            HB=H_trot.Hk[l]
            #obtencio coefficients b
            aux=np.real((psi_QITE.conj().T@scipy.linalg.expm(-2*HB*dt)@psi_QITE))
            c=cmath.sqrt(aux)
            expH=scipy.linalg.expm(-HB*dt)
            auxO=np.matmul(np.matmul(expH,P)-np.matmul(P,expH),psi_QITE)
            b = -1j*np.matmul(auxO,psi_QITE.conj())/c/dt
            #obtencio coefficients a
            invS_ex=np.linalg.pinv(S+S.transpose())
            a[i,l]=np.real(invS_ex@b).flatten()
            
            #ara evolucionem el qbit amb els nous coefficients
            operator=np.zeros((2**n_qubits,2**n_qubits),dtype=complex)
            for j in range(num_paulis):
                operator+=a[i,l,j]*P[j,:,:]
            psi_QITE = scipy.linalg.expm(-1j*operator*dt)@psi_QITE
            
        #valor esperat energia
        E_QITE[i+1] = np.real((psi_QITE.conj().T@H@psi_QITE))
        psi_out[:,i+1]=psi_QITE.copy()
        if vervose:
            print("Step",i+1,"/",N,"with energy",E_QITE[i+1])
    return E_QITE,psi_out,a


def ITE(H,psi_0,tmax,Nt):
    if sp.issparse(H) and sp.issparse(psi_0):
        return ITE_sparse(H,psi_0,tmax,Nt)
    else:
        return ITE_dense(H,psi_0,tmax,Nt)

def ITE_dense(H,psi_0,tmax,Nt):
    t_ITE=np.linspace(0,tmax,num=Nt)
    EITE=np.zeros(len(t_ITE))
    for i in range(len(t_ITE)):
        phi=scipy.linalg.expm(-H*t_ITE[i])@psi_0
        psiITE=phi/scipy.linalg.norm(phi)
        EITE[i]=np.real((psiITE.T.conj()@H@psiITE))
    return t_ITE,EITE

def ITE_sparse(H,psi_0,tmax,Nt):
    t_ITE=np.linspace(0,tmax,num=Nt)
    EITE=np.zeros(len(t_ITE))
    for i in range(len(t_ITE)):
        phi=sp.linalg.expm_multiply(-H*t_ITE[i],psi_0)
        psiITE=phi/sp.linalg.norm(phi)
        EITE[i]=np.real((psiITE.T.conj()@H@psiITE).trace())
    return t_ITE,EITE