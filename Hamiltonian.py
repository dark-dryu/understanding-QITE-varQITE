# %%
from fractions import Fraction
import numpy as np
import scipy.sparse as sp
from qiskit.quantum_info import Pauli
from qiskit.quantum_info import SparsePauliOp,Pauli


class TrotterHamiltonian:
    def __init__(self, T: int, Nk: int, Rq: int, Hk: list, indk: np.ndarray):
        self.T = T          #Trotter size
        self.Nk = Nk        #number of pieces
        self.Rq = Rq        #number of times each single qubit operator acts
        self.Hk = Hk        #list of all the Hammiltonian terms
        self.indk = indk    #first qbit on which each piece acts on

def TFIM(J,h,n_qubits,T=2,sparse=True):
    """
    This function ...
    """
    X=[]
    ZZ=[]
    M=[]
    for i in range(n_qubits):
        # Hamiltonian
        sx=["I"]*n_qubits
        szz=["I"]*n_qubits
        i1= (i+1)%n_qubits #periodic index
        sx[i]="X"
        szz[i]="Z"
        szz[i1]="Z"
        X.append(Pauli(''.join(sx)).to_matrix(sparse=sparse))
        ZZ.append(Pauli(''.join(szz)).to_matrix(sparse=sparse))
        #Magnetization
        mag=["I"]*n_qubits
        mag[i]="Z"
        M.append(Pauli(''.join(mag)).to_matrix(sparse=sparse))
        
    H = J*np.sum(ZZ,axis=0) + h*np.sum(X,axis=0)
    if sparse:
        H = sp.csc_matrix(H)
    #############################################################
    H_T=[]
    if T>n_qubits:
        print("Your Trotterization size is bigger than you system")
    if T<2:
        print("Your Trotterization size is too small, min value is T=2")
    N_k=Fraction(n_qubits,T).numerator
    R=Fraction(n_qubits,T).denominator
    if R==1:
        N_k=N_k*2
        R=R*2
    index=np.floor(n_qubits/N_k*np.arange(N_k)).astype(int)
    for i in range(N_k):
        ind=index[i]
        h_k=np.zeros((2**n_qubits,2**n_qubits),dtype=complex)
        for j in range(T):
            indT=(ind+j)%n_qubits
            if indT==(index[(i+1)%N_k]-1)%n_qubits:
                h_k=h_k+J*(ZZ[indT])/(R-1)+h*X[indT]/R
            elif j == T-1:
                h_k=h_k+h*X[indT]/R
            else:
                h_k=h_k+J*(ZZ[indT])/R+h*X[indT]/R
        if sparse:
            H_T.append(sp.csc_matrix(h_k))
        else:
            H_T.append(h_k)
    
    ###################################################
    #This is a check to see if the trotterization is the same as the original Hamiltonian
    if sparse:
        difH=sp.linalg.norm((H-sum(H_T)),ord=2)
    else:
        difH=np.linalg.norm((H-sum(H_T)),ord=2)
    if difH<1e-14:
        print('Succesfull Troterization')
        print("The Trotterization consists of",N_k,"terms with the starting qubit of each piece at",index)
        print("Each single qubit term appears",R,"times")
    else:
        print('Failed Trotterization, you can still use the generated full Hamiltonian')

    H_trot=TrotterHamiltonian(T,N_k,R,H_T,index)

    return H,H_trot

def TFIM_Pauli(J,h,n_qubits):
    zz_terms = []
    x_terms = []
    for i in range(n_qubits):
        zz_term = ['I']*n_qubits
        zz_term[i] = 'Z'
        zz_term[(i+1)%n_qubits] = 'Z'
        zz_terms.append(''.join(zz_term))

        x_term = ['I']*n_qubits
        x_term[i] = 'X'
        x_terms.append(''.join(x_term))

    Ham = SparsePauliOp(zz_terms, coeffs=[J]*n_qubits) + SparsePauliOp(x_terms, coeffs=[h]*n_qubits)
    return Ham