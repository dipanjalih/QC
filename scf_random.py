import numpy as np
import pyscf
from pyscf import gto, scf
from scipy.linalg import fractional_matrix_power


#Step1:Specify the molecule.

dist=3.0
mol=gto.M(unit='Angstrom',atom='''Li 0 0 0; H 0 0 '''+str(dist),basis='sto-3g' )

'''
mol=pyscf.gto.M(
atom=
"""
O          .000000     .000000     .000000
H          .000000     .000000     .957200
H          .926627     .000000    -.239987
""",
basis = 'sto3g',)
'''
spatial_orbitals= mol.nao_nr();ne=mol.nelectron;occ=int(ne/2);virt=spatial_orbitals-occ
spin_orbitals=2*spatial_orbitals
enuc=mol.energy_nuc()


#########################################################################################


#Step2: Calculation of all the required molecular integrals.

#Overlap integrals
S=mol.intor("int1e_ovlp")
#Kinetic integrals
T=mol.intor("int1e_kin")
#e-N attraction integrals
V_nucl=mol.intor("int1e_nuc")
#H_core
H_core=T+V_nucl
#2e integrals
e2_int=mol.intor("int2e")


#########################################################################################


#Step3: Obtain a transformation matrix, X..


#Transformation matrix, X=S^(-0.5) (Symmetric orthogonalization)
X = fractional_matrix_power(S, -0.5)
#print('Transformation matrix, X:')
#print(X)


#########################################################################################


np.random.seed(0)
P_initial = np.random.randn(spatial_orbitals, spatial_orbitals)
#print('Initial guess for density matrix, P:')
print(P_initial)
maxiter=100
E_initial=0.0


#########################################################################################

for z in range (0,maxiter):

#Step5: Calculate G.

    G=np.einsum('kl,uvkl->uv',P_initial,e2_int)-0.5*np.einsum('kl,ulkv->uv',P_initial,e2_int)
    #print(G)


#########################################################################################


#Step6: Add G to H_core to obtain F.

    F=H_core+G
    #print(F)

    E_new=0.5*np.sum(P_initial*(H_core+F))
    #print(E_new)

#########################################################################################


#Step7: Calculate transformed Fock matrix, F'.

    F_prime=np.matmul(X.T,(np.matmul(F,X)))
    #print('Transformed Fock matrix:')
    #print(F_prime)


#########################################################################################


#Step8: Diagonalization of F' to obtain C'.

    F_prime_eval, F_prime_evec = np.linalg.eigh(F_prime)
    C_prime=F_prime_evec
    #E=F_prime_eval
    #print(C_prime)


#########################################################################################


#Step9: Calculate C=XC'.

    C=np.matmul(X,C_prime)
    #print(C)


#########################################################################################


#Step10: Form a new density matrix, P_new from C.

    n=int(ne/2)
    P_new = 2.0*np.matmul(C[:,:n],np.transpose(C[:,:n]))


#########################################################################################


#Step11: Check for convergence.
    delta=0.0
    i,j = P_initial.shape
    for a in range(0,i):
        for b in range(0,j):
            delta = delta + (P_new[a][b]- P_initial[a][b])**2.0

    #delta=np.sqrt(np.sum(delta**2)/4.0)
    rms_delta=np.sqrt(delta)
    diff_energy = E_new - E_initial 
    print('Delta:',rms_delta)

    if rms_delta <= 1e-6 and diff_energy<=1e-6:
        total_energy = E_new + enuc
        print('Convergence reached, Energy value:',total_energy)
        break
    else:
        P_initial=P_new
        print('Iteration:',z+1)
        #E_initial = E_new
#print(P_initial)



