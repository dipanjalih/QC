# Code for scattering operators (generalized rank-two operators with a real excitation on one vertex and a hole-hole 
# or particle-particle scattering in the other vertex) and identifying the 'dominant' ones using lowest order perturbative estimates.


import numpy as np
from qiskit import Aer
from qiskit.aqua import QuantumInstance
from qiskit.aqua.algorithms import VQE, ExactEigensolver, NumPyMinimumEigensolver
from qiskit.aqua.operators import Z2Symmetries
from qiskit.aqua.components.optimizers import COBYLA,SLSQP,L_BFGS_B
from qiskit.chemistry import FermionicOperator
from qiskit.chemistry.drivers import PySCFDriver, UnitsType, HFMethodType, Molecule
from qiskit.chemistry.components.variational_forms import UCCSD,UCCSDS
from qiskit.chemistry.components.initial_states import HartreeFock
from qiskit.chemistry.core import hamiltonian
from qiskit.circuit import QuantumCircuit, ParameterVector
from pyscf import gto, scf, ao2mo, fci, cc
from qiskit.chemistry import MP2Info
import collections
from collections import Counter
import warnings
warnings.filterwarnings("ignore")



# The geometry of the molecule is defined.
molecule=Molecule(geometry=[['B',[0.00000, 0.00000 , 0.00000]],['H',[0.00000, 0.00000, 3.0000]]],
        charge=0,multiplicity=1)
driver=PySCFDriver(molecule=molecule,unit=UnitsType.ANGSTROM, basis='sto-3g')
qmolecule=driver.run()


# Nuclear repulsion energy
nuclear_repulsion_energy = qmolecule.nuclear_repulsion_energy


map_type = 'jordan_wigner'
num_alpha = qmolecule.num_alpha
num_beta = qmolecule.num_beta
no_electrons = qmolecule.num_alpha + qmolecule.num_beta
num_spin_orbitals = qmolecule.num_orbitals * 2
ferOp = FermionicOperator(h1=qmolecule.one_body_integrals, h2=qmolecule.two_body_integrals)
qubitOp = ferOp.mapping(map_type)
shift = nuclear_repulsion_energy
print('Number of spin orbitals',num_spin_orbitals)
num_qubits = qubitOp.num_qubits
print('Number of qubits:', num_qubits)
print("Shift=",shift)
print("Number of electrons=",no_electrons)


ints = qmolecule.mo_eri_ints
o_e = qmolecule.orbital_energies
print("Orbital energies:", o_e)

num_orbitals=int(num_spin_orbitals)


active_occ_list_alpha = []
active_occ_list_beta = []
active_unocc_list_alpha = []
active_unocc_list_beta = []

beta_idx = num_orbitals //2
#print(beta_idx)


active_occ_list_alpha = list(range(0, num_alpha))
active_occ_list_beta = [i + beta_idx for i in range(0, num_beta)]
active_unocc_list_alpha = list(range(num_alpha, num_orbitals // 2))
active_unocc_list_beta = [i + beta_idx for i in range(num_beta, num_orbitals // 2)]
print('Occupied spin orbitals:')
print(active_occ_list_alpha)
print(active_occ_list_beta)
print('Unoccupied spin orbitals:')
print(active_unocc_list_alpha)
print(active_unocc_list_beta)



double_excitations=[]



#Opposite spin scatterers

#hphh

#excitation from alpha and hole-hole scattering in beta            
for i_alpha in active_occ_list_alpha:
    for a_alpha in active_unocc_list_alpha:
        for i_beta in active_occ_list_beta:
            for i_beta1 in active_occ_list_beta:
                if (i_beta != i_beta1):             #To exclude diagonal terms
                    double_excitations.append([i_alpha, a_alpha, i_beta, i_beta1])


#hphh

#excitation from beta and hole-hole scattering in alpha            
for i_beta in active_occ_list_beta:
    for a_beta in active_unocc_list_beta:
        for i_alpha in active_occ_list_alpha:
            for i_alpha1 in active_occ_list_alpha:
                if (i_alpha != i_alpha1):       #To exclude diagonal terms
                    double_excitations.append([i_beta, a_beta, i_alpha, i_alpha1])


#hppp

#excitation from alpha and particle-particle scattering in beta            
for i_alpha in active_occ_list_alpha:
    for a_alpha in active_unocc_list_alpha:
        for a_beta in active_unocc_list_beta:
            for a_beta1 in active_unocc_list_beta:
                if (a_beta != a_beta1):     #To exclude diagonal terms
                    double_excitations.append([i_alpha, a_alpha, a_beta, a_beta1])



#hppp

#excitation from beta and particle-particle scattering in alpha            
for i_beta in active_occ_list_beta:
    for a_beta in active_unocc_list_beta:
        for a_alpha in active_unocc_list_alpha:
            for a_alpha1 in active_unocc_list_alpha:
                if (a_alpha != a_alpha1):   #To exclude diagonal terms
                    double_excitations.append([i_beta, a_beta, a_alpha, a_alpha1])




# Same spin scatterers

#hphh

for i_alpha in active_occ_list_alpha:
    for a_alpha in active_unocc_list_alpha:
        for i_alpha1 in active_occ_list_alpha:
            for i_alpha2 in active_occ_list_alpha:
                if (i_alpha!=i_alpha1):
                    #To exclude diagonal terms and same spin excitations with duplicated indices respectively
                    if (i_alpha1 != i_alpha2 and i_alpha != i_alpha2 ):
                        double_excitations.append([i_alpha, a_alpha, i_alpha1, i_alpha2])


#hphh

for i_beta in active_occ_list_beta:
    for a_beta in active_unocc_list_beta:
        for i_beta1 in active_occ_list_beta:
            for i_beta2 in active_occ_list_beta:
                if (i_beta!=i_beta1):
                    #To exclude diagonal terms and same spin excitations with duplicated indices respectively
                    if (i_beta1 != i_beta2 and i_beta != i_beta2):
                            double_excitations.append([i_beta, a_beta, i_beta1, i_beta2])

#hppp

for i_alpha in active_occ_list_alpha:
    for a_alpha in active_unocc_list_alpha:
        for a_alpha1 in active_unocc_list_alpha:
            for a_alpha2 in active_unocc_list_alpha:
                if (a_alpha!=a_alpha2):
                    #To exclude diagonal terms and same spin excitations with duplicated indices respectively
                    if (a_alpha1 != a_alpha2 and a_alpha != a_alpha1):
                        double_excitations.append([i_alpha, a_alpha, a_alpha1, a_alpha2])


#hppp

for i_beta in active_occ_list_beta:
   for a_beta in active_unocc_list_beta:
       for a_beta1 in active_unocc_list_beta:
           for a_beta2 in active_unocc_list_beta:
               if (a_beta!= a_beta2):
                   #To exclude diagonal terms and same spin excitations with duplicated indices respectively
                   if (a_beta1 != a_beta2 and a_beta != a_beta1):
                       double_excitations.append([i_beta, a_beta, a_beta1, a_beta2])



    
#print("List of scatterers:", double_excitations)
print("Total number of S operators:", len(double_excitations))



# Pruning the scatterers based on perturbative estimates

pruned_s=list()
amplitudes_s=list()
integrals_s=list()


for n, _ in enumerate(double_excitations):
    idxs = double_excitations[n]
    i = idxs[0] % beta_idx  # MO indexing
    j = idxs[2] % beta_idx
    a_i = idxs[1] % beta_idx
    b = idxs[3] % beta_idx

    tiajb = ints[i, a_i, j, b]
    tibja = ints[i, b, j, a_i]
    num = (2*tiajb-tibja)

    # MP2 like denominator
    denom = o_e[b] + o_e[a_i] - o_e[i] - o_e[j]
    
    # Perturbative estimates
    s_mp2 = -num/denom

    if (abs(s_mp2) > 1e-5):   # Tune the thresholds for S as per requirement
        pruned_s.append(double_excitations[n])
        amplitudes_s.append(s_mp2)
        integrals_s.append(num)



# Change the output of double_excitations from [from, to, from, to] -> ((from, from), (to, to))

pruned_s_nature=[]
for i in range (len(pruned_s)):
    pruned_s_nature.append(((pruned_s[i][0],pruned_s[i][2]),(pruned_s[i][1],pruned_s[i][3])))

print("Number of S operators after pruning:",len(pruned_s_nature))
print("PRUNED S OPERATORS:", pruned_s_nature)
print("Corresponding S amplitudes:", amplitudes_s)

