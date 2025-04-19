# Code to determine the 'important' singles using second order many-body perturbative correction to wavefunction.

import numpy as np
from qiskit import Aer
from qiskit.aqua import QuantumInstance
from qiskit.aqua.algorithms import VQE, ExactEigensolver, NumPyMinimumEigensolver
from qiskit.aqua.operators import Z2Symmetries
from qiskit.aqua.components.optimizers import COBYLA,SLSQP,L_BFGS_B
from qiskit.chemistry import FermionicOperator
from qiskit.chemistry.drivers import PySCFDriver, UnitsType, HFMethodType, Molecule
from qiskit.chemistry.components.variational_forms import UCCSD,UCCSDS
from pyscf import gto, symm, scf
from qiskit.chemistry.components.initial_states import HartreeFock
from qiskit.chemistry.core import hamiltonian
from qiskit.circuit import QuantumCircuit, ParameterVector
from pyscf import gto, scf, ao2mo, fci, cc
from qiskit.chemistry import MP2Info
import collections
import warnings
warnings.filterwarnings("ignore")


# The geometry of the molecule is defined.
d=3.0
molecule=Molecule(geometry=[['Li',[0.00000, 0.00000 , 0.00000]],['H',[0.00000, 0.00000, d]]],
       charge=0,multiplicity=1)

driver=PySCFDriver(molecule=molecule,unit=UnitsType.ANGSTROM, basis='sto-3g')
qmolecule=driver.run()

# Nuclear repulsion energy
nuclear_repulsion_energy = qmolecule.nuclear_repulsion_energy

map_type = 'jordan_wigner'
num_particles = qmolecule.num_alpha + qmolecule.num_beta
num_spin_orbitals = qmolecule.num_orbitals * 2
ferOp = FermionicOperator(h1=qmolecule.one_body_integrals, h2=qmolecule.two_body_integrals)
qubitOp = ferOp.mapping(map_type)
shift = nuclear_repulsion_energy 
print('Number of spin orbitals',num_spin_orbitals)
num_qubits = qubitOp.num_qubits
print('Number of qubits:', num_qubits)
print('Number of electrons:', num_particles)
print("Shift=",shift)


no_electrons = num_particles
num_orbitals = int(num_spin_orbitals/2)
print("Number of spatial orbitals:", num_orbitals)
ints = qmolecule.mo_eri_ints
o_e = qmolecule.orbital_energies
print("Orbital energies:",o_e)
num_alpha=qmolecule.num_alpha
num_beta=qmolecule.num_beta
print("Number of alpha electrons:", num_alpha)
print("Number of beta electrons:", num_beta)

beta_idx = num_spin_orbitals //2
print(beta_idx)


active_occ_list_alpha = list(range(0, num_alpha))
active_occ_list_beta = [i + beta_idx for i in range(0, num_beta)]
active_unocc_list_alpha = list(range(num_alpha, num_spin_orbitals // 2))
active_unocc_list_beta = [i + beta_idx for i in range(num_beta, num_spin_orbitals // 2)]
print('Occupied spin orbitals:')
print(active_occ_list_alpha)
print(active_occ_list_beta)
print('Unoccupied spin orbitals:')
print(active_unocc_list_alpha)
print(active_unocc_list_beta)




###########################################################
###########################UCCSD###########################
###########################################################


#STEP 3: UCCSD

HF_state = HartreeFock(num_spin_orbitals,num_particles,qubit_mapping=map_type,
                       two_qubit_reduction=False)
var_form = UCCSD(num_orbitals=num_spin_orbitals, num_particles=num_particles,
                 reps=1,qubit_mapping=map_type, initial_state=HF_state,
                 two_qubit_reduction=False,num_time_slices=1,method_doubles='ucc',excitation_type='sd')

n1 = var_form.num_parameters
c = [0]*n1
optimizer=L_BFGS_B(maxfun=50000, maxiter=100000,epsilon=1e-10)
vqe = VQE(qubitOp, var_form, optimizer, initial_point= c)
par = vqe._var_form_params
t_circuit = var_form.construct_circuit(par)
singles=var_form.single_excitations
doubles=var_form.double_excitations
#print(singles)
#print(doubles)


# Prune the T2 operators at MP2 level
pruned_t=[]
integrals_t2=[]
amplitudes_t2=[]
for n, _ in enumerate(doubles):
    idxs = doubles[n]
    i = idxs[0] % num_orbitals  # MO indexing
    j = idxs[2] % num_orbitals
    a_i = idxs[1] % num_orbitals
    b = idxs[3] % num_orbitals

    tiajb = ints[i, a_i, j, b]
    tibja = ints[i, b, j, a_i]
    
    
    num = (2 * tiajb - tibja)
    denom = o_e[b] + o_e[a_i] - o_e[i] - o_e[j]
    t_mp2 = -num/denom          # MP2 values for T2 operators
    if (abs(t_mp2) > 1e-5):     # Set a user defined threshold
        #print(doubles[n],num,t_mp2)
        pruned_t.append(doubles[n])
        amplitudes_t2.append(t_mp2)
        integrals_t2.append(num)
#print(pruned_t)


t2_index_1=[]
t2_index_2=[]
t2_index_3=[]
t2_index_4=[]
for i in range (len(pruned_t)):
    t2_index_1.append(pruned_t[i][0])
    t2_index_2.append(pruned_t[i][1])
    t2_index_3.append(pruned_t[i][2])
    t2_index_4.append(pruned_t[i][3])


# Define the operators for checking the contractions with T2 at the second order wf correction
V_excitations=[]

# Opposite spin sector
for a_beta in active_unocc_list_beta:
    for k_beta in active_occ_list_beta:
        for i_alpha in active_occ_list_alpha:
            for j_alpha in active_occ_list_alpha:
                V_excitations.append([a_beta, k_beta,i_alpha, j_alpha])



for a_alpha in active_unocc_list_alpha:
    for k_alpha in active_occ_list_alpha:
        for i_beta in active_occ_list_beta:
            for j_beta in active_occ_list_beta:
                V_excitations.append([a_alpha, k_alpha,i_beta, j_beta])


for c_beta in active_unocc_list_beta:
    for k_beta in active_occ_list_beta:
        for a_alpha in active_unocc_list_alpha:
            for b_alpha in active_unocc_list_alpha:
                V_excitations.append([c_beta, k_beta,a_alpha, b_alpha])


for c_alpha in active_unocc_list_alpha:
    for k_alpha in active_occ_list_alpha:
        for a_beta in active_unocc_list_beta:
            for b_beta in active_unocc_list_beta:
                V_excitations.append([c_alpha, k_alpha,a_beta, b_beta])



#Same spin sector
for a_alpha in active_unocc_list_alpha:
    for k_alpha in active_occ_list_alpha:
        for i_alpha in active_occ_list_alpha:
            for j_alpha in active_occ_list_alpha:
                V_excitations.append([a_alpha, k_alpha,i_alpha, j_alpha])


for a_beta in active_unocc_list_beta:
    for k_beta in active_occ_list_beta:
        for i_beta in active_occ_list_beta:
            for j_beta in active_occ_list_beta:
                V_excitations.append([a_beta, k_beta,i_beta, j_beta])


for c_alpha in active_unocc_list_alpha:
    for k_alpha in active_occ_list_alpha:
        for a_alpha in active_unocc_list_alpha:
            for b_alpha in active_unocc_list_alpha:
                V_excitations.append([c_alpha, k_alpha,a_alpha, b_alpha])


for c_beta in active_unocc_list_beta:
    for k_beta in active_occ_list_beta:
        for a_beta in active_unocc_list_beta:
            for b_beta in active_unocc_list_beta:
                V_excitations.append([c_beta, k_beta,a_beta, b_beta])

#print(V_excitations)


integrals_V=[]
amplitudes_V=[]
pruned_V=[]
for n, _ in enumerate(V_excitations):
    idxs = V_excitations[n]
    i = idxs[0] % num_orbitals  # Since spins are same drop to MO indexing
    j = idxs[2] % num_orbitals
    a_i = idxs[1] % num_orbitals
    b = idxs[3] % num_orbitals

    tiajb = ints[i, a_i, j, b]
    tibja = ints[i, b, j, a_i]

    #num = tiajb
    num=(2 * tiajb - tibja)
    denom = o_e[b] + o_e[a_i] - o_e[i] - o_e[j]     #MP2 like denominators for V
    V_mp2 = -num/denom
    if (num > 1e-15):
        #print(V_excitations[n],num,V_mp2)
        pruned_V.append(V_excitations[n])
        amplitudes_V.append(t_mp2)
        integrals_V.append(num)

#print(pruned_V)
#print(len(pruned_V))
V_index_1=[]
V_index_2=[]
V_index_3=[]
V_index_4=[]
for i in range (len(pruned_V)):
    V_index_1.append(pruned_V[i][0])
    V_index_2.append(pruned_V[i][1])
    V_index_3.append(pruned_V[i][2])
    V_index_4.append(pruned_V[i][3])

#print(V_index_1)


print("###########################################################")
print("############################T1#############################")
print("###########################################################")


# Generation of T1. 8 different cases
T_cont=[]
V_cont=[]
T1=[]
amp=[]

for i in range (len(pruned_V)):
    #print("V:",pruned_V[i])
    for j in range (len(pruned_t)):
        if ((t2_index_3[j] == V_index_4[i] and t2_index_2[j] == V_index_1[i] and t2_index_1[j] == V_index_2[i])):
                #print(pruned_t[j])
                T_cont.append(pruned_t[j])
                V_cont.append(pruned_V[i])
                T1.append(np.array([V_index_3[i],t2_index_4[j]]))
                i_t3=V_index_3[i] % num_orbitals
                a_t3=t2_index_4[j] % num_orbitals
                
                amp.append(amplitudes_t2[j]*(-1)*(-1)*integrals_V[i]/(o_e[a_t3] - o_e[i_t3]))


        elif ((t2_index_1[j] == V_index_4[i] and t2_index_4[j] == V_index_1[i] and t2_index_3[j] == V_index_2[i])):
                #print(pruned_t[j])
                T_cont.append(pruned_t[j])
                V_cont.append(pruned_V[i])
                T1.append(np.array([V_index_3[i],t2_index_2[j]]))
                i_t3=V_index_3[i] % num_orbitals
                a_t3=t2_index_2[j] % num_orbitals

                amp.append(amplitudes_t2[j]*(-1)*(-1)*integrals_V[i]/(o_e[a_t3] - o_e[i_t3]))


        elif (t2_index_1[j] == V_index_4[i] and t2_index_2[j] == V_index_1[i] and t2_index_3[j] == V_index_2[i]):
                #print(pruned_t[j])
                T_cont.append(pruned_t[j])
                V_cont.append(pruned_V[i])
                T1.append(np.array([V_index_3[i],t2_index_4[j]]))
                i_t3=V_index_3[i] % num_orbitals
                a_t3=t2_index_4[j] % num_orbitals
                
                amp.append(amplitudes_t2[j]*(-1)*integrals_V[i]/(o_e[a_t3] - o_e[i_t3] ))


        elif (t2_index_3[j] == V_index_4[i] and t2_index_1[j] == V_index_2[i] and t2_index_4[j] == V_index_1[i]):
                #print(pruned_t[j])
                T_cont.append(pruned_t[j])
                V_cont.append(pruned_V[i])
                T1.append(np.array([V_index_3[i],t2_index_2[j]]))
                i_t3=V_index_3[i] % num_orbitals
                a_t3=t2_index_2[j] % num_orbitals

                amp.append(amplitudes_t2[j]*(-1)*integrals_V[i]/(o_e[a_t3] - o_e[i_t3] ))


        elif (t2_index_4[j] == V_index_3[i] and t2_index_2[j] == V_index_1[i] and t2_index_1[j] == V_index_2[i]):
                #print(pruned_t[j])
                T_cont.append(pruned_t[j])
                V_cont.append(pruned_V[i])
                T1.append(np.array([t2_index_3[j],V_index_4[i]]))
                i_t3=t2_index_3[j] % num_orbitals
                a_t3=V_index_4[i] % num_orbitals
                
                amp.append(amplitudes_t2[j]*(-1)*integrals_V[i]/(o_e[a_t3] - o_e[i_t3]))

        
        elif (t2_index_2[j] == V_index_3[i] and t2_index_4[j] == V_index_1[i] and t2_index_3[j] == V_index_2[i]):
                #print(pruned_t[j])
                T_cont.append(pruned_t[j])
                V_cont.append(pruned_V[i])
                T1.append(np.array([t2_index_1[j],V_index_4[i]]))
                i_t3=t2_index_1[j] % num_orbitals
                a_t3=V_index_4[i] % num_orbitals

                amp.append(amplitudes_t2[j]*(-1)*integrals_V[i]/(o_e[a_t3] - o_e[i_t3]))


        elif (t2_index_2[j] == V_index_3[i] and t2_index_1[j] == V_index_2[i] and t2_index_4[j] == V_index_1[i]):
                #print(pruned_t[j])
                T_cont.append(pruned_t[j])
                V_cont.append(pruned_V[i])
                T1.append(np.array([t2_index_3[j],V_index_4[i]]))
                i_t3=t2_index_3[j] % num_orbitals
                a_t3=V_index_4[i] % num_orbitals
                
                amp.append(amplitudes_t2[j]*(-1)*(-1)*integrals_V[i]/(o_e[a_t3] - o_e[i_t3]))


        elif (t2_index_4[j] == V_index_3[i] and t2_index_3[j] == V_index_2[i] and t2_index_2[j] == V_index_1[i]):
                #print(pruned_t[j])
                T_cont.append(pruned_t[j])
                V_cont.append(pruned_V[i])
                T1.append(np.array([t2_index_1[j],V_index_4[i]]))
                i_t3=t2_index_1[j] % num_orbitals
                a_t3=V_index_4[i] % num_orbitals

                amp.append(amplitudes_t2[j]*(-1)*(-1)*integrals_V[i]/(o_e[a_t3] - o_e[i_t3]))

#print(T1)

T1_nature=[]
for i in range (len(T1)):
    T1_nature.append(((T1[i][0],),(T1[i][1],)))
#print(T1_nature)    

singles_dict=collections.Counter(T1_nature)
#print(singles_dict)
unique_singles=list(singles_dict.keys())
print("Pruned T1 list:", unique_singles)





