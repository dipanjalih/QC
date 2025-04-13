# imports
import qiskit as qt
import math
from qiskit_nature.drivers.second_quantization import PySCFDriver
from qiskit_nature.problems.second_quantization.electronic import ElectronicStructureProblem
from qiskit_nature.converters.second_quantization import QubitConverter
from qiskit_nature.transformers.second_quantization.electronic import FreezeCoreTransformer
from qiskit_nature.mappers.second_quantization import JordanWignerMapper
from qiskit_nature.circuit.library import HartreeFock, UCC
from qiskit.algorithms.optimizers import COBYLA, CG, SLSQP, L_BFGS_B
from qiskit.algorithms import VQE
from qiskit_nature.algorithms.ground_state_solvers.minimum_eigensolver_factories import NumPyMinimumEigensolverFactory
from qiskit_nature.algorithms.ground_state_solvers import GroundStateEigensolver
from qiskit.circuit.library import EvolvedOperatorAnsatz
from qiskit.circuit import Parameter, QuantumCircuit, QuantumRegister
from qiskit import Aer
import numpy as np
import scipy.optimize as spo
from qiskit.circuit import Parameter, QuantumCircuit, QuantumRegister
from qiskit.opflow import StateFn
import warnings
warnings.filterwarnings("ignore")


import logging
logging.basicConfig(filename="cnot_eff.log", level=logging.INFO)
logger=logging.getLogger()
logger.setLevel(logging.DEBUG)



# Molecule geometry
dist=3.0
driver = PySCFDriver(atom='Li 0.0 0.0 0.0; H 0.0 0.0 '+str(dist), charge=0, spin=0, basis='sto3g')
es_problem = ElectronicStructureProblem(driver)



# Obtaining qubit Hamiltonian
mapper = JordanWignerMapper()
converter = QubitConverter(mapper=mapper, two_qubit_reduction=False)
second_q_op = es_problem.second_q_ops()
qubit_op = converter.convert(second_q_op['ElectronicEnergy'])
#print(qubit_op)

# Obtaining some molecular  properties
es_particle_number = es_problem.grouped_property_transformed.get_property('ParticleNumber')
num_particles = es_particle_number.num_particles
num_spin_orbitals = es_particle_number.num_spin_orbitals
es_energy = es_problem.grouped_property_transformed.get_property('ElectronicEnergy')
Enuc = es_energy.nuclear_repulsion_energy
print("Nuclear repulsion energy:", Enuc)
print("Number of spin orbitals:", num_spin_orbitals)
print("Number of electrons:", num_particles)
num_spatial_orbitals=int(num_spin_orbitals/2)
print("Number of spatial orbitals:", num_spatial_orbitals)
num_qubits = num_spin_orbitals
print("Number of qubits:", num_qubits)


# Initial_state
init_state = HartreeFock(num_spin_orbitals, num_particles, converter)

# Optimizer and backend information
optimizer = L_BFGS_B(maxfun=100000)
backend = Aer.get_backend('statevector_simulator')



# Evaluating Hartree-Fock Energy
# Create dummy parametrized circuit
theta = Parameter('a')
n = qubit_op.num_qubits
qc = QuantumCircuit(qubit_op.num_qubits)
qc.rz(theta*0,0)
ansatz_dummy = qc
ansatz_dummy.compose(init_state, front=True, inplace=True)
#Pass it through VQE
algorithm = VQE(ansatz_dummy,quantum_instance=backend)
res = algorithm.compute_minimum_eigenvalue(qubit_op).eigenvalue.real
hf_energy = res + Enuc
print("Hartree-Fock Energy:", hf_energy)


def Hartree_Fock(qc):
    for i in range(num_particles[0]):
        qc.x(i)
        qc.x(int(num_spatial_orbitals) + i)
    return qc


# CNOT Efficient circuit implementation for single excitations
def single_excitation_circ(qc,i,k,theta):
    qc.rz(math.pi/2, k)
    qc.rx(math.pi/2, k)
    qc.rx(math.pi/2, i)
    qc.cx(k,i)
    qc.rx(theta,k)
    qc.rz(theta,i)
    qc.cx(k,i)
    qc.rx(-math.pi/2, k)
    qc.rx(-math.pi/2, i)
    qc.rz(-math.pi/2, k)

    return qc



# CNOT Efficient circuit implementation for double excitations
def double_excitation_circ(qc,i,j,k,l,theta):    
    qc.cx(l,k)
    qc.cx(j,i)
    qc.cx(l,j)
    qc.x(k)
    qc.x(i) 
    qc.ry(theta/8, l)
    qc.h(k)
    qc.cx(l,k)
    qc.ry(-theta/8, l)
    qc.h(i)
    qc.cx(l,i)
    qc.ry(theta/8, l)
    qc.cx(l,k)
    qc.ry(-theta/8, l)
    qc.h(j)
    qc.cx(l,j)
    qc.ry(theta/8, l)
    qc.h(j)
    qc.cx(l,k)
    qc.ry(-theta/8, l)
    qc.cx(l,i)    
    qc.ry(theta/8, l)
    qc.h(i)
    qc.cx(l,k)
    qc.ry(math.pi/2, j)
    qc.ry(-theta/8, l)
    qc.h(k)
    qc.p(math.pi/2, j)
    qc.cx(l,j)
    qc.p(math.pi/2, l)
    qc.x(k)
    qc.p(-math.pi/2, j)
    qc.ry(-math.pi/2, j)
    qc.x(i)
    qc.cx(l,k)
    qc.cx(j,i)
    return qc


# Define the chemically motivated ansatz
# kUpCCGSD excitations list
gen_paired_list=[]
for i in range(num_spatial_orbitals):
    for j in range(num_spatial_orbitals):
        if (i != j and j > i):
            gen_paired_list.append(((i, i+num_spatial_orbitals), (j, j+num_spatial_orbitals)))

gen_singles_list=[]
for i in range(num_spatial_orbitals):
    for j in range(num_spatial_orbitals):
        if (i != j and j > i):
            gen_singles_list.append(((i, ), (j, )))
            gen_singles_list.append(((i+num_spatial_orbitals, ), (j+num_spatial_orbitals, )))

final_exc = gen_paired_list+gen_singles_list
print("kUpCCGSD excitations list:", final_exc)
num_params = len(final_exc)
print('Number of parameters(operators) in the ansatz:',num_params)





initial_points=[0]*num_params
phis = [Parameter(f"phi{i}") for i in range(num_params)]
#print(phis)

qc = QuantumCircuit(num_qubits)

Hartree_Fock(qc)
#print(qc)

for n in range(len(final_exc)):
    if (len(final_exc[n][0]) == 2 and len(final_exc[n][1]) == 2):
        double_excitation_circ(qc, final_exc[n][0][0], final_exc[n][0][1], final_exc[n][1][0], final_exc[n][1][1], phis[n])
    if (len(final_exc[n][0]) == 1 and len(final_exc[n][1]) == 1):
        single_excitation_circ(qc, final_exc[n][0][0], final_exc[n][1][0], phis[n])
#print(qc)


# Calculation of CNOT count for the final ansatz
def count_cnot_gates(circuit):
    cnot_count = 0
    for gate in circuit.data:
        if gate[0].name == 'cx':
            cnot_count+=1
    return cnot_count

cnot_gates=count_cnot_gates(qc)
print("CNOT COUNT: ",cnot_gates)



# Perform VQE with the CNOT Efficient circuit
vqe_final = VQE(qc, optimizer=optimizer, quantum_instance=backend, initial_point=initial_points)
vqe_result_final = vqe_final.compute_minimum_eigenvalue(qubit_op)
E_final = np.real(vqe_result_final.eigenvalue) + Enuc
print("VQE Energy for the ansatz: ", E_final)


# Classical computation of energy
solver = NumPyMinimumEigensolverFactory()
calc = GroundStateEigensolver(converter, solver)
numpy_result = calc.solve(es_problem)
fci_energy = np.real(numpy_result.eigenenergies[0]) + Enuc
print("FCI ENERGY:", fci_energy)


print("\n")
print("Distance :", dist,' ',  "Number of parameters:", num_params,' ', "Energy (Our method)", E_final,' ', "FCI Energy:", fci_energy)




