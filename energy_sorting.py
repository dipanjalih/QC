# imports
import qiskit as qt
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
# molecule geometry

import logging
logging.basicConfig(filename="bh_energy_sorting.log", level=logging.INFO)
logger=logging.getLogger()
logger.setLevel(logging.DEBUG)


# User-defined threshold for pruning operator space and number of repetitions.
threshold=1e-6
n_reps=1

#Optimizer and backend information
optimizer = L_BFGS_B(maxfun=500000)
backend = Aer.get_backend('statevector_simulator')


driver = PySCFDriver(atom='H 0.00000 0.00000 0.000000;H 0.00000 0.00000 1.000000; H 0.00000 0.00000 2.000000; H 0.00000 0.00000 3.000000', charge=0, spin=0, basis='sto3g')
es_problem = ElectronicStructureProblem(driver)


# Obtaining qubit Hamiltonian
mapper = JordanWignerMapper()
converter = QubitConverter(mapper=mapper, two_qubit_reduction=False)
second_q_op = es_problem.second_q_ops()
qubit_op = converter.convert(second_q_op['ElectronicEnergy'])

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


# Initial_state
init_state = HartreeFock(num_spin_orbitals, num_particles, converter)


#Evaluating Hartree-Fock Energy
#Create dummy parametrized circuit
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


# Variational Form for the ansatz
# Pick out only the rank-two excitations to start with
var_form = UCC(num_particles=num_particles, num_spin_orbitals=num_spin_orbitals, excitations="d", initial_state=init_state, qubit_converter=converter)
excitation_list = var_form._get_excitation_list()
print('Total number of UCCD Operators:',len(excitation_list))
print(excitation_list)
fer_excitation_op = var_form.excitation_ops()  # getting the second_q operator for excitations in UCC
#print(fer_excitation_op)
print("\n")




# For energy sorting one-parameter energy is evaluated
op_energy=[]
excitation_list_pauli = list()
for ex in range(len(fer_excitation_op)):
    pauli_exc = converter.convert(fer_excitation_op[ex]) 
    evolved_op = EvolvedOperatorAnsatz(pauli_exc, initial_state=init_state)
    vqe_one_p = VQE(evolved_op, optimizer=optimizer, quantum_instance=backend, initial_point=[0])
    vqe_result_one_p = vqe_one_p.compute_minimum_eigenvalue(qubit_op)
    opt_params_one_p = vqe_result_one_p.optimal_point
    opt_params_list = opt_params_one_p.tolist()
    optimized_circuit = evolved_op.bind_parameters(opt_params_list)
    E_one_p = np.real(vqe_result_one_p.eigenvalue) + Enuc
    # Calculating difference wrt. HF energy
    delta_E_hf = hf_energy - E_one_p
    #print(excitation_list[ex], E_one_p, delta_E_hf)
    # Pruning the operator pool based on ope-parameter energies
    if (delta_E_hf > threshold):
        op_energy.append((excitation_list[ex], delta_E_hf))
    
#print(op_energy)
print("\n")
print("====================================")
print("Sorted list of pruned T2 operators :")
print("====================================")
op_energy.sort(key=lambda x: x[1], reverse=True)
#print(op_energy)
#print(len(op_energy))

final_op_list = [exc for exc, _ in op_energy]
print(final_op_list)

# Append the singles at th end.
var_form_singles = UCC(num_particles=num_particles, num_spin_orbitals=num_spin_orbitals, excitations="s", initial_state=init_state, qubit_converter=converter)
singles = var_form_singles._get_excitation_list()
print("Singles:", singles)
print('Total number of singles:',len(singles))

print("Final Ansatz ordered operators list:")
final_op_list = final_op_list + singles
print(final_op_list)




# Running VQE on the final ansatz
def custom_ex_final_ansatz(num_spin_orbitals,num_particles):
    excitations_final=final_op_list
    return excitations_final

var_form_final = UCC(num_particles=num_particles,num_spin_orbitals=num_spin_orbitals, excitations=custom_ex_final_ansatz, initial_state=init_state, qubit_converter=converter, reps=n_reps)
exc_list_final = var_form_final._get_excitation_list()
n_params = var_form_final.num_parameters
print("Number of parameters in the final ansatz:", n_params)
decomposed_circuit=var_form_final.decompose().decompose().decompose()
vqe_final = VQE(var_form_final, optimizer=optimizer, quantum_instance=backend, initial_point=[0]*n_params)
vqe_result_final = vqe_final.compute_minimum_eigenvalue(qubit_op)
E_final = np.real(vqe_result_final.eigenvalue) + Enuc
print("VQE Energy for the ansatz: ", E_final)


