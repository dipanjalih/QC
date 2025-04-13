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
logging.basicConfig(filename="energy_landscape.log", level=logging.INFO)
logger=logging.getLogger()
logger.setLevel(logging.DEBUG)



# Define the user defined input parameters
threshold=1e-6
n_reps=1

#Optimizer and backend information
optimizer = L_BFGS_B(maxfun=100000)
backend = Aer.get_backend('statevector_simulator')



# Molecular geometry
dist=2.5
driver = PySCFDriver(atom='B 0.00000 0.00000 0.000000;H 0.00000 0.00000 2.50000', charge=0, spin=0, basis='sto3g')
# Freezing the core
tf = FreezeCoreTransformer(freeze_core=True)
es_problem = ElectronicStructureProblem(driver, transformers=[tf])


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
Enuc = es_energy.nuclear_repulsion_energy + np.real(es_energy._shift['FreezeCoreTransformer'])
print("Total energy shift:", Enuc)
print("Number of spin orbitals:", num_spin_orbitals)
print("Number of electrons:", num_particles)
num_spatial_orbitals=int(num_spin_orbitals/2)
print("Number of spatial orbitals:", num_spatial_orbitals)


#initial_state
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


# Defining a custom chemistry motivated ansatz
final_ansatz=[((1, 6), (2, 7)), ((1,), (2,)), ((6,), (7,)), ((0,), (2,)), ((5,), (7,)), ((0,), (1,)), ((5,), (6,)), ((0, 5), (3, 8)), ((0,), (1,)), ((5,), (6,)), ((1,), (2,)), ((6,), (7,)), ((0,), (2,)), ((5,), (7,)), ((0, 5), (4, 9)), ((0,), (1,)), ((5,), (6,)), ((1,), (2,)), ((6,), (7,)), ((0,), (2,)), ((5,), (7,)), ((0, 5), (2, 7)), ((0,), (1,)), ((5,), (6,)), ((1,), (2,)), ((6,), (7,)), ((1, 6), (3, 8)), ((0,), (1,)), ((5,), (6,)), ((1, 6), (4, 9)), ((0,), (1,)), ((5,), (6,))]


block=[]
init_point=[0]
energy_op_list=[]
selected_energy_op_list=[]

for i in range(len(final_ansatz)):
    # Growing the ansatz
    block.append(final_ansatz[i])
    # Running VQE blockwise
    def custom_ex_final_ansatz(num_spin_orbitals,num_particles):
        excitations_final=block
        return excitations_final
    var_form_final = UCC(num_particles=num_particles,num_spin_orbitals=num_spin_orbitals, excitations=custom_ex_final_ansatz, initial_state=init_state, qubit_converter=converter, reps=n_reps)
    excitation_list_final = var_form_final._get_excitation_list()
    n_params = var_form_final.num_parameters
    decomposed_circuit=var_form_final.decompose().decompose().decompose()
    vqe_final = VQE(var_form_final, optimizer=optimizer, quantum_instance=backend, initial_point=init_point) 
    vqe_result_final = vqe_final.compute_minimum_eigenvalue(qubit_op)
    E_final = np.real(vqe_result_final.eigenvalue) + Enuc
    print("\n")    
    # Get the optimal parameters
    opt_params = vqe_result_final.optimal_point
    print("Block number:", i+1, ' ', "Number of parameters:", n_params,' ', "Energy (Our method):", E_final) 
    opt_params_list = opt_params.tolist()
    opt_params_list.append(0)
    init_point=np.array(opt_params_list)
