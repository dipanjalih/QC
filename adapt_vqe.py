from qiskit_nature.units import DistanceUnit
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_algorithms import NumPyMinimumEigensolver
from qiskit_nature.second_q.algorithms import GroundStateEigensolver
from qiskit_nature.second_q.operators import FermionicOp
from sympy import Symbol
from qiskit_algorithms.optimizers import L_BFGS_B
from qiskit.primitives import Estimator
from qiskit_aer.primitives import Estimator as AerEstimator
from qiskit_nature.second_q.circuit.library import HartreeFock, UCCSD, PUCCSD
from qiskit_algorithms.minimum_eigensolvers import VQE
from qiskit_nature.second_q.operators import SpinOp
from qiskit_nature.second_q.properties import AngularMomentum
#from qiskit_nature.cost.estimators import PauliExpectation
from qiskit_algorithms.minimum_eigensolvers import AdaptVQE
from qiskit_nature.second_q.circuit.library import UCC
import random
import math
import numpy as np
import scipy
from scipy.optimize import minimize
from qiskit_nature.second_q.transformers import FreezeCoreTransformer

import warnings
warnings.filterwarnings('ignore')

import logging
logging.basicConfig(filename="lih_3_adapt_gsd.log", level=logging.INFO)
logger=logging.getLogger()
logger.setLevel(logging.DEBUG)



driver = PySCFDriver(
    atom='Li .0 .0 .0; H .0 .0 3.0',
    unit=DistanceUnit.ANGSTROM,
    basis='sto3g',
)
problem = driver.run()
print("Number of spatial orbitals:", problem.num_spatial_orbitals)
print("Number of electrons:", problem.num_particles)
print("Constants:", problem.hamiltonian.constants)
shift= (problem.hamiltonian.constants['nuclear_repulsion_energy'])
# setup the qubit mapper
mapper = JordanWignerMapper()
fermionic_op = problem.hamiltonian.second_q_op()
qubit_jw_op = mapper.map(fermionic_op)
Enuc= shift
print("Energy shift:", Enuc)



# FCI calculations
solver = GroundStateEigensolver(mapper,NumPyMinimumEigensolver(),)

result = solver.solve(problem)
exact_energy = result.eigenvalues[0]
print("===============")
print("Exact Energy with nuclear repulsion and shift: ", exact_energy + Enuc)
print("===============")



initial_state=HartreeFock(problem.num_spatial_orbitals,problem.num_particles,mapper,)



optimizer = L_BFGS_B(maxiter=500000)

# Print the Hartree Fock energy
estimator = Estimator()
hf_energy = estimator.run(initial_state , qubit_jw_op).result().values
print("Hartree Fock Energy:", hf_energy + Enuc)


'''
paired_list=[((0, 6), (2, 8)), ((0, 6), (3, 9)), ((0, 6), (4, 10)), ((0, 6), (5, 11)), ((1, 7), (2, 8)), ((1, 7), (3, 9)), ((1, 7), (4, 10)), ((1, 7), (5, 11))]
gen_singles_list=[]
for i in range(problem.num_spatial_orbitals):
    for j in range(problem.num_spatial_orbitals):
        if (i != j and j > i):
            gen_singles_list.append(((i, ), (j, )))
            gen_singles_list.append(((i+problem.num_spatial_orbitals, ), (j+problem.num_spatial_orbitals, )))

#print(gen_singles_list)


final_exc=paired_list
for i in range(len(gen_singles_list)):
    final_exc.append(gen_singles_list[i])

print(final_exc)


def custom_ex(num_spatial_orbitals,num_particles):
    excitations=final_exc
    return excitations


ansatz = UCC(num_spatial_orbitals=problem.num_spatial_orbitals, num_particles=problem.num_particles, excitations=custom_ex, qubit_mapper=mapper, initial_state= initial_state)
'''

ansatz = UCC(num_spatial_orbitals=problem.num_spatial_orbitals, num_particles=problem.num_particles, excitations='sd', generalized=True, qubit_mapper=mapper, initial_state= initial_state)


excitation_list = ansatz._get_excitation_list()
print('Number of parameters(operators) in the final ansatz:',len(excitation_list))
#n_params = ansatz.num_parameters
#print("Number of parameters in the final ansatz:", n_params)


print("###############################################################################################")
print("#########################################ADAPT Results#########################################")
print("###############################################################################################")

vqe = VQE(Estimator(), ansatz, L_BFGS_B())

adapt_vqe = AdaptVQE(vqe, gradient_threshold=1e-8, eigenvalue_threshold=1e-8)

#eigenvalue, _ = adapt_vqe.compute_minimum_eigenvalue(qubit_jw_op)
result=adapt_vqe.compute_minimum_eigenvalue(qubit_jw_op)
#print("Adapt energy:", np.real(result.eigenvalue)+Enuc)
final_circuit=result.optimal_circuit
final_parameters=final_circuit.num_parameters
final_list=final_circuit._get_excitation_list()

print("Final parameters", final_parameters)
print("Final list", final_list)
print("Distance:", 3.0, ' ', "Adapt energy:", np.real(result.eigenvalue)+Enuc, ' ', "Adapt parameters:", final_circuit.num_parameters)
print(result.eigenvalue_history)
print(result.optimal_parameters)
print("-----------------------------------------------------------------------------------------------------------------------------------")


