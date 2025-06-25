from qiskit_algorithms import VarQITE, TimeEvolutionProblem
from qiskit.primitives import Estimator
import numpy as np
from qiskit_algorithms.time_evolvers.variational import ImaginaryMcLachlanPrinciple
from qiskit_algorithms.gradients import ReverseEstimatorGradient, ReverseQGT

def var_qite(H,circuit_ansatz,initial_parameters,dt,N,efficient = True):
    """
    Variational Quantum Imaginary Time Evolution (VarQITE) algorithm.
    
    Args:
        H (SparsePauliOp): Hamiltonian operator.
        circuit_ansatz (QuantumCircuit): Ansatz circuit for the variational form.
        initial_parameters (list): Initial parameters for the ansatz circuit.
        dt (float): Time step for the evolution.
        N (int): Number of time steps.
        efficient (bool): Flag to use efficient QITE or not.
    
    Returns:
        VarQITE: Instance of the VarQITE algorithm.
    """
    # Define Imaginary McLachlan Principle for Variational Time Evolution using Qiskit Function
    var_principle = ImaginaryMcLachlanPrinciple(qgt= ReverseQGT() if efficient else None, gradient= ReverseEstimatorGradient() if efficient else None)

    # Initial parameters (variation anstaz)
    init_param_values = {name:value for name,value in zip(circuit_ansatz.parameters,initial_parameters)}
    
    # Set up evolution problem (fix: add aux_operators)
    time = N * dt
    evolution_problem = TimeEvolutionProblem(H, time, aux_operators=[H])

    #  VarQITE using the github function
    var_qite = VarQITE(circuit_ansatz, init_param_values, var_principle, Estimator(),num_timesteps=N)
    evolution_result = var_qite.evolve(evolution_problem)

    # Return results
    return evolution_result


    
    