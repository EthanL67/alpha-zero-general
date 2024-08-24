import numpy as np
from qubobrute.core import *
from qubobrute.simulated_annealing import simulate_annealing_gpu
from pyqubo import Spin

# -1 won
pieces = np.array([[-1, -1,  0],
                   [-1,  0,  1],
                   [ 0,  1,  1],
                   [0, 0, 0],
                   [0, 1, 0],
                   [0, 0, 0]]
                  )

v = 3


def qubo_energy(qubo: np.ndarray, offset: np.number, sample: np.ndarray) -> np.number:
    """Calculate the energy of a sample."""
    return np.dot(sample, np.dot(qubo, sample)) + offset


J = np.copy(pieces[:v, :])
np.fill_diagonal(J, 0)
vertices = np.diag(pieces[:v, :])

# Define binary variables and construct the Hamiltonian
spins = np.array([Spin(f'spin_{i}') for i in range(v)])
H = 0.5 * np.sum(J * np.outer(spins, spins))

# Compile the model to a binary quadratic model (BQM)
model = H.compile()
qubo, offset = model.to_qubo(index_label=True)

# Initialize the 2D NumPy array with zeros and fill it with qubo values
q = np.zeros((v, v))
for (i, j), value in qubo.items():
    q[i, j] = value

if v < 24:
    # brute-force
    energies = solve_gpu(q, offset)

    # Find the minimum energy and its indices
    min_energy = np.min(energies)
    min_indices = np.where(energies == min_energy)[0]

    # Generate unique solutions
    unique_solutions = {tuple(bits(idx, nbits=v)) for idx in min_indices}

else:
    # Simulated annealing
    energies, solutions = simulate_annealing_gpu(q, offset, n_iter=1000, n_samples=10000, temperature=1.0,
                                                 cooling_rate=0.99)

    # Find the minimum energy and its indices
    min_energy = np.min(energies)
    min_indices = np.where(energies == min_energy)[0]

    # Generate unique solutions
    unique_solutions = {tuple(solutions[idx]) for idx in min_indices}

# Equal probability for each ground state
prob = 1 / len(unique_solutions)

# Convert unique solutions to NumPy array
unique_solutions_np = np.array(list(unique_solutions))

# Calculate correlation and scores
C = np.corrcoef(unique_solutions_np, rowvar=False)
scores = np.sum(C, axis=1) - 1

print(scores)
