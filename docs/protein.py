#!/usr/bin/env python
# coding: utf-8

# # Protein Folding

# ### Introduction

# The structure and function of many natural and human-engineered
# proteins is still only poorly understood. As a result, our understanding of
# processes connected with protein folding, such as those encountered in
# Alzheimer’s disease, vaccine development, and crop improvement
# research, has remained limited.
# 
# Unfolded polypeptides have a very large number of degrees of freedom
# and thus an enormous number of potential conformations. For example, a
# chain with $100$ aminoacids has on the order of $10^{47}$ conformations. In
# reality, however, many proteins fold to their native structure within
# seconds. This is known as Levinthal’s paradox [1].
# 
# The exponential growth of potential conformations with chain length
# makes the problem intractable for classical computers. In the quantum
# framework, our resource-efficient algorithm scales linearly with
# the number of aminoacids N.

# The goal of this work is to determine the minimum energy conformation of a protein. Starting from a random configuration, the protein's structure is optimized to lower the energy. This can be achieved by encoding the protein folding problem into a qubit operator and ensuring that all physical constraints are satisfied. 

# For the problem encoding we use: 
# 
# - Configuration qubits: qubits that are used to describe the configurations and the relative position of the different beads
# 
# - Interaction qubits: qubits that encode interactions between the different aminoacids
# 
# For our case we use a tetrahedral lattice (diamond shape lattice) where we encode the movement through the configuration qubits (see image below). 
# 
# <img src="aux_files/lattice_protein.png" width="300">

# The Hamiltonian of the system for a set of qubits $\mathbf{q}=\{\mathbf{q}_{cf}, \mathbf{q}_{in}\}$ is 
# 
# $$H(\mathbf{q}) = H_{gc}(\mathbf{q}_{cf}) + H_{ch}(\mathbf{q}_{cf}) + H_{in}(\mathbf{q}_{cf}, \mathbf{q}_{in}) $$
# 
# where 
# 
# - $H_{gc}$ is the geometrical constraint term (governing the growth of the primary sequence of aminoacids without bifurcations)
# 
# - $H_{ch}$ is the chirality constraint (enforcing the right stereochemistry for the system)
# 
# - $H_{in}$ is the interaction energy terms of the system. In our case we consider only nearest neighbor interactions. 

# Further details about the used model and the encoding of the problem can be found in [2].

# In[ ]:


from protein_folding.interactions.random_interaction import (
    RandomInteraction,
)
from protein_folding.interactions.miyazawa_jernigan_interaction import (
    MiyazawaJerniganInteraction,
)
from protein_folding.peptide.peptide import Peptide
from protein_folding.protein_folding_problem import (
    ProteinFoldingProblem,
)

from protein_folding.penalty_parameters import PenaltyParameters

from qiskit.utils import algorithm_globals, QuantumInstance

from time import perf_counter

algorithm_globals.random_seed = 23


# ### Protein Main Chain

# The Protein consists of a main chain that is a linear chain of aminoacids. For the naming of different residues we use the one-letter code as defined in Ref. [3]. Further details about the naming and the type of aminoacids can also be found in [4].
# 
# For this particular case we demonstrate the generation of the qubit operator in a neuropeptide with the main chain consisting of 7 aminoacids with letter codes APRLRFY (see also [2]).

# In[ ]:


main_chain = "APRLRFY"
main_chain = "MAPAGVSLRATILCLLAWAGLAAGDRVYIHPFHLVIHNESTCEQLAKANAGKPKDPTFIPAPIQAKTSPVDEKALQDQLVLVAAKLDTEDKLRAAMVGMLANFLGFRIYGMHSELWGVVHGATVLSPTAVFGTLASLYLGALDHTADRLQAILGVPWKDKNCTSRLDAHKVLSALQAVQGLLVAQGRADSQAQLLLSTVVGVFTAPGLHLKQPFVQGLALYTPVVLPRSLDFTELDVAAEKIDRFMQAVTGWKTGCSLMGASVDSTLAFNTYVHFQGSCSVTQAAV"
main_chain = "MAPAGVSL"


# ### Side Chains

# Beyond the main chain of the protein there may be aminoacids attached to the residues of the main chain. Our model allows for side chains of the maximum length of one. Elongated side chains would require the introduction of additional penalty terms which are still under development. In this example we do not consider any side chains to keep the real structure of the neuropeptide. 

# In[ ]:


side_chains = [""] * len(main_chain)


# ### Interaction between Aminoacids

# For the description of inter-residue contacts for proteins we use knowledge-based (statistical) potentials derived using quasi-chemical approximation. The potentials used here are introduced by Miyazawa, S. and Jernigan, R. L. in [5]. 
# 
# Beyond this model we also allow for random contact maps (interactions) that provide a random interaction map. One can also introduce a custom interaction map that enhances certain configurations of the protein (e.g. alpha helix, beta sheet etc). 

# In[ ]:


random_interaction = RandomInteraction()
mj_interaction = MiyazawaJerniganInteraction()


# ### Physical Constraints

# To ensure that all physical constraints are respected we introduce penalty functions. The different penalty terms used are: 
# 
# - penalty_chiral: A penalty parameter used to impose the right chirality.
# 
# - penalty_back: A penalty parameter used to penalize turns along the same axis. This term is used to eliminate sequences where the same axis is chosen twice in a row. In this way we do not allow for a chain to fold back into itself.
# 
# - penalty_1: A penalty parameter used to penalize local overlap between beads within a nearest neighbor contact.

# In[ ]:


penalty_back = 10
penalty_chiral = 10
penalty_1 = 10

penalty_terms = PenaltyParameters(penalty_chiral, penalty_back, penalty_1)


# ### Peptide Definition
# 

# Based on the main chain and possible side chains we define the peptide object that includes all the structural information of the modeled system.

# In[ ]:

start = perf_counter()
print("Creating peptide object...", end='')
peptide = Peptide(main_chain, side_chains)
print(f"done. ({perf_counter() - start:.2f}s)")

# ### Protein Folding Problem 

# Based on the defined peptide, the interaction (contact map) and the penalty terms we defined for our model we define the protein folding problem that returns qubit operators.
# 

# In[ ]:

start = perf_counter()
print("Creating protein folding problem...", end='')
protein_folding_problem = ProteinFoldingProblem(peptide, mj_interaction, penalty_terms)
print(f"done. ({perf_counter() - start:.2f}s)")

start = perf_counter()
print("Constructing qubit operator...", end='')
qubit_op = protein_folding_problem.qubit_op()
print(f"done. ({perf_counter() - start:.2f}s)")

# In[ ]:


print(qubit_op)


# ### Using VQE with CVaR expectation value for the solution of the problem

# The problem that we are tackling has now implemented all the physical constraints and has a diagonal Hamiltonian. For the particular case we are targeting the single bitstring that gives us the minimum energy (corresponding to the folded structure of the protein). Thus, we can use the Variational Quantum Eigensolver with Conditional Value at Risk (CVaR) expectation values for the solution of the problem and for finding the minimum configuration energy [6] . We follow the same approach as in Ref. [2] but here we use COBYLA for the classical optimization part. One can also use the standard VQE or QAOA algorithm for the solution of the problem, though as discussed in Ref. [2] CVaR is more suitable. 

# In[ ]:


from qiskit.circuit.library import RealAmplitudes
from qiskit.algorithms.optimizers import COBYLA
from qiskit.algorithms import NumPyMinimumEigensolver
from qiskit.algorithms.minimum_eigensolvers import SamplingVQE
from qiskit import execute, Aer
from qiskit.primitives import Sampler

# set classical optimizer
optimizer = COBYLA(maxiter=50)

# set variational ansatz
ansatz = RealAmplitudes(reps=1)

counts = []
values = []


def store_intermediate_result(eval_count, parameters, mean, std):
    counts.append(eval_count)
    values.append(mean)


# initialize VQE using CVaR with alpha = 0.1
vqe = SamplingVQE(
    Sampler(),
    ansatz=ansatz,
    optimizer=optimizer,
    aggregation=0.1,
    callback=store_intermediate_result,
)

start = perf_counter()
print("Computing minimum eigenvalue...", end='')
raw_result = vqe.compute_minimum_eigenvalue(qubit_op)
print(f"done. ({perf_counter() - start:.2f}s)")
print(raw_result)


# In[ ]:


import matplotlib.pyplot as plt

fig = plt.figure()

plt.plot(counts, values)
plt.ylabel("Conformation Energy")
plt.xlabel("VQE Iterations")

fig.add_axes([0.44, 0.51, 0.44, 0.32])

plt.plot(counts[40:], values[40:])
plt.ylabel("Conformation Energy")
plt.xlabel("VQE Iterations")
# plt.show()
plt.savefig("conformation_energy.png")


# ### Visualizing the answer

# In order to reduce computational costs, we have reduced the problem's qubit operator to the minimum amount of qubits needed to represent the shape of the protein. In order to decode the answer we need to understand how this has been done.
# * The shape of the protein has been encoded by a sequence of turns , $\{0,1,2,3\}$. Each turn represents a different direction in the lattice.
# * For a main bead of $N_{aminoacids}$ in a lattice, we need $N_{aminoacids}-1$ turns in order to represent its shape. However, the orientation of the protein is not relevant to its energy. Therefore the first two turns of the shape can be set to $[1,0]$ without loss of generality.
# * If the second bead does not have any side chain, we can also set the $6^{th}$ qubit to $[1]$ without breaking symmetry.
# * Since the length of the secondary chains is always limited to $1$ we only need one turn to describe the shape of the chain.
# 
# The total amount of qubits we need to represent the shape of the protein will be $2(N_{aminoacids}-3)$ if there is a secondary chain coming out of the second bead or $2(N_{aminoacids}-3) - 1$, otherwise. All the other qubits will remain unused during the optimization process. See:

# In[ ]:


result = protein_folding_problem.interpret(raw_result=raw_result)
print(
    "The bitstring representing the shape of the protein during optimization is: ",
    result.turn_sequence,
)
print("The expanded expression is:", result.get_result_binary_vector())


# Now that we know which qubits encode which information, we can decode the bitstring into the explicit turns that form the shape of the protein.

# In[ ]:


print(
    f"The folded protein's main sequence of turns is: {result.protein_shape_decoder.main_turns}"
)
print(f"and the side turn sequences are: {result.protein_shape_decoder.side_turns}")


# From this sequence of turns we can get the cartesian coordinates of each of the aminoacids of the protein.

# In[ ]:


# print(result.protein_shape_file_gen.get_xyz_data())


# And finally, we can also plot the structure of the protein in 3D. Note that when rendered with the proper backend this plot can be interactively rotated.

# In[ ]:


fig = result.get_figure(title="Protein Structure", ticks=False, grid=True)
fig.get_axes()[0].view_init(10, 70)
plt.savefig("protein_structure_1.png")


# And here is an example with side chains.

# In[ ]:


# peptide = Peptide("APRLR", ["", "", "F", "Y", ""])
# protein_folding_problem = ProteinFoldingProblem(peptide, mj_interaction, penalty_terms)
# qubit_op = protein_folding_problem.qubit_op()

# # set classical optimizer
# optimizer = COBYLA(maxiter=50)

# # set variational ansatz
# ansatz = RealAmplitudes(reps=1)

# counts = []
# values = []


# def store_intermediate_result(eval_count, parameters, mean, std):
#     counts.append(eval_count)
#     values.append(mean)


# # initialize VQE using CVaR with alpha = 0.1
# vqe = SamplingVQE(
#     Sampler(),
#     ansatz=ansatz,
#     optimizer=optimizer,
#     aggregation=0.1,
#     callback=store_intermediate_result,
# )

# raw_result = vqe.compute_minimum_eigenvalue(qubit_op)
# result_2 = protein_folding_problem.interpret(raw_result=raw_result)


# In[ ]:


# fig = result_2.get_figure(title="Protein Structure", ticks=False, grid=True)
# fig.get_axes()[0].view_init(10, 60)
# fig.savefig("protein_structure_2.png")


from Bio.PDB import PDBIO, Structure, Model, Chain, Residue, Atom
import numpy as np


# Example usage
protein_coordinates = result.protein_shape_file_gen.get_xyz_data()
protein_sequence = main_chain

result.protein_shape_file_gen.save_xyz_file("protein_structure", replace=True)

exit()

# Function to create a PDB structure from the protein coordinates
def create_pdb_structure(protein_coordinates, protein_sequence):
    structure = Structure.Structure("protein")
    model = Model.Model(0)
    chain = Chain.Chain("A")
    
    for i, (residue_name, coord) in enumerate(zip(protein_sequence, protein_coordinates)):
        coord = coord[1:].astype(float)
        residue = Residue.Residue((' ', i, ' '), residue_name, ' ')
        print(residue_name, coord)
        atom = Atom.Atom('CA', coord, 1, 1, ' ', 'CA', i, 'C')
        residue.add(atom)
        chain.add(residue)
    
    model.add(chain)
    structure.add(model)
    return structure

# Convert the coordinates to a numpy array
protein_coordinates = np.array(protein_coordinates)

# Create the PDB structure
pdb_structure = create_pdb_structure(protein_coordinates, protein_sequence)

# Save the structure to a PDB file
io = PDBIO()
io.set_structure(pdb_structure)
io.save("protein_structure.pdb")


# ### References

# <font size='2'>[1] https://en.wikipedia.org/wiki/Levinthal%27s_paradox </font>
# 
# <font size='2'>[2] A.Robert, P.Barkoutsos, S.Woerner and I.Tavernelli, Resource-efficient quantum algorithm for protein folding, NPJ Quantum Information, 2021, https://doi.org/10.1038/s41534-021-00368-4 </font>
# 
# <font size="2">[3] IUPAC–IUB Commission on Biochemical Nomenclature (1972). "A one-letter notation for aminoacid sequences". Pure and Applied Chemistry. 31 (4): 641–645. doi:10.1351/pac197231040639. PMID 5080161.</font> <br>
# 
# <font size="2">[4] https://en.wikipedia.org/wiki/Amino_acid</font>
# 
# <font size="2"> [5] S. Miyazawa and R. L.Jernigan, Residue – Residue Potentials with a Favorable Contact Pair Term and an Unfavorable High Packing Density Term for Simulation and Threading, J. Mol. Biol.256, 623–644, 1996, Table 3, https://doi.org/10.1006/jmbi.1996.0114 </font>
# 
# <font size="2"> [6] P.Barkoutsos, G. Nannichini, A.Robert, I.Tavernelli, S.Woerner, Improving Variational Quantum Optimization using CVaR, Quantum 4, 256, 2020, https://doi.org/10.22331/q-2020-04-20-256  </font>

# In[ ]:


# import qiskit.tools.jupyter

# get_ipython().run_line_magic('qiskit_version_table', '')
# get_ipython().run_line_magic('qiskit_copyright', '')

