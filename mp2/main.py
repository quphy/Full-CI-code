import funs
import pyscf
from pyscf import scf
from pyscf import ao2mo

mol = pyscf.gto.Mole()
mol.atom = [
    ['H', ( 1.0, 0.0, 0.0)],
    ['O', ( 3.0, 0.0, 0.0)],
    ['H', ( 5.0, 0.0, 0.0)]
    ]
mol.basis  = 'sto-6g'
mol.build()
m = scf.RHF(mol)
m.kernel()
hf_energy = m.e_tot
mo_energy = m.mo_energy
nocc = mol.nelectron // 2
norb = m.mo_coeff.shape[1]
h1 = m.mo_coeff.T @ m.get_hcore() @ m.mo_coeff
g2 = ao2mo.kernel(mol, m.mo_coeff, compact=False)
g2 = g2.reshape(norb, norb, norb, norb)
corr_energy =funs.mp2(mo_energy, g2, nocc, norb)
total_mp2_energy = hf_energy + corr_energy

print("MP2 Energy", total_mp2_energy, "Hartree")