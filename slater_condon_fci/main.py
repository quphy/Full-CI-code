import funs
import pyscf
from pyscf import scf
from pyscf import ao2mo

mol = pyscf.gto.Mole()
mol.atom = [
    ['O', ( 1.0, 0.0, 0.0)],
    ['H', ( 3.0, 0.0, 0.0)],
    ['H', ( 8.0, 0.0, 0.0)],
    ['H', ( 1.0, 6.0, 0.0)],
    ['H', ( 1.0, 0.0,-7.0)]
    ]
mol.basis  = 'sto-6g'
mol.build()
m = scf.UHF(mol)
m.kernel()
norb = m.mo_coeff.shape[1]
nelec=mol.nelectron
nspin=mol.spin
h1 = m.mo_coeff.T @ m.get_hcore() @ m.mo_coeff
g2 = ao2mo.kernel(mol, m.mo_coeff, compact=False)
g2 = g2.reshape(norb, norb, norb, norb)
method="david"
eignvalue,eigenvector =funs.sc_fci(h1,g2,nelec,nspin,method,nroots=1)

print("Ground-state energy:", eignvalue[0],"Hartree")