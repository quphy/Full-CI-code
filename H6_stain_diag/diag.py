import numpy as np

nelec = 6
nspin = 0   
nalpha = (nelec + nspin) // 2
nbeta  = (nelec - nspin) // 2

# 读取积分
h1 = np.load("h1e.npy")  
g2 = np.load("h2e.npy")   

norb = h1.shape[0]
assert h1.shape == (norb, norb)
assert g2.shape == (norb, norb, norb, norb)

# construct CI basis
def comb_bit(n, k):
    result = []
    for mask in range( 2**n ):
        if bin(mask).count("1") == k:
            comb = []
            for i in range(n):
                if mask & (1 << i):
                    comb.append(i)
            result.append(comb)
    return result

alpha_combos = comb_bit(norb, nalpha)
beta_combos  = comb_bit(norb, nbeta)

basis = []
for beta_occ in beta_combos:
    for alpha_occ in alpha_combos:
        basis.append((alpha_occ, beta_occ))

ndet = len(basis)

# construct Hamiltonian in CI basis
H = np.zeros((ndet, ndet))

for I, (alphaI, betaI) in enumerate(basis):
    # construct diagonal element
    e = 0.0
    # one electron part
    for p in alphaI:
        e += h1[p, p]
    for p in betaI:
        e += h1[p, p]

    # two electron part of the same spin
    for occ in (alphaI, betaI):
        for p in occ:
            for q in occ:
                e += 0.5 * (g2[p, p, q, q] - g2[p, q, q, p])

    # two electron part of different spins
    for p in alphaI:
        for q in betaI:
            e += g2[p, p,  q, q]

    H[I, I] = e

    # construct nondiagonal part
    for J in range(I + 1, ndet):
        alphaJ, betaJ = basis[J]

        # constrcut the excitation amplitude
        alpha_bra_only = sorted(set(alphaI) - set(alphaJ))
        alpha_ket_only = sorted(set(alphaJ) - set(alphaI))
        beta_bra_only  = sorted(set(betaI) - set(betaJ))
        beta_ket_only  = sorted(set(betaJ) - set(betaI))

        n_exc = len(alpha_bra_only) + len(beta_bra_only)
        if n_exc > 2:
            continue  

        # construct the phase
        def phase_string(exc, ori):
            occ = ori[:]
            phase = 1
            for h in sorted(set(ori)-set(exc)):
                idx = occ.index(h)
                phase *= -1 if (idx % 2) else 1
                occ.pop(idx)
            for c in sorted(set(exc)-set(ori), reverse=True):
                idx = sum(o < c for o in occ)
                phase *= -1 if (idx % 2) else 1
                occ.insert(idx, c)
            if occ != exc:
                return 0
            return phase

        phase = phase_string(alphaI, alphaJ) * phase_string(betaI, betaJ)

        val = 0.0

        # one excitation part
        if n_exc == 1:
           
            if len(alpha_bra_only) == 1:
                a = alpha_bra_only[0]
                i = alpha_ket_only[0]
                
                val = h1[a, i]
                
                shared_alpha = sorted(set(alphaI).intersection(alphaJ))
                shared_beta  = sorted(set(betaI).intersection(betaJ))
                for j in shared_alpha + shared_beta:
                    val += g2[a, i, j, j]
                for j in shared_alpha:
                    val -= g2[a, j, j, i]
            
            elif len(beta_bra_only) == 1:
                a = beta_bra_only[0]
                i = beta_ket_only[0]

                val = h1[a, i]

                shared_alpha = sorted(set(alphaI).intersection(alphaJ))
                shared_beta  = sorted(set(betaI).intersection(betaJ))

                for j in shared_alpha + shared_beta:
                    val += g2[a, i, j, j]
                for j in shared_beta:
                    val -= g2[a, j, j, i]

        # two exictation part
        elif n_exc == 2:
            
            if len(alpha_bra_only) == 2 and len(beta_bra_only) == 0:
                a, b = alpha_bra_only
                i, j = alpha_ket_only
                val = g2[a, i,b, j] - g2[a,  j, b, i]
            elif len(beta_bra_only) == 2 and len(alpha_bra_only) == 0:
                a, b = beta_bra_only
                i, j = beta_ket_only
                val = g2[a, i, b, j] - g2[a, j, b, i]
            
            elif len(alpha_bra_only) == 1 and len(beta_bra_only) == 1:
                a = alpha_bra_only[0]
                i = alpha_ket_only[0]
                b = beta_bra_only[0]
                j = beta_ket_only[0]
                val = g2[a, i, b, j]

        H[I, J] = val * phase
        H[J, I] = H[I, J]

np.save("H.npy",H) #store the Hamiltonian

# diagonalization
eigvals, eigvecs = np.linalg.eigh(H)
print("Ground State Energy:", eigvals[0],"Hartree")
