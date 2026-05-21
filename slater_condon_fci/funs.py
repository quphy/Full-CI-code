import numpy as np
from itertools import combinations

#construct the basis
def comb_bit(n, k):
    return [list(c) for c in combinations(range(n), k)]

def construct_basis(norb,nalpha,nbeta):
    alpha_combos = comb_bit(norb, nalpha)
    beta_combos  = comb_bit(norb, nbeta)

    basis = []
    for beta_occ in beta_combos:
        for alpha_occ in alpha_combos:
            basis.append((alpha_occ, beta_occ))
    return basis

#calculate the phase between 2 cofigurations
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

# construct Hamiltonian in CI basis
def construct_hamiltonian(h1,g2,num_elec,num_spin):
    norb = h1.shape[0]
    assert h1.shape == (norb, norb), "One electron integral is not a square matrix"
    assert g2.shape == (norb, norb, norb, norb), "Two electron integral is not a equal-dimensional fourth-order tensor"
    nalpha = (num_elec + num_spin) // 2
    nbeta = (num_elec - num_spin) // 2
    basis= construct_basis(norb,nalpha,nbeta)
    ndet = len(basis)
    
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
        
        # two electron part of different spins
        for J in range(I+1,ndet):
            alphaJ, betaJ = basis[J]

            # constrcut the excitation amplitude
            alpha_bra_only = sorted(set(alphaI) - set(alphaJ))
            alpha_ket_only = sorted(set(alphaJ) - set(alphaI))
            beta_bra_only  = sorted(set(betaI) - set(betaJ))
            beta_ket_only  = sorted(set(betaJ) - set(betaI))

            n_exc = len(alpha_bra_only) + len(beta_bra_only)
            if n_exc >2:
                continue
            
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
    return H

def davidson(H, nroots, maxiter, tol):
    N = H.shape[0]
            
    # extract diagonal elements for preconditioning and initial guess
    diag_H = np.diag(H)
    
    # unit vectors corresponding to the smallest nroots diagonal elements as initial guess
    idx_sort = np.argsort(diag_H)
    V = np.zeros((N, nroots))
    for i in range(nroots):
        V[idx_sort[i], i] = 1.0
    
    # precompute and maintain HV to avoid full H @ V matrix multiplication in the loop
    HV = H @ V

    for it in range(maxiter):
        # calculate the projection matrix T = V^T * H * V
        T = V.T @ HV
        # solve the subspace eigenvalue problem
        evals, evecs = np.linalg.eigh(T)
        # extract the current ritz values and ritz vectors
        ritz_vals = evals[:nroots]
        ritz_vecs = V @ evecs[:, :nroots]
        H_ritz = HV @ evecs[:, :nroots]
        
        converged_count = 0
        new_dirs = []
        
        for i in range(nroots):
            # compute the residual
            res = H_ritz[:, i] - ritz_vals[i] * ritz_vecs[:, i]
            
            if np.linalg.norm(res) < tol:
                converged_count += 1
            else:
                diff = diag_H - ritz_vals[i]
                diff[np.abs(diff) < 1e-12] = 1e-12
                q = res / diff
                new_dirs.append(q)
                
        if converged_count == nroots:
            print(f"Davidson converged in {it+1} iterations")
            return ritz_vals, ritz_vecs
            
        # orthogonalization of new vector
        if new_dirs:
            q_mat = np.column_stack(new_dirs)
            for _ in range(2): 
                q_mat = q_mat - V @ (V.T @ q_mat)

            q_orth, R = np.linalg.qr(q_mat)

            # filter out vectors with small norms
            valid_cols = np.abs(np.diag(R)) > 1e-8
            q_orth = q_orth[:, valid_cols]
            
            if q_orth.shape[1] > 0:
                # expand subspace 
                V = np.hstack([V, q_orth])

                HV = np.hstack([HV, H @ q_orth]) 

    print("Davidson did not converge fully")
    return ritz_vals, ritz_vecs

def sc_fci(h1, g2, num_elec, num_spin, method = "david", nroots=20, maxiter=1000, tol=1e-6):
    H=construct_hamiltonian(h1,g2,num_elec,num_spin)
    if method == "diag":
        eigvalues, eigvectors = np.linalg.eigh(H)
    elif method == "david":
        eigvalues, eigvectors = davidson(H, nroots=20, maxiter=1000, tol=1e-6)
    else:
        raise ValueError(f"unknown method '{method}'. only 'david' and 'diag' are supported.")

    return eigvalues,eigvectors
