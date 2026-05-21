import numpy as np
import math
from itertools import combinations

# Z function in KH paper (eq11)
def cal_Z_element(k, l, N, M):
    if k == N:
        return l - N
    else:
        total_sum = 0
        for m in range(M - l + 1, M - k + 1):
            total_sum += (math.comb(m, N - k) - math.comb(m - 1, N - k - 1))
        return total_sum
    
# resort the order and confirm the phase
def resort_and_phase(configuration):
    sorted_config = sorted(configuration)
    
    inversions = 0
    n = len(configuration)
    for i in range(n):
        for j in range(i + 1, n):
            if configuration[i] > configuration[j]:
                inversions += 1
                
    if inversions % 2 == 0:
        phase = 1.0 
    else:
        phase = -1.0
    
    return phase, sorted_config

# calculate the index of configuration in KH paper (eq12)
def index_order(config, N, M):
    assert resort_and_phase(config)[1] == config, "Configuration must be sorted first"
    index = 0
    for k, l in enumerate(config):
        index += cal_Z_element(k + 1, l + 1, N, M)
    return index

# Listings that can be connected to a determinant via single-electron excitation.
def exication_list(ref, num_elec, num_orb):
    virtual = [i for i in range(num_orb) if i not in set(ref)]
    conf_list = []
    for i in ref:
        conf_list.append({
            "j->i": (i, i), # density matrix \gamma_{ij}=a^{\dagger}_j a_i
            "phase": 1.0,
            "index": index_order(ref, num_elec, num_orb)
        })

    for i in virtual:
        for index, j in enumerate(ref):
            excited_unsort = ref[:index] + [i] + ref[index + 1:]
            phase, excited = resort_and_phase(excited_unsort)
            conf_list.append({
                "j->i": (i, j), # density matrix \gamma_{ij}=a^{\dagger}_j a_i
                "phase": phase,
                "index": index_order(excited, num_elec, num_orb)
            })
   
    return conf_list

# construct CI basis 
def comb_bit(n, k):
    return [list(c) for c in combinations(range(n), k)]

def construct(h1, g2, num_elec, num_spin):
    norb = h1.shape[0]
    assert h1.shape == (norb, norb), "One electron integral is not a square matrix"
    assert g2.shape == (norb, norb, norb, norb), "Two electron integral is not a equal-dimensional fourth-order tensor"
    nalpha = (num_elec + num_spin) // 2
    nbeta = (num_elec - num_spin) // 2

    alpha_combos = comb_bit(norb, nalpha)
    beta_combos  = comb_bit(norb, nbeta)

    beta_excited = []
    alpha_excited = []
    for i in range(len(beta_combos)):  
        beta_excited.append(exication_list(beta_combos[i], nbeta, norb))
    for i in range(len(alpha_combos)):  
        alpha_excited.append(exication_list(alpha_combos[i], nalpha, norb))

    return {
        "h1": h1,
        "g2": g2,
        "norb": norb,
        "alpha_combos": alpha_combos,
        "beta_combos": beta_combos,
        "nalpha": nalpha,
        "nbeta": nbeta,
        "alpha_excited": alpha_excited,
        "beta_excited": beta_excited,
    }

# construct the H[i,i] for Davidson
def diag_element(data):
    h1 = data["h1"]
    g2 = data["g2"]
    alpha_combos = data["alpha_combos"]
    beta_combos = data["beta_combos"]
    ndet = len(alpha_combos) * len(beta_combos)
    
    diag = np.zeros((ndet))

    basis = []
    for beta_occ in beta_combos:
        for alpha_occ in alpha_combos:
            basis.append((alpha_occ, beta_occ))
            
    # construct diagonal element
    for I, (alphaI, betaI) in enumerate(basis):
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
                e += g2[p, p, q, q]

        diag[I] = e     

    return diag

# function that perform H on c (c'=Hc) without storing H
def sigma(ci, data):
    h1 = data["h1"]
    g2 = data["g2"]
    norb = data["norb"]
    alpha_combos = data["alpha_combos"]
    beta_combos = data["beta_combos"]
    alpha_excited = data["alpha_excited"]
    beta_excited = data["beta_excited"]
    
    num_alpha = len(alpha_combos)
    ndet = num_alpha * len(beta_combos)
    
    E_tensor = np.zeros((ndet, norb, norb))
    K = h1 - 0.5 * np.einsum("ijjk -> ik", g2)
    
    # calculate eq (6a) alpha part
    for ia, alpha_excitation in enumerate(alpha_excited):
        for alpha_config in alpha_excitation:
            for ib in range(len(beta_combos)):
                i, j = alpha_config["j->i"]
                ci_index = ia + ib * num_alpha
                E_tensor_index = alpha_config["index"] + ib * num_alpha
                E_tensor[E_tensor_index, i, j] += alpha_config["phase"] * ci[ci_index]
    
    # calculate eq (6a) one electron beta part
    for ib, beta_excitation in enumerate(beta_excited):
        for beta_config in beta_excitation:
            for ia in range(num_alpha):
                i, j = beta_config["j->i"]
                ci_index = ia + ib * num_alpha
                E_tensor_index = ia + beta_config["index"] * num_alpha
                E_tensor[E_tensor_index, i, j] += beta_config["phase"] * ci[ci_index] 

    # eq(6b) contract two_electron_integral with E_tensor
    g2_contract = np.einsum("Ikl,ijkl->Iij", E_tensor, g2)
    
    # 1 electron part
    Hc = np.einsum("Iij, ij ->I", E_tensor, K)
    
    # 2 electron part - alpha part
    for ia, alpha_excitation in enumerate(alpha_excited):
        for alpha_config in alpha_excitation:
            for ib in range(len(beta_combos)):
                i, j = alpha_config["j->i"]
                ci_index = ia + ib * num_alpha
                E_tensor_index = alpha_config["index"] + ib * num_alpha
                Hc[ci_index] += 0.5 * alpha_config["phase"] * g2_contract[E_tensor_index, i, j]

    # 2 electron part - beta part
    for ib, beta_excitation in enumerate(beta_excited):
        for beta_config in beta_excitation:
            for ia in range(num_alpha):
                i, j = beta_config["j->i"]
                ci_index = ia + ib * num_alpha
                E_tensor_index = ia + beta_config["index"] * num_alpha
                Hc[ci_index] += 0.5 * beta_config["phase"] * g2_contract[E_tensor_index, i, j]

    return Hc

def Davidson(sigma, diag, data, nroots, maxiter, tol):
    N = diag.shape[0]
    # random orthogonal vector as initial guess
    V = np.random.rand(N, nroots)
    V, _ = np.linalg.qr(V)

    
    HV = np.column_stack([sigma(V[:, i], data) for i in range(V.shape[1])])

    for it in range(maxiter):
        T = V.T @ HV
        evals, evecs = np.linalg.eigh(T)
        ritz = V @ evecs[:, :nroots]
        ritz_vals = evals[:nroots]

        converged = True
        new_vecs = []
        new_Hvecs = []
        for i in range(nroots):
            Hv = HV @ evecs[:, i]
            res = Hv - ritz_vals[i] * ritz[:, i]
            if np.linalg.norm(res) > tol:
                converged = False
                # orthogonalization and add it in subspace  
                q = res / (diag - ritz_vals[i] + 1e-12)
                q = q - V @ (V.T @ q) 
                if len(new_vecs) > 0:
                    Q_new = np.column_stack(new_vecs)
                    q = q - Q_new @ (Q_new.T @ q)
                norm_q = np.linalg.norm(q)
                if norm_q > 1e-10:
                    q = q / norm_q
                    new_vecs.append(q)  
                    new_Hvecs.append(sigma(q, data)) 
        if converged:
            print(f"Davidson converged in {it} iterations")
            return ritz_vals, ritz

        if new_vecs:
            V = np.column_stack([V] + new_vecs)
            HV = np.column_stack([HV] + new_Hvecs)

    print("Davidson did not converge fully")
    return ritz_vals, ritz

def kh_fci(h1, g2, num_elec, num_spin, nroots=20, maxiter=1000, tol=1e-6):
    data = construct(h1, g2, num_elec, num_spin)
    diag = diag_element(data)
    ndet = len(data["alpha_combos"]) * len(data["beta_combos"])
    nroots = min(nroots, ndet)
    eignvalue, eignvector = Davidson(sigma, diag, data, nroots, maxiter, tol)
    return eignvalue, eignvector