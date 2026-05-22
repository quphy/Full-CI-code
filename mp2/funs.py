import numpy as np

def doubles(nocc, norb):
    #construct double excitation order, i,j in occ, ab in vir
    for i in range(nocc):
        for j in range(nocc):
            for a in range(nocc, norb):
                for b in range(nocc, norb):
                    yield i, j, a, b

def mp2(mo_energy, g2, nocc, norb):
    correlation = 0.0
    
    for i, j, a, b in doubles(nocc, norb):
        comb = 2 * (g2[i, a, j, b] ** 2)
        ex = g2[i, a, j, b] * g2[i, b, j, a]
        
        correlation += (comb - ex) / (mo_energy[i] + mo_energy[j] - mo_energy[a] - mo_energy[b])
        
    return correlation