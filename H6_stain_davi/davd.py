import numpy as np

def davidson(H, nroots=20, maxiter=1000, tol=1e-6):
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

H = np.load("H.npy")
eigvals, eigvecs = davidson(H, nroots=1)
print("Ground-state energy:", eigvals[0],"Hartree")