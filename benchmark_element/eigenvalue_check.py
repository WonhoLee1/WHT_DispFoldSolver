import numpy as np
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dispsolver.element_shell.mitc3_plus_numba import compute_mitc3_plus_stiffness
from dispsolver.element_shell.mitc4_plus_numba import compute_mitc4_plus_stiffness

def check_eigenvalues():
    print("==================================================")
    print(" Eigenvalue Check (Rigid Body Modes Test)")
    print("==================================================")
    
    E = 2.0e11
    nu = 0.3
    thickness = 0.01
    
    # MITC3 Single Element
    coords3 = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.2],
        [0.0, 1.0, -0.1]
    ])
    K3 = compute_mitc3_plus_stiffness(coords3, E, nu, thickness)
    eigvals3 = np.linalg.eigvalsh(K3)
    zero_eigs3 = np.sum(np.abs(eigvals3) < 1e-4)
    print(f"[MITC3+] Total DOFs: {K3.shape[0]}")
    print(f"         Zero eigenvalues (RBMs expected = 6): {zero_eigs3}")
    print(f"         Lowest non-zero eigenvalue: {eigvals3[6]:.2e}")
    
    # MITC4 Single Element
    coords4 = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.2],
        [0.0, 1.0, 0.2]
    ])
    K4 = compute_mitc4_plus_stiffness(coords4, E, nu, thickness)
    eigvals4 = np.linalg.eigvalsh(K4)
    zero_eigs4 = np.sum(np.abs(eigvals4) < 1e-4)
    print(f"\n[MITC4+] Total DOFs: {K4.shape[0]}")
    print(f"         Zero eigenvalues (RBMs expected = 6): {zero_eigs4}")
    print(f"         Lowest non-zero eigenvalue: {eigvals4[6]:.2e}")

    if zero_eigs3 == 6 and zero_eigs4 == 6:
        print("\n[Result] SUCCESS: Both elements correctly exhibit exactly 6 Rigid Body Modes.")
    else:
        print("\n[Result] FAILED: Incorrect number of Rigid Body Modes detected.")

if __name__ == "__main__":
    check_eigenvalues()
