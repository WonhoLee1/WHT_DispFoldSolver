import numpy as np
from numba import njit
from .shell_kinematics import calc_drilling_penalty_stiffness

@njit(fastmath=True)
def get_local_frame_tri(coords):
    center = (coords[0] + coords[1] + coords[2]) / 3.0
    v1 = coords[1] - coords[0]
    v2 = coords[2] - coords[0]
    e1 = v1 / np.linalg.norm(v1)
    e3_temp = np.cross(e1, v2)
    e3 = e3_temp / np.linalg.norm(e3_temp)
    e2 = np.cross(e3, e1)
    
    T_node = np.zeros((3, 3))
    T_node[0, :] = e1
    T_node[1, :] = e2
    T_node[2, :] = e3
    
    local_coords = np.zeros((3, 2))
    for i in range(3):
        vec = coords[i] - center
        local_coords[i, 0] = np.dot(vec, e1)
        local_coords[i, 1] = np.dot(vec, e2)
        
    T_sys = np.zeros((18, 18))
    for i in range(6):
        T_sys[i*3:i*3+3, i*3:i*3+3] = T_node
        
    return local_coords, T_sys

@njit(fastmath=True)
def compute_mitc3_plus_stiffness(coords, E, nu, thickness):
    """
    Computes MITC3 local stiffness and transforms to global.
    """
    local_coords, T_sys = get_local_frame_tri(coords)
    K_loc = np.zeros((18, 18))
    
    # Material
    D_m = E * thickness / (1 - nu**2) * np.array([
        [1, nu, 0],
        [nu, 1, 0],
        [0, 0, (1 - nu)/2]
    ])
    D_b = (E * thickness**3 / 12.0) / (1 - nu**2) * np.array([
        [1, nu, 0],
        [nu, 1, 0],
        [0, 0, (1 - nu)/2]
    ])
    kappa = 5.0 / 6.0
    G = E / (2 * (1 + nu))
    D_s = kappa * G * thickness * np.array([
        [1, 0],
        [0, 1]
    ])
    
    # Linear shape functions for Triangle (Constant Strain/Curvature)
    x1, y1 = local_coords[0, 0], local_coords[0, 1]
    x2, y2 = local_coords[1, 0], local_coords[1, 1]
    x3, y3 = local_coords[2, 0], local_coords[2, 1]
    
    area = 0.5 * abs((x2-x1)*(y3-y1) - (x3-x1)*(y2-y1))
    
    b1 = y2 - y3
    b2 = y3 - y1
    b3 = y1 - y2
    c1 = x3 - x2
    c2 = x1 - x3
    c3 = x2 - x1
    
    dN_dx = np.array([b1, b2, b3]) / (2*area)
    dN_dy = np.array([c1, c2, c3]) / (2*area)
    
    Bm = np.zeros((3, 18))
    Bb = np.zeros((3, 18))
    Bs = np.zeros((2, 18))
    
    N_pts = np.array([
        [0.5, 0.5, 0.0],
        [0.0, 0.5, 0.5],
        [0.5, 0.0, 0.5]
    ])
    
    for i in range(3):
        idx = i * 6
        Bm[0, idx] = dN_dx[i]
        Bm[1, idx+1] = dN_dy[i]
        Bm[2, idx] = dN_dy[i]
        Bm[2, idx+1] = dN_dx[i]
        
        Bb[0, idx+4] = dN_dx[i]
        Bb[1, idx+3] = -dN_dy[i]
        Bb[2, idx+4] = dN_dy[i]
        Bb[2, idx+3] = -dN_dx[i]
        
    K_loc += area * (Bm.T @ D_m @ Bm)
    K_loc += area * (Bb.T @ D_b @ Bb)
    
    for ip in range(3):
        L = N_pts[ip]
        Bs = np.zeros((2, 18))
        for i in range(3):
            idx = i * 6
            Bs[0, idx+2] = dN_dx[i]
            Bs[1, idx+2] = dN_dy[i]
            Bs[0, idx+4] = L[i]
            Bs[1, idx+3] = -L[i]
        K_loc += (area / 3.0) * (Bs.T @ D_s @ Bs)
    
    K_drill = calc_drilling_penalty_stiffness(area, E, nu, thickness, penalty_factor=1e-4)
    for i in range(3):
        K_loc[i*6+5, i*6+5] += K_drill
        
    return T_sys.T @ K_loc @ T_sys

@njit(fastmath=True)
def assemble_mitc3_plus(nodes, elements, E, nu, thickness):
    n_dofs = nodes.shape[0] * 6
    K_global = np.zeros((n_dofs, n_dofs))
    
    for i in range(elements.shape[0]):
        n1, n2, n3 = elements[i]
        coords = np.zeros((3, 3))
        coords[0] = nodes[n1]
        coords[1] = nodes[n2]
        coords[2] = nodes[n3]
        Ke = compute_mitc3_plus_stiffness(coords, E, nu, thickness)
        
        idx = np.array([
            n1*6, n1*6+1, n1*6+2, n1*6+3, n1*6+4, n1*6+5,
            n2*6, n2*6+1, n2*6+2, n2*6+3, n2*6+4, n2*6+5,
            n3*6, n3*6+1, n3*6+2, n3*6+3, n3*6+4, n3*6+5
        ])
        
        for r in range(18):
            for c in range(18):
                K_global[idx[r], idx[c]] += Ke[r, c]
                
    return K_global
