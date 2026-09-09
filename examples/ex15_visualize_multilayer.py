import numpy as np
import matplotlib.pyplot as plt
import koreanize_matplotlib
import pickle
import os
import matplotlib.animation as animation
from dispsolver.mesh3d import Mesh3D

def plot_2d_section_slip(mesh, u, out_png):
    plt.figure(figsize=(10, 6))
    
    for eid, elem in mesh.elements.items():
        nids = elem.node_ids
        face_nids = nids[0:4] # Nodes at Z=0
        
        face_x = []
        face_y = []
        for nid in face_nids:
            node = mesh.nodes[nid]
            idx = mesh.node_id_to_index()[nid]
            ux = u[3*idx+0]
            uy = u[3*idx+1]
            face_x.append(node.x + ux)
            face_y.append(node.y + uy)
        
        face_x.append(face_x[0])
        face_y.append(face_y[0])
        
        color = 'blue' if elem.pid == 0 else 'orange'
        alpha = 0.5
        
        plt.plot(face_x, face_y, color=color, linewidth=0.5)
        plt.fill(face_x, face_y, color=color, alpha=alpha, label='PET (PID 0)' if elem.pid==0 else 'PSA (PID 1)')
        
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='best')
    
    plt.title("4-Layer Thin Bar Folding - Interlayer Slip & Separation (2D Section)")
    plt.xlabel("X (mm)")
    plt.ylabel("Y (mm)")
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()

def create_animation(mesh, history, out_gif):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    def update(frame):
        ax.clear()
        u = history[frame]
        
        for eid, elem in mesh.elements.items():
            nids = elem.node_ids
            face_nids = nids[0:4]
            
            face_x = []
            face_y = []
            for nid in face_nids:
                node = mesh.nodes[nid]
                idx = mesh.node_id_to_index()[nid]
                ux = u[3*idx+0]
                uy = u[3*idx+1]
                face_x.append(node.x + ux)
                face_y.append(node.y + uy)
            
            face_x.append(face_x[0])
            face_y.append(face_y[0])
            
            color = 'blue' if elem.pid == 0 else 'orange'
            ax.fill(face_x, face_y, color=color, alpha=0.5)
            ax.plot(face_x, face_y, color='black', linewidth=0.3)
            
        ax.set_title(f"4-Layer Folding Animation (Step {frame+1}/{len(history)})")
        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Y (mm)")
        ax.axis('equal')
        # Fix axis limits
        ax.set_xlim(-25, 25)
        ax.set_ylim(-15, 15)
        
    ani = animation.FuncAnimation(fig, update, frames=len(history), interval=100)
    ani.save(out_gif, writer='pillow')
    plt.close()

def main():
    res_path = 'examples/ex15_result.pkl'
    if not os.path.exists(res_path):
        print(f"Result file {res_path} not found.")
        return
        
    print(f"Loading {res_path}...")
    with open(res_path, 'rb') as f:
        res = pickle.load(f)
        
    mesh = res['mesh']
    u = res['displacement']
    history = res.get('history', [u])
    
    print("Generating 2D Section Plot...")
    plot_2d_section_slip(mesh, u, 'examples/ex15_2d_section_slip.png')
    print("Saved to examples/ex15_2d_section_slip.png")
    
    if len(history) > 1:
        print(f"Generating Animation with {len(history)} frames...")
        create_animation(mesh, history, 'examples/ex15_folding_animation.gif')
        print("Saved to examples/ex15_folding_animation.gif")
    else:
        print("No history found in result. Cannot generate animation.")

if __name__ == '__main__':
    main()
