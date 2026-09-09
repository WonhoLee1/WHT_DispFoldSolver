"""
launch_post_viewer.py
=====================
Launch the interactive PySide6 + Matplotlib 2D FEA Postprocessing Viewer
for any saved folding simulation result (.pkl).

Usage:
    python examples/launch_post_viewer.py [result_path]

Default result_path:
    examples/ex13_build_result.pkl (or examples/ex12_result.pkl)
"""
import sys
import os

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from dispsolver.postprocess.viewer import launch_from_result

if __name__ == '__main__':
    default_pkl = os.path.join(repo_root, 'examples', 'ex13_build_result.pkl')
    if not os.path.exists(default_pkl):
        default_pkl = os.path.join(repo_root, 'examples', 'ex12_result.pkl')

    target_path = sys.argv[1] if len(sys.argv) > 1 else default_pkl
    print(f'Launching Post Viewer for: {target_path}')
    if not os.path.exists(target_path):
        print(f'Error: result file not found at {target_path}')
        sys.exit(1)

    sys.exit(launch_from_result(target_path))
