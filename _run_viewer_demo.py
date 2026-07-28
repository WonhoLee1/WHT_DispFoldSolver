import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "examples"))
from dispsolver.postprocess.viewer import launch_from_result
launch_from_result("examples/ex13_build_result.pkl")
