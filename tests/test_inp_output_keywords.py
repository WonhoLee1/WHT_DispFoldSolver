import pytest
from dispsolver.io.abaqus_reader import _parse_source
from dispsolver.output.variables import FieldPosition

SAMPLE_INP_WITH_OUTPUT = """*HEADING
Model with Output Keywords
*NODE
1, 0.0, 0.0, 0.0
2, 1.0, 0.0, 0.0
3, 1.0, 1.0, 0.0
4, 0.0, 1.0, 0.0
5, 0.0, 0.0, 1.0
6, 1.0, 0.0, 1.0
7, 1.0, 1.0, 1.0
8, 0.0, 1.0, 1.0
*ELEMENT, TYPE=C3D8, ELSET=HEX_ELEM
1, 1, 2, 3, 4, 5, 6, 7, 8
*SOLID SECTION, ELSET=HEX_ELEM, MATERIAL=STEEL
*MATERIAL, NAME=STEEL
*ELASTIC
210000.0, 0.3
*STEP, NAME=Step-1
*STATIC
0.1, 1.0
*BOUNDARY
1, 1, 3, 0.0
*OUTPUT, FIELD, FREQUENCY=2
*NODE OUTPUT
U, RF
*ELEMENT OUTPUT, POSITION=INTEGRATION POINTS
S, E
*OUTPUT, HISTORY, FREQUENCY=1
*NODE OUTPUT
U1, RF1
*END STEP
"""

def test_inp_output_parsing(tmp_path):
    inp_file = tmp_path / "test_out.inp"
    inp_file.write_text(SAMPLE_INP_WITH_OUTPUT, encoding="utf-8")
    
    reader, model = _parse_source(str(inp_file), dim=3)
    assert "Step-1" in model.steps
    step = model.steps["Step-1"]
    
    assert len(step.field_output_requests) >= 1
    field_reqs = list(step.field_output_requests.values())
    all_vars = set()
    for req in field_reqs:
        all_vars.update(req.variables)
    assert "U" in all_vars
    assert "RF" in all_vars
    assert "S" in all_vars
    assert "E" in all_vars
    
    assert len(step.history_output_requests) >= 1
    hist_reqs = list(step.history_output_requests.values())
    all_hist_vars = set()
    for req in hist_reqs:
        all_hist_vars.update(req.variables)
    assert "U1" in all_hist_vars
    assert "RF1" in all_hist_vars
