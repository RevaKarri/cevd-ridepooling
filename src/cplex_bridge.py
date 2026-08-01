"""Solves a docplex Model using the real CPLEX 20.1 engine via a subprocess.

CPLEX 20.1's Python bindings only work under Python 3.7/3.8 x86_64, while
TensorFlow needs to run natively on arm64 (Rosetta doesn't support the AVX
instructions TF requires). So the two can't live in one process on this
machine: the model is exported to an LP file, handed to a solver subprocess
running in the .venv-cplex (Rosetta) environment, and the resulting variable
values are read back.
"""
import os
import json
import subprocess
import tempfile

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_BRIDGE_PYTHON = os.path.join(_REPO_ROOT, '.venv-cplex', 'bin', 'python3.7')
_WORKER_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cplex_bridge_worker.py')


class BridgeSolution:
    def __init__(self, values, objective_value, status_string):
        self._values = values
        self.objective_value = objective_value
        self.status_string = status_string

    def get_value(self, var):
        name = var.get_name() if hasattr(var, 'get_name') else var
        return self._values.get(name, 0.0)

    def __bool__(self):
        return True


def solve(model):
    with tempfile.NamedTemporaryFile(suffix='.lp', delete=False) as lp_file:
        lp_path = lp_file.name
    out_path = lp_path + '.json'

    try:
        model.export_as_lp(lp_path)
        subprocess.run(
            [_BRIDGE_PYTHON, _WORKER_SCRIPT, lp_path, out_path],
            check=True, capture_output=True, text=True,
        )
        with open(out_path) as f:
            result = json.load(f)
    finally:
        for path in (lp_path, out_path):
            if os.path.exists(path):
                os.remove(path)

    if 'optimal' not in result['status_string'].lower():
        return None

    return BridgeSolution(result['values'], result['objective'], result['status_string'])
