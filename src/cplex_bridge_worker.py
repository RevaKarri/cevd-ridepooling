"""Solves an LP/MIP file with the real CPLEX engine.

Runs under the Rosetta (x86_64) Python 3.7 venv, since CPLEX 20.1's Python
bindings only support that combination on this machine. Invoked as a
subprocess from cplex_bridge.py, which runs in the main (arm64) venv
alongside TensorFlow -- the two can't share a process here because Rosetta
doesn't support the AVX instructions TensorFlow requires.
"""
import sys
import json

import cplex


def main(lp_path, out_path):
    cpx = cplex.Cplex(lp_path)
    cpx.set_log_stream(None)
    cpx.set_error_stream(None)
    cpx.set_warning_stream(None)
    cpx.set_results_stream(None)
    cpx.solve()

    result = {
        "status_string": cpx.solution.get_status_string(),
        "objective": cpx.solution.get_objective_value(),
        "values": dict(zip(cpx.variables.get_names(), cpx.solution.get_values())),
    }
    with open(out_path, 'w') as f:
        json.dump(result, f)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
