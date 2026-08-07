from pathlib import Path

# Change for new evaluation run
RUN_ID = "run_01"
MESH_MAX_RECORD = 1000
IMDB_MAX_TITLES = 1000
REOPTIMIZATION_INSERT_COUNT = 100

RUN_LABEL = (f"mesh_{MESH_MAX_RECORD}_"
             f"imdb_{IMDB_MAX_TITLES}_"
             f"{RUN_ID}")

OUTPUT_ROOT = (Path("prototype/output/evaluation_runs") / RUN_LABEL)

def experiment_path(rel_path):
    """
    Returns a path.
    """
    return OUTPUT_ROOT / Path(rel_path)