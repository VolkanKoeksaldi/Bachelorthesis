from pathlib import Path

# Change for new evaluation run
RUN_ID = "docker_run_01"

# Number of rows in the unscaled base table
SOURCE_ROWS = 56341

# 0 -> x1 = 56341 rows
# 1 -> x2 = 112682 rows
# 2 -> x4 = 225364 rows
# 3 -> x8 = 450728 rows
DUPLICATION_LEVEL = 0

# Maximum number of available nodes
NUM_NODES = 10

# Number of synthetic insertions during reoptimization
REOPTIMIZATION_INSERT_COUNT = 100

copy_factor = 2 ** DUPLICATION_LEVEL
target_rows = SOURCE_ROWS * copy_factor

# Parameters used for reproducible dataset generation
MESH_SAMPLE_SEED = 42
MESH_PATIENT_SEED = 42
MESH_PATIENT_COUNT = 10000
IMDB_RANDOM_SEED = 42

# Required replication factor
REPLICATION_FACTOR = 3

# Additional node-capacity buffer
CAPACITY_BUFFER = 0.50

MESH_FRAGMENTATION_SCHEMES = ("top_category", "branch_code", "subbranch_code")

IMDB_FRAGMENTATION_SCHEMES = ("title_type", "decade", "primary_genre")

source_dir = Path(__file__).resolve().parent
prototype_dir = source_dir.parent

RUN_LABEL = (f"base_{SOURCE_ROWS}_"
             f"x{copy_factor}_"
             f"{RUN_ID}")

OUTPUT_ROOT = (prototype_dir / "output" / "evaluation_runs" / RUN_LABEL)

def experiment_path(rel_path):
    """
    Returns an output path relative to the current evaluation run.
    """
    return OUTPUT_ROOT / Path(rel_path)