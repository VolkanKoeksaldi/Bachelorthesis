from pathlib import Path

# Change for new evaluation run
RUN_ID = "docker_run_01"

# standard evaluation based on wiese et al
EVALUATION_PROFILE = "wiese"

# Size according to Wiese et al. basis table
SOURCE_ROWS = 56341

# 0 = 56341
# 1 = 112682
# 2 = 225364
# 3 = 450728
DUPLICATION_LEVEL = 3

NUM_NODES = 10

REOPTIMIZATION_INSERT_COUNT = 100

copy_factor = 2 ** DUPLICATION_LEVEL
target_rows = SOURCE_ROWS * copy_factor

MESH_SAMPLE_SEED = 42
# Reproducible patient information and ids
MESH_PATIENT_SEED = 42
MESH_PATIENT_COUNT = 10000
IMDB_RANDOM_SEED = 42
REPLICATION_FACTOR = 3
CAPACITY_BUFFER = 0.50

MESH_FRAGMENTATION_SCHEMES = (
    "top_category",
    "branch_code",
    "subbranch_code",
)

IMDB_FRAGMENTATION_SCHEMES = (
    "title_type",
    "decade",
    "primary_genre",
)

source_dir = Path(__file__).resolve().parent
prototype_dir = source_dir.parent

RUN_LABEL = (f"{EVALUATION_PROFILE}_"
             f"base_{SOURCE_ROWS}_"
             f"x{copy_factor}_"
             f"{RUN_ID}")

OUTPUT_ROOT = (prototype_dir / "output" / "evaluation_runs" / RUN_LABEL)

def experiment_path(rel_path):
    """
    Returns a path.
    """
    return OUTPUT_ROOT / Path(rel_path)