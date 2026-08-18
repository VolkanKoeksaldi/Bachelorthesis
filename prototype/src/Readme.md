# Overlap-Aware Data Distribution Using Bin Packing in Distributed Database Systems

This repository contains the Python prototype developed for the bachelor thesis *Overlap-Aware Data Distribution Using Bin Packing in Distributed Database Systems*.

It implements and evaluates three fragment-placement methods:
* a deterministic and simple Round-Robin baseline
* a tuple-based Integer Linear Programming (ILP) formulation
* a conflict-locality-based ILP formulation

The prototype preprocesses MeSH and IMDb data, creates three horizontal fragmentations for each dataset, computes their fragment overlaps, assigns fragments to simulated database nodes according to the placement method, and lastly materializes the used nodes as separate SQLite database files. It also generates and executes a reproducible workload and evaluates storage, replication, locality, reoptimization, and recovery metrics.

## Scope
The SQLite database simulates database nodes locally within a Docker container. The prototype evaluates placement behavior, but does not simulate network latency, distributed transactions, or distributed database servers.

The copy factors x1, x2, x4, and x8 increase the number of items and fragment weights, while preserving the fragment categories, overlap-pair structure, and membership-pattern structure.
Therefore, these runs evaluate the scalability of the placement methods with increasing dataset sizes.
The evaluations assumed `r=m=3`.

## Main configuration
The central experiment parameters are defined in **src/experiment_config.py**.

Each run is written below as:
output/evaluation_runs/base_<SOURCE_ROWS>_x<COPY_FACTOR>_<RUN_ID>/

By using a separate RUN_ID, container name, and Docker output volume, results from several runs can be retained.

## Required Source Data
Place the following source files at the following paths before starting the prototype:
`data/raw/mesh/desc2026.xml`

`data/raw/imdb/title.basics.tsv`

`data/raw/imdb/title.ratings.tsv`

These source datasets can be downloaded from:
- [Medical Subject Headings (MeSH)](https://healthdata.gov/NIH/Medical-Subject-Headings-MeSH-/rc3i-uvpj/about_data)
- [IMDb non-commercial datasets](https://developer.imdb.com/non-commercial-datasets/)

The IMDb downloads are provided as compressed `.tsv.gz` files. Therefore, these need to be decompressed to `.tsv` files beforehand.

## Docker Execution on Windows PowerShell
Open PowerShell in the `prototype` directory and build the image:
```powershell
docker build -t bachelor-prototype .
```

Then create an output volume. The following example is for an x1 run:

```powershell
docker volume create bachelor-x1-output
```

Afterwards the container can be started. For different copy-factor runs, the x1 identifiers need to be changed accordingly:
```powershell
docker run --name bachelor-x1 -it `
  --mount "type=bind,source=$($PWD.Path)\src,target=/app/src" `
  --mount "type=bind,source=$($PWD.Path)\data,target=/app/data,readonly" `
  --mount "type=volume,source=bachelor-x1-output,target=/app/output" `
  bachelor-prototype
```

Scripts can be run inside the container with:
```bash
python src/<script_name>.py
```

The scripts use manual selectors near the beginning of the respective files.
Before execution, the selectors `DATASET`, `PLACEMENT`, and `MODE` used by the respective script need to be configured as required.

## Script settings
| Script | Selectors that may need to be changed |
|---|---|
| `4_compute_mesh_overlaps.py` | `MODE = "baseline"` or `"updates"` |
| `7_compute_imdb_overlaps.py` | `MODE = "baseline"` or `"updates"` |
| `8_assign_fragments_baseline_round_robin.py` | `DATASET` |
| `9_bin_packing_ILP_tuple_based.py` | `DATASET`, `MODE` |
| `10_conflict_locality_ILP.py` | `DATASET`, `MODE` |
| `11_create_sqlite_nodes.py` | `DATASET`, `PLACEMENT`, `MODE` |
| `12_evaluate.py` | `DATASET`, `PLACEMENT` |
| `13_generate_workload.py` | `DATASET` |
| `14_execute_workload.py` | `DATASET`, `PLACEMENT` |
| `15_compute_workload_metrics.py` | `DATASET` |
| `16_compute_workload_affinities.py` | `DATASET` |
| `17_evaluate_affinity_locality.py` | `DATASET` |
| `18_reoptimization.py` | `DATASET`, `MODE = "prepare"` or `"evaluate"` |
| `19_recovery.py` | `DATASET`, `MODE = "baseline"` or `"updates"` |


## Execution Order
The following pipeline shows the execution order for every desired copy factor.

### 1. Dataset Preparation and Fragmentation
For MeSH, run the following code. However be sure to set `MODE = baseline` in script 4, before running it:
```bash
python src/1_create_mesh_sample.py
python src/2_parse_mesh_terms.py
python src/3_create_mesh_fragments.py
python src/4_compute_mesh_overlaps.py
```

For IMDb, before running script 7 set the `MODE = baseline` as well. Afterwards, run:
```bash
python src/5_prepare_imdb_titles.py
python src/6_create_imdb_fragments.py
python src/7_compute_imdb_overlaps.py
```

### 2. Round-Robin Assignment, Workload, and Affinities
For both datasets:

1. Run `8_assign_fragments_baseline_round_robin.py` once.
2. Run `13_generate_workload.py` once.
3. Run `16_compute_workload_affinities.py` once.

It is important to generate the workload and compute its affinities before the conflict-locality ILP because the formulation uses the workload-derived affinities in its objective function.

### 3. Initial ILP Placements
For both datasets, set `MODE = "baseline"` and run:

```bash
python src/9_bin_packing_ILP_tuple_based.py
python src/10_conflict_locality_ILP.py
```

### 4. Initial SQLite Nodes and Placement Evaluation
For each dataset and each placement (`round_robin`, `tuple_ilp`, and
`conflict_locality_ilp`):

1. Run `11_create_sqlite_nodes.py` with `MODE = "baseline"`.
2. Run `12_evaluate.py`.
3. Run `14_execute_workload.py`.

After all three workload-result files exist for one dataset, run:

```bash
python src/15_compute_workload_metrics.py
```

### 5. Insertions and Reoptimization
For each dataset:

1. Run `18_reoptimization.py` with `MODE = "prepare"`.
2. Recompute overlaps in update mode:
   - MeSH: run script 4 with `MODE = "updates"`;
   - IMDb: run script 7 with `MODE = "updates"`.
3. Run scripts 9 and 10 with `MODE = "updates"`.
4. Run script 11 with `MODE = "updates"` for all three placements.
5. Run script 18 with `MODE = "evaluate"`.
6. Run script 17 to evaluate locality for the initial and reoptimized assignments.

### 6. Recovery
For each dataset, run `19_recovery.py` twice:

1. once with `MODE = "baseline"`; and
2. once with `MODE = "updates"`.

The recovery simulation treats `node_1` as unavailable as a recovery source. The file is retained only to identify the items affected by the simulated failure.

## Copying the Results from the Container
To reduce the storage required when copying the output to the host, the following command can optionally be executed to delete the generated `*.db` files:

```powershell
docker exec bachelor-x1 sh -c "find /app/output -type f -name '*.db' -delete"
```

After the pipeline has finished, exit the container:

```bash
exit
```

Then you can copy the output to the host. The following example is for x1:

```powershell
docker cp bachelor-x1:/app/output/. .\docker_results\x1\
```

The old container can then be removed:
```powershell
docker rm bachelor-x1
```

After that the output volume may also be removed:
```powershell
docker volume rm bachelor-x1-output
```