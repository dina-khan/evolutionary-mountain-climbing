# Evolutionary AI Mountain Climbing Morphologies - Genetic Algorithm with Physics Simulation

> An artificial intelligence research project reimplementing and extending 
> Karl Sims' groundbreaking 1994 work on evolving virtual creatures, using a 
> genetic algorithm to evolve procedurally generated, physically simulated 
> creatures (PyBullet rigid-body morphologies) that learn to climb a 
> Gaussian-mesh mountain through a custom multi-term fitness function, 
> compared across 6 structured genetic algorithm and encoding scheme 
> experiments.

**Tech Stack:** Python · PyBullet · NumPy · Pandas · Matplotlib

🎬 **[Experiment Results Demo: Evolved Morphologies Walkthrough](https://drive.google.com/file/d/1wGN32-j-egyGldijdJC62wX9IqttgUVu/view?usp=sharing)** 

 📄 **[View Full Report](docs/report.pdf)**

---

## How It Works

This project applies **evolutionary computation**, a core artificial 
intelligence paradigm, to the open problem Karl Sims first explored: can 
artificial creatures with no hand-designed morphology or motor control learn 
complex physical behaviors purely through simulated natural selection? 
Creatures are encoded as genomes (arrays of genes mapping to link shape, 
length, radius, mass, joint configuration, and motor waveform/amplitude/
frequency). Each generation:

1. Every creature's genome is decoded into a URDF robot body and simulated 
   in a PyBullet arena for 2400 steps
2. A custom **fitness function** rewards/penalizes 15+ behaviors - radius 
   reduction towards the mountain, mountain contact, platform height gained, 
   ascent while climbing, flying, wall proximity, useless motor effort, 
   excessive limb growth, and inefficient (long) paths
3. The fittest creature is preserved as an elite; the rest of the next 
   generation is produced via crossover and point/grow/shrink mutation
4. Metrics are logged per generation and aggregated across 3 runs per 
   experiment

## Fitness Function

**Rewarded:** closest/decreasing radius to mountain centre, mountain 
collisions, highest platform reached, lifting, mountain climbing while in 
contact

**Penalized:** increasing radius from mountain, long inefficient paths, 
useless motor effort (wobbling without movement), flying, wall proximity 
(running and at simulation end), excessive limb growth

## Experiments

Six experiments compare genetic algorithm and encoding scheme variations 
against base settings (population 30, gene count 3):

| Experiment | Change |
|---|---|
| Increased population | Population size 30 → 50 |
| Increased gene count | Gene count 3 → 5 |
| Increased mutation | Higher mutation rate and range |
| Stronger motors | Higher motor amplitude and frequency scaling |
| Evolve-only-motors | Mutation restricted to motor-related genes only (no morphology change) |

Each experiment runs 3 times; results are aggregated (mean ± standard 
deviation) across generations and visualised in 4 graph categories: fitness, 
horizontal movement, mountain climbing, and stability.

## Key Findings

- Base settings learned mountain approach and sustained contact, but never 
  unlocked ascent
- Increased population improved reliability and convergence speed without 
  unlocking ascent
- Increased gene count and mutation expanded exploration but reduced 
  refinement after convergence
- Stronger motors caused unreliable motor control rather than faster climbing
- Restricting evolution to motor genes only reduced mountain contact due to 
  inflexible morphology
- No single setting unlocked mountain ascent - a combination of approaches 
  is likely needed

---

## Folder Structure

| File / Folder | Purpose |
|---|---|
| `creature.py` | Defines the `Creature` and `Motor` classes - decodes a genome into URDF links and motors |
| `genome.py` | Defines the `Genome` class - gene specification, mutation, crossover, genome-to-link decoding |
| `population.py` | Defines the `Population` class - creates and manages a population of creatures, fitness-proportionate parent selection |
| `ga_runner_cw_envt.py` | Main entry point - defines the `Simulation` class (runs a creature in the PyBullet arena and computes its fitness and metrics) and runs the genetic algorithm experiment(s), logging metrics to CSV |
| `prepare_shapes.py` | Generates the mountain `.obj`/`.urdf` files (called automatically by `ga_runner_cw_envt.py` - no need to run separately) |
| `realtime_from_csv_arena.py` | Loads a saved elite creature's genome CSV and visualises it climbing in a live PyBullet GUI window |
| `graph_experiment_results.py` | Aggregates and plots metrics (fitness, movement, climbing, stability) across an experiment's 3 runs |
| `table_mountain_radius.py` | Aggregates elite minimum mountain radius reached (and improvement over generations) across **all** experiments into `summary.xlsx` |
| `shapes/` | Generated mountain mesh files (`.obj`, `.urdf`) |
| `experiment_results/` | Original experiment data referenced in the report (`base`, `gene_5`, `mutation_high`, `only_motors_evolution`, `population_50`, `strong_motors`), each with 3 runs and per-generation metrics/genome CSVs |
| `result_logs/` | Created automatically when `ga_runner_cw_envt.py` is run - contains newly generated experiment data, kept separate from `experiment_results/` |
| `docs/report.pdf` | Full written report with methodology, all 6 experiments, and conclusions |

---



## Setup

Requires Python 3, PyBullet, NumPy, Pandas, Matplotlib, openpyxl, and the 
`noise` package.

```bash
pip install pybullet numpy pandas matplotlib openpyxl noise
```

---

## Exploring the Original Experiment Results

The `experiment_results/` folder already contains all data referenced in 
the report - no need to run anything to explore it.

**View the aggregated graphs for an experiment** (4 graph windows will pop 
up - fitness, horizontal movement, mountain climbing, and stability):

```bash
python graph_experiment_results.py experiment_results/base
```

Replace `base` with `gene_5`, `mutation_high`, `only_motors_evolution`, 
`population_50`, or `strong_motors` to view a different experiment.

**View summary statistics across all experiments** - produces a 
`summary.xlsx` table of each experiment's final-generation elite minimum 
mountain radius and improvement in that radius over generations (a key 
indicator of mountain-climbing progress):

```bash
python table_mountain_radius.py
```

> By default this script looks for experiment folders inside `result_logs/`. 
> To run it against the original results, either rename 
> `experiment_results/` to `result_logs/`, or edit the folder path at the 
> bottom of the script to point to `experiment_results`.

**Watch a saved elite creature climb in real time** (opens a PyBullet GUI 
window):

```bash
python realtime_from_csv_arena.py experiment_results/base/run1/generation99_best.csv
```

Swap in any other experiment/run/generation CSV to view a different creature.

---

## Generating New Results

Running the genetic algorithm creates a **new** `result_logs/` folder, 
entirely separate from `experiment_results/` - nothing gets overwritten.

```bash
python ga_runner_cw_envt.py
```

By default this runs the `base` experiment (population 30, gene count 3) 
for 100 generations, 3 times, logging to `result_logs/base/run1/`, `run2/`, 
`run3/`.

**Watch your newly evolved creature climb in real time:**
```bash
python realtime_from_csv_arena.py result_logs/base/run1/generation99_best.csv
```

**Graph your new results:**
```bash
python graph_experiment_results.py result_logs/base
```

**Generate summary statistics for your new results** (`summary.xlsx`):
```bash
python table_mountain_radius.py
```

**Run a different experiment setting** by editing the parameters passed to 
`run_genetic_algorithm_experiment()` at the bottom of `ga_runner_cw_envt.py` 
(e.g. `pop_size=50` for increased population, `gene_count=5` for increased 
gene count).
