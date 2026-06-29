#!/bin/bash
set -e

# HumanResources
python src/main.py configs/hr_experiment_k_neighbors_pt.json
python src/main.py configs/hr_experiment_k_neighbors_ft.json
python src/main.py configs/hr_experiment_n_shots_pt.json
python src/main.py configs/hr_experiment_n_shots_ft.json

# WebOfScience
python src/main.py configs/wos_experiment_k_neighbors_pt.json
python src/main.py configs/wos_experiment_k_neighbors_ft.json
python src/main.py configs/wos_experiment_n_shots_pt.json
python src/main.py configs/wos_experiment_n_shots_ft.json
