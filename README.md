# RUMBoost boosted from the parameter space - hEART 2025

This repository contains the code used to produce the results presented at the hEART 2025 conference. All the SwissMetro dataset figures are [here](src/results/SwissMetro/figures/) and all the LPMC dataset figures are [here](src/results/LPMC/figures/).

All RUMBoost models have been run from the [run_models.py](src/run_models.py) file. This include a 5-fold cross validation on the train set to find the optimal number of trees, and a final training on the full train set with optimal number of trees. No further regularisation or hyperparameter search has been used. Alternatively, the models can be run from a terminal with the following commands:

- LPMC:
    - Piece-wise constant
      - Monotonic
        - `python main.py --model "RUMBoost" --model_type "constant" --monotone "true" --dataset "LPMC" --all_boosters "true"`
      - Non-monotonic
        - `python main.py --model "RUMBoost" --model_type "constant" --monotone "false" --dataset "LPMC" --all_boosters "true"`
    - Piece-wise linear
      - Monotonic
        - `python main.py --model "RUMBoost" --model_type "linear" --monotone "true" --dataset "LPMC" --all_boosters "true"`
      - Non-monotonic 
        - `python main.py --model "RUMBoost" --model_type "linear" --monotone "false" --dataset "LPMC" --all_boosters "true"`
- SwissMetro
    - Piece-wise constant
      - Monotonic
        - `python main.py --model "RUMBoost" --model_type "constant" --monotone "true" --dataset "SwissMetro" --all_boosters "true"`
      - Non-monotonic
        - `python main.py --model "RUMBoost" --model_type "constant" --monotone "false" --dataset "SwissMetro" --all_boosters "true"`
    - Piece-wise linear
      - Monotonic
        - `python main.py --model "RUMBoost" --model_type "linear" --monotone "true" --dataset "SwissMetro" --all_boosters "true"`
      - Non-monotonic 
        - `python main.py --model "RUMBoost" --model_type "linear" --monotone "false" --dataset "SwissMetro" --all_boosters "true"`

All the model results on the SwissMetro are [here](src/results/SwissMetro/RUMBoost/) and on the LPMC are [here](src/results/LPMC/RUMBoost/).

The assisted-specified MNL have been run from the [biogeme_models_run.ipynb](src/biogeme_models_run.ipynb), with results [here](src/results/SwissMetro/assisted_specification/).




        