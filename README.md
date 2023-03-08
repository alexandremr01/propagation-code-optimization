# propagation-code-optimization

# Initialize environment

Create a `.env` file with

```
export USER= 
export GROUP=
```

and initialize it with `source .env`.

## Uploading

`make upload`

## Downloading log file

`make get_log`

## Deployment Examples

```
python3 -m optimizer.main --algorithm hill_climbing --steps 4
python3 -m optimizer.main --algorithm greedy --steps 4
python3 -m optimizer.main --algorithm tabu_greedy --steps 4 --hparams '{"n_tabu":5}'
python3 -m optimizer.main --algorithm simulated_annealing --steps 10 --hparams '{"t0":20}'
python3 -m optimizer.main --algorithm csa --steps 10 --batch
python3 -m optimizer.main --algorithm cmaes --steps 10
```

Flag `--batch`: runs 4 instances. Can either 

- Calculate the cost function in parallel (CSA, CMAES)
- Runs sequential programs in different instances in parallel with different initializations (Hill Climbing, Greedy, Tabu Greedy, Simulated Annealing)

## FAQ

Installation of CMA: Check if the pip used to install is the same as the Python being used to run. Sometimes we need to use `/usr/bin/pip` (TODO: avoid this).
