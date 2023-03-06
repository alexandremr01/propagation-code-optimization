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

## Deployments

```python3 src/optimizer.py --algorithm hill_climbing --steps 4```
```python3 src/optimizer.py --algorithm tabu_greedy --steps 4 --hparams '{"n_tabu":5}'```

Hill Climbing: The deploy phase runs mpirun with 1 process with access to 16 cores.

Flag `-kangaroo`: runs 4 in parallel with different initializations
