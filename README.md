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

Hill Climbing: The deploy phase runs mpirun with 1 process with access to 16 cores.

Successive Descents: The deploy phases run sbatch with 4 nodes, each one with one hill climbing in run phase.

