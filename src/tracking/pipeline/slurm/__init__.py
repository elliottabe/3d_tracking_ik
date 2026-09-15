"""The SLURM chain -- a multi-recording campaign, submittable as a job graph.

`graph.py` builds the dependency graph as DATA (`Job` dataclasses); `render.py`
turns one `Job` into an `sbatch` invocation. Splitting it this way is the whole
point of this package: the graph's SHAPE (barriers in the right place, arrays
sized to the bout count, one collect gating every recording) is then a plain
unit test against `graph.py` alone, verifiable on a login node without
`sbatch` ever running -- see `graph.py`'s module docstring for why that
matters here specifically.
"""
