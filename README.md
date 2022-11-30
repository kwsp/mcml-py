# MCML Python

Port of [MCML (Monte Carlo Multi-Layered)](https://omlc.org/software/mc/mcml/index.html) to Python.

**Requires Python 3.10, `numba`, and `tqdm`.** As of Nov 2022, `numba` doesn't yet support Python 3.11.

## Getting started

1. Clone the repo (or download the zip file and extract).

2. Create a virtual environment. Use [Miniconda](https://docs.conda.io/en/latest/miniconda.html) if you don't know what that means. After installing miniconda, open "Anaconda Powershell Prompt" and run

```
conda create -n mcml python=3.10 numba tqdm
```

This creates a new virtual environment named `mcml`, and installs Python 3.10 along with `numba` and `tqdm`.

4. Activate the `mcml` virtual environment:

```
conda activate mcml
```

5. `cd` into the working directory. This is where you cloned this git repo, or where you unzipped the downloaded zip file.

```
cd [path to mcml-py]
```

6. Run the MCML main program with a sample input file.

On Windows:

```
python main.py sample.mci
```

On Unix

```
python3 main.py sample.mci
```

## Accuracy

The program is checked against the original MCML program. Errors should be <1%.

## Speed

Most of this program is JIT compiled with [numba](https://numba.pydata.org/), hence the speed is reasonable. Simulating 1,000,000 photons in a 3 layer medium takes <20s on my laptop.
