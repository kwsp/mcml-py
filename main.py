import multiprocessing
import concurrent.futures
import os
import sys
import time

from tqdm import tqdm
from numba import njit
from numba.typed import List
import numpy as np

from mcml.defs import InputParams, OutputParams, read_mci, write_mco
from mcml.photon import make_Photon
import mcml.go as go
from mcml.proc import sum_scale_result


@njit
def launch_one_photon(
    rsp: float,  # specular reflection
    inp: InputParams,  # input params, read only
    layers: go.Layers,  # layers, read only
    a_rz: np.ndarray,  # absorption output
    rd_ra: np.ndarray,  # diffuse reflectance output
    tt_ra: np.ndarray,  # transmittance output
):
    """
    Run simulation for one photon.
    Writes results to a_rz, rd_ra, tt_ra
    """
    photon = make_Photon(rsp, layers)
    while not photon.dead:
        go.hop_drop_spin(
            photon,
            inp,
            layers=layers,
            a_rz=a_rz,
            rd_ra=rd_ra,
            tt_ra=tt_ra,
        )


def _work(n: int, rsp: float, inp: InputParams, layers: go.Layers, bar_position: int, lock):
    """
    Launch n photons.
    """
    # initialize output (result) buffers
    out = OutputParams.init(rsp, inp)
    layers = List(layers)  # Convert to numba.typed.List

    with tqdm(desc=f"Worker {bar_position}", total=n, position=bar_position, file=sys.stdout, leave=False) as _bar:
        for i in range(n):
            launch_one_photon(rsp, inp, layers, out.a_rz, out.rd_ra, out.tt_ra)
            if (i+1) % 100 == 0:
                with lock:
                    _bar.update(100)
    with lock:
        sys.stdout.flush()

    return out


def do_one_simulation_parallel(
    rsp: float, inp: InputParams, layers: go.Layers, n_workers: int, executor, lock
):
    """
    Run one simulation by launching `num_photons` as defined by the input.
    """
    # No. photons to simulate per thread
    n_photons = inp.num_photons // n_workers
    this_n_photons = n_photons + inp.num_photons % n_workers

    futures = [
        executor.submit(_work, n_photons, rsp, inp, layers, i, lock)
        for i in range(1, n_workers)
    ]
    out = _work(this_n_photons, rsp, inp, layers, 0, lock)
    for f in futures:
        out += f.result()

    # launch_one_photon saves unscaled results to the 2D buffers.
    # calculate the 1D results and scale all results properly
    out = sum_scale_result(inp, layers, out)
    return out


def do_one_simulation(rsp: float, inp: InputParams, layers: go.Layers):
    """
    Run one simulation by launching `num_photons` as defined by the input.
    Singled-threaded implementation
    """
    # initialize output (result) buffers
    out = OutputParams.init(rsp, inp)
    layers = List(layers)
    for _ in tqdm(range(inp.num_photons)):
        launch_one_photon(rsp, inp, layers, out.a_rz, out.rd_ra, out.tt_ra)

    # launch_one_photon saves unscaled results to the 2D buffers.
    # calculate the 1D results and scale all results properly
    out = sum_scale_result(inp, layers, out)
    return out


def cli():
    """
    Command line interface
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="MCML-py. Python implementation of MCML (Monte-Carlo Multi-Layered."
    )
    parser.add_argument("input_file", help="MCI input file. See sample.mci")
    return parser.parse_args()


def main():
    args = cli()

    # parse the input MCI file
    inps = read_mci(args.input_file)

    # Use max 4 workers, since the overhead of pickling results is high
    n_workers = min(os.cpu_count() or 4, 4)

    print()
    lock = multiprocessing.Manager().Lock()
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers - 1) as executor:
        for i, (inp, layers, out_fname) in enumerate(inps):
            # calculate specular reflectance
            rsp = go.calc_r_specular(layers)

            # run the current simulation as define by the input file
            with lock:
                print(f"Simulation {i+1}/{len(inps)} started with {n_workers} workers...", flush=True)
            _start = time.perf_counter()

            # out = do_one_simulation(rsp, inp, layers)
            out = do_one_simulation_parallel(rsp, inp, layers, n_workers, executor, lock)

            elapsed = time.perf_counter() - _start
            with lock:
                print(f"\rSimulation {i + 1}/{len(inps)} finished in {elapsed:.4g} sec.\n", flush=True)

            # write results to the MCO file named in the input MCI
            write_mco(out_fname, inp, layers, out, elapsed)


if __name__ == "__main__":
    main()
