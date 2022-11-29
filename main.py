import concurrent.futures
import os
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


def _work(n: int, rsp: float, inp: InputParams, layers: go.Layers, bar=False):
    """
    Launch n photons.
    """
    # initialize output (result) buffers
    out = OutputParams.init(rsp, inp)
    layers = List(layers)  # Convert to numba.typed.List

    if bar:
        for _ in tqdm(range(n)):
            launch_one_photon(rsp, inp, layers, out.a_rz, out.rd_ra, out.tt_ra)
    else:
        for _ in range(n):
            launch_one_photon(rsp, inp, layers, out.a_rz, out.rd_ra, out.tt_ra)
    return out


def do_one_simulation_parallel(
    rsp: float, inp: InputParams, layers: go.Layers, n_workers: int
):
    """
    Run one simulation by launching `num_photons` as defined by the input.
    """

    # No. photons to simulate per thread
    n_photons = [inp.num_photons // n_workers] * (n_workers - 1)
    this_n_photons = n_photons[0] + inp.num_photons % n_workers

    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers - 1) as executor:
        futures = []
        for n_photon in n_photons:
            f = executor.submit(_work, n_photon, rsp, inp, layers)
            futures.append(f)

        out = _work(this_n_photons, rsp, inp, layers, bar=True)
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

    n_workers = os.cpu_count() or 4

    for i, (inp, layers, out_fname) in enumerate(inps):
        # calculate specular reflectance
        rsp = go.calc_r_specular(layers)

        # run the current simulation as define by the input file
        print(f"Simulation {i + 1} started with {n_workers} workers...")
        _start = time.perf_counter()

        # out = do_one_simulation(rsp, inp, layers)
        out = do_one_simulation_parallel(rsp, inp, layers, n_workers)

        elapsed = time.perf_counter() - _start
        print(f"Simulation {i + 1} finished in {elapsed:.4g} sec.")

        # write results to the MCO file named in the input MCI
        write_mco(out_fname, inp, layers, out, elapsed)


if __name__ == "__main__":
    main()
