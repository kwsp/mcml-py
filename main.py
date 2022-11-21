import time

from tqdm import tqdm
from numba import njit
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


def do_one_run(rsp, inp: InputParams, layers: go.Layers, out: OutputParams):
    for _ in tqdm(range(inp.num_photons)):
        launch_one_photon(rsp, inp, layers, out.a_rz, out.rd_ra, out.tt_ra)

    sum_scale_result(inp, layers, out)


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

    inps = read_mci(args.input_file)

    for i, (inp, layers, out_fname) in enumerate(inps):
        rsp = go.calc_r_specular(layers)
        out = OutputParams.init(rsp, inp)

        print(f"Running simulation {i} . . .")
        _start = time.perf_counter()
        do_one_run(rsp, inp, layers, out)
        elapsed = time.perf_counter() - _start

        write_mco(out_fname, inp, layers, out, elapsed)


if __name__ == "__main__":
    main()
