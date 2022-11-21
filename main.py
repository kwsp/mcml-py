from tqdm import tqdm
# from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED

from mcml.defs import InputParams, OutputParams, read_mci
from mcml.photon import make_Photon
import mcml.go as go
import numpy as np
from numba import njit
import matplotlib.pyplot as plt


# @njit
def do_one_run(rsp, inp: InputParams, layers: go.Layers, a_rz, rd_ra, tt_ra):
    photon = make_Photon(rsp, layers)
    # _photon = photon.to_struct()
    breakpoint()

    while not photon.dead:
        go.hop_drop_spin(
            photon,
            inp,
            layers=layers,
            a_rz=a_rz,
            rd_ra=rd_ra,
            tt_ra=tt_ra,
        )
        breakpoint()


def main():
    """"""
    inps = read_mci("./sample.mci")
    

    for i, (inp, layers, out_fname) in enumerate(inps):
        rsp = go.calc_r_specular(layers)
        out = OutputParams.init(rsp, inp)

        print(f"Running simulation {i} . . .")

        for _ in tqdm(range(inp.num_photons)):
            # do_one_run(inp, layers, out)
            do_one_run(rsp, inp, layers, out.a_rz, out.rd_ra, out.tt_ra)

        breakpoint()


if __name__ == "__main__":
    main()
