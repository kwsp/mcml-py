from tqdm import tqdm
from numba import njit
import matplotlib.pyplot as plt

from mcml.defs import InputParams, OutputParams, read_mci
from mcml.photon import make_Photon
import mcml.go as go
from mcml.proc import sum_scale_result


@njit
def do_one_run(rsp, inp: InputParams, layers: go.Layers, a_rz, rd_ra, tt_ra):
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


def main():
    inps = read_mci("./sample.mci")

    for i, (inp, layers, out_fname) in enumerate(inps):
        rsp = go.calc_r_specular(layers)
        out = OutputParams.init(rsp, inp)

        print(f"Running simulation {i} . . .")
        for _ in tqdm(range(inp.num_photons)):
            do_one_run(rsp, inp, layers, out.a_rz, out.rd_ra, out.tt_ra)

        breakpoint()
        sum_scale_result(inp, layers, out)
        breakpoint()


if __name__ == "__main__":
    main()
# def print_photon(photon):
# print(f"Photon(x={photon.x}, y={photon.y}, z={photon.z}, ux={photon.ux}, uy={photon.uy}, uz={photon.uz}, w={photon.w}, dead={photon.dead}, layer={photon.layer}, s={photon.s}, sleft={photon.sleft}")
