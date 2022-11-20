from tqdm import tqdm

from mcml.defs import InputParams, OutputParams, Photon
import mcml.go as go

def do_one_run():
    ...


def main():
    """"""
    inps = InputParams.read_mci("./sample.mci")

    for i, inp in enumerate(inps):
        rsp = go.calc_r_specular(inp.layers)
        out = OutputParams.init(rsp, inp)

        print(f"Running simulation {i} . . .")
        for _ in tqdm(range(inp.num_photons)):
            photon = Photon.init(out.rsp, inp.layers)
            while not photon.dead:
                go.hop_drop_spin(photon, inp, out)

        breakpoint()



if __name__ == "__main__":
    main()

