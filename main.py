from tqdm import tqdm
# from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED

from mcml.defs import InputParams, OutputParams, Photon
import mcml.go as go
import numpy as np


def do_one_run(inp: InputParams, layers: go.Layers, out: OutputParams):
    photon = Photon.init_(out.rsp, layers)
    # _photon = photon.to_struct()

    while not photon.dead:
        go.hop_drop_spin(
            photon,
            inp,
            layers=layers,
            a_rz=out.a_rz,
            rd_ra=out.rd_ra,
            tt_ra=out.tt_ra,
        )


def main():
    """"""
    inps = InputParams.read_mci("./sample.mci")

    for i, inp in enumerate(inps):
        _inp = inp.to_struct()
        _layers = np.hstack([l.to_struct() for l in inp.layers])
        breakpoint()

        rsp = go.calc_r_specular(_layers)
        out = OutputParams.init(rsp, inp)

        print(f"Running simulation {i} . . .")

        for _ in tqdm(range(inp.num_photons)):
            # do_one_run(inp, layers, out)
            do_one_run(_inp, _layers, out)

        # QUEUE_LIMIT = 64
        # with ThreadPoolExecutor() as executor:
            # futures = []
            # for _ in tqdm(range(inp.num_photons)):
                # f = executor.submit(do_one_run, inp, layers, out)
                # futures.append(f)

                # if len(futures) > QUEUE_LIMIT:
                    # done, not_done = wait(futures, timeout=1, return_when=FIRST_COMPLETED)

                    # # consume finished
                    # [i.result() for i in done]
                    # # keep unfinished
                    # futures = list(not_done)



if __name__ == "__main__":
    main()
