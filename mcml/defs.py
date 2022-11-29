"""
Centimeters [cm] are used throughout as the base unit
Absorption and scattering coefficients measured in [cm^{-1}]
"""
from __future__ import annotations
from typing import NamedTuple
from dataclasses import dataclass
import numpy as np
import copy


WEIGHT = 1e-4
CHANCE = 0.1


# class StructMixin:
# def to_struct(self):
# """
# Convert a dataclass instance to a numpy structured array
# https://numpy.org/doc/stable/user/basics.rec.html
# """
# types = []
# vals = []

# for field in fields(self):
# name = field.name
# if name.startswith("_"):
# continue

# val = getattr(self, field.name)
# _type = field.type

# if _type.startswith("list"):
# # special case this for now
# # TODO: find generic way to handle nested
# # sequence containers
# continue

# # assume the sequence contains a dataclass that
# # can be converted into a numpy structured array
# # val = np.hstack([v.to_struct() for v in val])
# # _type = val.dtype

# types.append((name, _type))
# vals.append(val)

# dtype = np.dtype(types, align=True)
# try:
# _struct = np.array([tuple(vals)], dtype=dtype)
# except Exception as e:
# breakpoint()
# print("")
# return _struct


class Layer(NamedTuple):
    n: float = 1.0  # refractive index
    mua: float = 0.0  # absorption coefficient
    mus: float = 0.0  # scattering coefficient
    g: float = 0.0  # anisotropy

    # z coordinates of the layer [cm]
    z0: float = 0.0
    z1: float = 0.0

    # cosine of the critical angle
    cos_crit0: float = 0.0
    cos_crit1: float = 0.0


class InputParams(NamedTuple):
    ### 2D homogenous grid system in the r and z direction
    # total no. of grid elements for r, z, and alpha
    nr: int
    nz: int
    na: int

    # grid element sizes in the r, z, and alpha
    dr: float
    dz: float
    da: float  # da = pi / (2 * Na)

    num_photons: int  # no. of photons
    wth: float  # Roulette if photon weight < wth

    num_layers: int  # no. of layers


def read_mci(fname: str) -> list[tuple[InputParams, list[Layer], str]]:
    """
    Returns a list of tuples containing (inp_params, layers, out_fname)

    """
    with open(fname, "r") as fp:
        lines = fp.read().splitlines()
    lines = [l.strip() for l in lines]
    lines = [l for l in lines if l and not l.startswith("#")]

    def clean_line(s: str):
        idx = s.find("#")
        if idx > 0:
            s = s[:idx].strip()
        return s

    def parse(line: str, _type) -> list:
        line = clean_line(line)
        return [_type(x) for x in line.split()]

    it = iter(lines)
    ver = parse(next(it), str)[0]
    print(f"Reading input file, version {ver}")

    n_runs = parse(next(it), int)[0]

    inputs: list[tuple[InputParams, list[Layer], str]] = []

    for _ in range(n_runs):
        out_fname = parse(next(it), str)[0]

        n_photons = parse(next(it), int)[0]
        assert n_photons > 0
        dz, dr = parse(next(it), float)[:2]
        assert dz > 0
        assert dr > 0
        nz, nr, na = parse(next(it), int)[:3]
        assert nz > 0
        assert nr > 0
        assert na > 0
        da = 0.5 * np.pi / na

        ## Parse layers
        n_layers = parse(next(it), int)[0]
        assert n_layers > 0

        n1 = parse(next(it), float)[0]
        assert n1 > 0

        # parse into list of dicts first since
        # Layer is a NamedTuple and therefore immutable
        _layers: list[dict] = []
        _layers.append(dict(n=n1))  # top ambient layer

        z = 0.0
        for _ in range(n_layers):
            n1, _mua, _mus, _g, _d = parse(next(it), float)[:5]
            # layers.append(Layer(n=n1, mua=_mua, mus=_mus, g=_g, z0=z, z1=z + _d))
            _layers.append(dict(n=n1, mua=_mua, mus=_mus, g=_g, z0=z, z1=z + _d))
            z += _d

        n1 = parse(next(it), float)[0]
        assert n1 > 0
        _layers.append(dict(n=n1))  # top ambient layer
        # bottom ambient layer

        # Set critical angles
        for i in range(1, n_layers + 1):
            n1 = _layers[i]["n"]
            n2 = _layers[i - 1]["n"]
            cos_crit0 = np.sqrt(1.0 - n2 * n2 / (n1 * n1)) if n1 > n2 else 0.0

            n2 = _layers[i + 1]["n"]
            cos_crit1 = np.sqrt(1.0 - n2 * n2 / (n1 * n1)) if n1 > n2 else 0.0

            _layers[i]["cos_crit0"] = cos_crit0
            _layers[i]["cos_crit1"] = cos_crit1

        layers: list[Layer] = [Layer(**l) for l in _layers]

        inp = InputParams(
            nr=nr,
            nz=nz,
            na=na,
            dr=dr,
            dz=dz,
            da=da,
            num_photons=n_photons,
            wth=WEIGHT,
            num_layers=n_layers,
        )
        inputs.append((inp, layers, out_fname))

    return inputs


@dataclass
class OutputParams:
    # specular reflectance
    rsp: float

    rd_ra: np.ndarray  # 2D distribution of diffuse reflectance [1/cm^2 sr]
    a_rz: np.ndarray  # 2D probability density in turbid media over r & z [1/cm3]
    tt_ra: np.ndarray  # 2D distribution of total transmittance [1/(cm2 sr)]

    rd_r: np.ndarray | None = (
        None  # 1D radial distribution of diffuse reflectance  [1/cm2]
    )
    rd_a: np.ndarray | None = (
        None  # 1D angular distribution of diffuse reflectance  [1/cm2]
    )
    rd: float = 0.0  # total diffuse reflectance [1/cm2]

    a_z: np.ndarray | None = None  # 1D probability density over z [1/cm]
    a_l: np.ndarray | None = None  # each layer's absorption probability [1/cm]
    a: float = 0.0  # total absorption probability # [1/cm]

    tt_r: np.ndarray | None = None  # 1D radial distribution of transmittance [1/(cm2)]
    tt_a: np.ndarray | None = None  # 1D angular distribution of transmittance [1/sr]
    tt: float = 0.0  # total transmittance [1/sr]

    @classmethod
    def init(cls, rsp: float, inp: InputParams) -> OutputParams:
        nz = inp.nz
        nr = inp.nr
        na = inp.na
        assert nz > 0 and nr > 0 and na > 0

        return cls(
            rsp=rsp,
            rd=0.0,
            a=0.0,
            tt=0.0,
            rd_ra=np.zeros((nr, na)),
            a_rz=np.zeros((nr, nz)),
            tt_ra=np.zeros((nr, na)),
        )

    def __add__(self, other: OutputParams):
        assert self.rsp == other.rsp
        res = copy.deepcopy(self)
        res.rd_ra += other.rd_ra
        res.a_rz += other.a_rz
        res.tt_ra += other.tt_ra
        return res


def write_mco(
    fname: str,
    inp: InputParams,
    layers: list[Layer],
    out: OutputParams,
    elapsed_s: float,
):
    s = open(fname, "w")

    s.write("A1 # Output file produced by mcml-py\n\n")
    s.write(
        """
####
# Data categories include: 
# InParm, RAT, 
# A_l, A_z, Rd_r, Rd_a, Tt_r, Tt_a, 
# A_rz, Rd_ra, Tt_ra 
####\n\n"""
    )
    elapsed_h = elapsed_s / 60 / 60
    s.write(
        f"User time: {elapsed_s:.2f} sec = {elapsed_h:.2f} hr. Simulation time of this run.\n"
    )

    ### Write input params
    s.write("\nInParm  # Input parameters. cm is used\n")
    s.write(f"{fname} A  # output filename.\n")
    s.write(f"{inp.num_photons}  # No. of photons\n")
    s.write(f"{inp.dz} {inp.dr}  # dz, dr [cm]\n")
    s.write(f"{inp.nz} {inp.nr} {inp.na}  # No. of dz, dr, da.\n")

    s.write(f"\n{inp.num_layers}  # No. of layers\n")
    s.write("#n	mua	mus	g	d	# One line for each layer\n")
    s.write(f"{layers[0].n}  # n for medium above\n")
    for i in range(1, inp.num_layers + 1):
        d = layers[i].z1 - layers[i].z0
        s.write(
            f"{layers[i].n} {layers[i].mua} {layers[i].mus} {layers[i].g} {d}  # n for medium above\n"
        )
    s.write(f"{layers[-1].n}  # n for medium below\n")

    ### RAT
    s.write("\nRAT  # Reflectance, absorption, transmission. \n")
    s.write(f"{out.rsp:.6g}  # Specular reflectance [-]\n")
    s.write(f"{out.rd:.6g}  # Diffuse reflectance [-]\n")
    s.write(f"{out.a:.6g}  # Absorbed fraction [-]\n")
    s.write(f"{out.tt:.6g}  # Transmittance [-]\n\n")

    ### Write A_l, A_z
    s.write("A_l #Absorption as a function of layer. [-]\n")
    for i in range(1, inp.num_layers + 1):
        s.write(f"\t{out.a_l[i]:.6g}\n")

    s.write("\nA_z #A[0], [1],..A[nz-1]. [1/cm]\n")
    for i in range(inp.nz):
        s.write(f"\t{out.a_z[i]:.4e}\n")

    s.write("\nRd_r #Rd[0], [1],..Rd[nr-1]. [1/cm2]\n")
    for i in range(inp.nr):
        s.write(f"\t{out.rd_r[i]:.4e}\n")

    s.write("\nRd_a #Rd[0], [1],..Rd[na-1]. [sr-1]\n")
    for i in range(inp.na):
        s.write(f"\t{out.rd_a[i]:.4e}\n")

    s.write("\nTt_r #Tt[0], [1],..Tt[nr-1]. [1/cm2]\n")
    for i in range(inp.nr):
        s.write(f"\t{out.tt_r[i]:.4e}\n")

    s.write("\nTt_a #Tt[0], [1],..Tt[na-1]. [sr-1]\n")
    for i in range(inp.na):
        s.write(f"\t{out.tt_a[i]:.4e}\n")

    s.write(
        """\n
# A[r][z]. [1/cm3]
# A[0][0], [0][1],..[0][nz-1]
# A[1][0], [1][1],..[1][nz-1]
# ...
# A[nr-1][0], [nr-1][1],..[nr-1][nz-1]\nA_rz\n"""
    )
    for r in range(inp.nr):
        _s = "\t".join(f"{x:.4e}" for x in out.a_rz[r])
        s.write("\t")
        s.write(_s)
        s.write("\n")

    s.write(
        """\n
# Rd[r][angle]. [1/(cm2sr)].
# Rd[0][0], [0][1],..[0][na-1]
# Rd[1][0], [1][1],..[1][na-1]
# ...
# Rd[nr-1][0], [nr-1][1],..[nr-1][na-1]\nRd_ra"""
    )
    for r in range(inp.nr):
        _s = "\t".join(f"{x:.4e}" for x in out.rd_ra[r])
        s.write("\t")
        s.write(_s)
        s.write("\n")

    s.write(
        """\n
# Tt[r][angle]. [1/(cm2sr)].
# Tt[0][0], [0][1],..[0][na-1]
# Tt[1][0], [1][1],..[1][na-1]
# ...
# Tt[nr-1][0], [nr-1][1],..[nr-1][na-1]\nTt_ra\n"""
    )
    for r in range(inp.nr):
        _s = "\t".join(f"{x:.4e}" for x in out.tt_ra[r])
        s.write("\t")
        s.write(_s)
        s.write("\n")

    s.write("\n")
    s.close()
