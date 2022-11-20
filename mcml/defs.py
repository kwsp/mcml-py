"""
Centimeters [cm] are used throughout as the base unit
Absorption and scattering coefficients measured in [cm^{-1}]
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from mcml.rand import get_random


WEIGHT = 1e-4
CHANCE = 0.1


@dataclass
class Layer:
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


@dataclass
class InputParams:
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
    layers: list[Layer]

    @classmethod
    def read_mci(cls, fname: str) -> list[InputParams]:
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
        n_runs = parse(next(it), int)[0]

        inputs: list[InputParams] = []

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
            layers: list[Layer] = []
            n_layers = parse(next(it), int)[0]
            assert n_layers > 0

            n1 = parse(next(it), int)[0]
            assert n1 > 0
            layers.append(Layer(n=n1))  # top ambient layer

            z = 0.0
            for _ in range(n_layers):
                n1, _mua, _mus, _g, _d = parse(next(it), float)[:5]

                layers.append(Layer(n=n1, mua=_mua, mus=_mus, g=_g, z0=z, z1=z + _d))

                z += _d

            n1 = parse(next(it), int)[0]
            assert n1 > 0
            layers.append(Layer(n=n1))  # top ambient layer
            # bottom ambient layer

            # Set critical angles
            for i in range(1, n_layers + 1):
                n1 = layers[i].n
                n2 = layers[i - 1].n
                cos_crit0 = np.sqrt(1.0 - n2 * n2 / (n1 * n1)) if n1 > n2 else 0.0

                n2 = layers[i + 1].n
                cos_crit1 = np.sqrt(1.0 - n2 * n2 / (n1 * n1)) if n1 > n2 else 0.0

                layers[i].cos_crit0 = cos_crit0
                layers[i].cos_crit1 = cos_crit1

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
                layers=layers,
            )
            inputs.append(inp)

        return inputs


@dataclass
class OutputParams:
    # specular reflectance
    rsp: float
    rd_ra: np.ndarray  # 2D distribution of diffuse reflectance [1/cm^2 sr]
    rd_r: np.ndarray  # 1D radial distribution of diffuse reflectance  [1/cm2]
    rd_a: np.ndarray  # 1D angular distribution of diffuse reflectance  [1/cm2]
    rd: float  # total diffuse reflectance [1/cm2]

    a_rz: np.ndarray  # 2D probability density in turbid media over r & z [1/cm3]
    a_z: np.ndarray  # 1D probability density over z [1/cm]
    a_l: np.ndarray  # each layer's absorption probability [1/cm]
    a: float  # total absorption probability # [1/cm]

    tt_ra: np.ndarray  # 2D distribution of total transmittance [1/(cm2 sr)]
    tt_r: np.ndarray  # 1D radial distribution of transmittance [1/(cm2)]
    tt_a: np.ndarray  # 1D angular distribution of transmittance [1/sr]
    tt: float  # total transmittance [1/sr]

    @classmethod
    def init(cls, rsp: float, inp: InputParams) -> OutputParams:
        nz = inp.nz
        nr = inp.nr
        na = inp.na
        nl = inp.num_layers
        assert nz > 0 and nr > 0 and na > 0

        return cls(
            rsp=rsp,
            rd=0.0,
            a=0.0,
            tt=0.0,
            rd_ra=np.zeros((nr, na)),
            rd_r=np.zeros(nr),
            rd_a=np.zeros(na),
            a_rz=np.zeros((nr, nz)),
            a_z=np.zeros(nz),
            a_l=np.zeros(nl),
            tt_ra=np.zeros((nr, na)),
            tt_r=np.zeros(nr),
            tt_a=np.zeros(na),
        )


@dataclass
class Photon:
    w: float = 1  # weight
    layer: int = 1  # index of layer where photon packet resides
    # scatters: int = 0  # no. of scattering events experienced
    s: float = 0  # dimensionless step size
    sleft: float = 0  # dimensionless step size
    dead: bool = False  # False if photon is propagating, True if terminated

    # cartesian coordinates of photon packet
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    # direction cosines of photon propagation
    ux: float = 0.0
    uy: float = 0.0
    uz: float = 1.0

    @classmethod
    def init(cls, r_sp: float, layers: list[Layer]) -> Photon:
        w = 1 - r_sp
        if layers[1].mua == 0.0 and layers[1].mus == 0.0:
            layer = 2
            z = layers[2].z0
            return cls(w=w, layer=layer, z=z)

        return cls(w=w)

    def hop(self):
        """
        Move the photon s away in the current layer of medium
        """

        # Eq. (3.23)
        s = self.s
        self.x += s * self.ux
        self.y += s * self.uy
        self.z += s * self.uz

    def roulette(self):
        """
        The photon weight is small, and the photon packet tries to survive a roulette
        """
        # Eq. (3.44)
        if self.w == 0.0:
            self.dead = True
        elif get_random() < CHANCE:
            self.w /= CHANCE
        else:
            self.dead = True
