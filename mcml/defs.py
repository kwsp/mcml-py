"""
Centimeters [cm] are used throughout as the base unit
Absorption and scattering coefficients measured in [cm^{-1}]
"""
from dataclasses import dataclass
import numpy as np


WEIGHT = 1e-4
CHANGE = 0.1


@dataclass
class Layer:
    # z coordinates of the layer [cm]
    z0: float
    z1: float

    n: float  # refractive index
    mua: float  # absorption coefficient
    mus: float  # scattering coefficient
    g: float  # anisotropy

    # cosine of the critical angle
    cos_crit0: float
    cos_crit0: float


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


@dataclass
class OutputParams:
    rsp: float  # specular reflectance
    rd_ra: np.ndarray[1, float]  # 2D distribution of diffuse reflectance [1/cm^2 sr]
    rd_r: np.ndarray[1, float]  # 1D radial distribution of diffuse reflectance  [1/cm2]
    rd_a: np.ndarray[
        1, float
    ]  # 1D angular distribution of diffuse reflectance  [1/cm2]
    rd: float  # total diffuse reflectance [1/cm2]

    a_rz: np.ndarray[
        2, float
    ]  # 2D probability density in turbid media over r & z [1/cm3]
    a_z: np.ndarray[1, float]  # 1D probability density over z [1/cm]
    a_l: np.ndarray[1, float]  # each layer's absorption probability [1/cm]
    a: float  # total absorption probability # [1/cm]

    tt_ra: np.ndarray[2, float]  # 2D distribution of total transmittance [1/(cm2 sr)]
    tt_r: np.ndarray[1, float]  # 1D radial distribution of transmittance [1/(cm2)]
    tt_a: np.ndarray[1, float]  # 1D angular distribution of transmittance [1/sr]
    tt: float  # total transmittance [1/sr]


@dataclass
class Photon:
    # cartesian coordinates of photon packet
    x: float
    y: float
    z: float

    # direction cosines of photon propagation
    ux: float
    uy: float
    uz: float

    w: float  # weight
    dead: bool  # False if photon is propagating, True if terminated

    scatters: int  # no. of scattering events experienced
    layer: int  # index of layer where photon packet resides
    s_: float  # dimensionless step size
