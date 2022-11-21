import numpy as np
from numba import njit
import numba as nb

from mcml.defs import Layer, InputParams, CHANCE
from mcml.photon import Photon
from mcml.rand import get_random

Layers = list[Layer]


@njit
def calc_r_specular(layers: Layers) -> float:
    """
    Compute the specular reflection.

    If the first layer is a turbid medium, use the Fresnel reflection
    from the boundary of the first layer as the specular reflectance.

    If the first layer is glass, multiple reflections in the first layer
    is considered to get the specular reflectance.
    """

    # Use Eq. (3.8)
    r1 = ((layers[0].n - layers[1].n) / (layers[0].n + layers[1].n)) ** 2

    if layers[1].mua == 0.0 and layers[1].mus == 0.0:
        # First layer is a clear medium (glass layer)
        # Eq. (3.9)
        r2 = ((layers[1].n - layers[2].n) / (layers[1].n + layers[2].n)) ** 2

        r1 += (1 - r1) ** 2 * r2 / (1 - r1 * r2)
    return r1


@njit
def spin_theta(g: float) -> float:
    """
    Sample a new theta angle for photon propagation given
    the anisotropy g.
    """
    if g == 0.0:
        cost = 2.0 * get_random() - 1.0
    else:
        temp = (1.0 - g * g) / (1.0 - g + 2.0 * g * get_random())
        cost = (1.0 + g * g - temp * temp) / (2.0 * g)
        cost = max(min(cost, 1.0), -1.0)
    return cost  # cos(theta)


@njit
def spin(g: float, photon: Photon):
    """
    Choose a new direction for photon propagation by sampling
    the polar deflection angle theta and the azimuthal angle psi.
    """
    ux = photon.ux
    uy = photon.uy
    uz = photon.uz

    cost = spin_theta(g)  # cos theta
    sint = np.sqrt(1.0 - cost * cost)

    psi = 2.0 * np.pi * get_random()
    cosp = np.cos(psi)
    sinp = 0.0
    if psi < np.pi:
        sinp = np.sqrt(1.0 - cosp * cosp)
    else:
        sinp = -np.sqrt(1.0 - cosp * cosp)

    if np.abs(1.0 - uz) < 1e-8:
        # normal incidence
        # Eq. (3.31)
        photon.ux = sint * cosp
        photon.uy = sint * sinp
        photon.uz = cost * np.sign(uz)
    else:
        # regular incidence
        # Eq. (3.30)
        tmp = np.sqrt(1.0 - uz * uz) + 1e-9
        photon.ux = sint * (ux * uz * cosp - uy * sinp) / tmp + ux * cost
        photon.uy = sint * (uy * uz * cosp + ux * sinp) / tmp + uy * cost
        photon.uz = -sint * cosp * tmp + uz * cost


@njit
def update_step_size_in_glass(photon: Photon, layers: Layers):
    """
    If uz != 0, return the photon step size in glass.
    Otherwise return 0.

    The step size is the distance between the current position and
    the boundary in the photon direction.

    Make sure uz != 0 before calling this function.
    """
    layer = photon.layer
    uz = photon.uz
    dl_b = 0.0
    if uz > 0.0:
        dl_b = (layers[layer].z1 - photon.z) / uz
    elif uz < 0.0:
        dl_b = (layers[layer].z0 - photon.z) / uz

    photon.s = dl_b


@njit
def update_step_size_in_tissue(photon: Photon, layers: Layers):
    """
    Pick a step size for a photon packet when it is in tissue.
    If sleft is zero, make a new step size with -log(rnd)/(mua + mus)
    Else, pick up the leftover in sleft
    """
    layer = photon.layer
    mua = layers[layer].mua
    mus = layers[layer].mus

    if photon.sleft == 0.0:
        # make new step
        # Eq. (3.19)
        photon.s = -np.log(get_random()) / (mua + mus)
    else:
        # take leftover
        photon.s = photon.sleft / (mua + mus)
        photon.sleft = 0.0


@njit
def hop(photon: Photon):
    """
    Move the photon s away in the current layer of medium
    """

    # Eq. (3.23)
    s = photon.s
    photon.x += s * photon.ux
    photon.y += s * photon.uy
    photon.z += s * photon.uz


@njit
def roulette(photon: Photon):
    """
    The photon weight is small, and the photon packet tries to survive a roulette
    """
    # Eq. (3.44)
    if photon.w == 0.0:
        photon.dead = True
    elif get_random() < CHANCE:
        photon.w /= CHANCE
    else:
        photon.dead = True


@njit
def hit_boundary(photon: Photon, layers: Layers) -> bool:
    """
    Check if the step will hit the boundary.
    Return True if hit

    If hit, photon.s and sleft are updated
    """
    layer = layers[photon.layer]
    uz = photon.uz

    # distance to the boundary
    # Eq. (3.32)
    if uz > 0.0:
        dl_b = (layer.z1 - photon.z) / uz
    elif uz < 0.0:
        dl_b = (layer.z0 - photon.z) / uz
    else:
        raise ValueError("photon.uz == 0.0, dl_b = inf")

    if uz != 0.0 and photon.s > dl_b:
        # Not horizontal crossing
        mut = layer.mua + layer.mus
        photon.sleft = (photon.s - dl_b) * mut
        photon.s = dl_b
        return True

    return False


@njit
def drop(photon: Photon, inp: InputParams, layers: Layers, a_rz: np.ndarray):
    """
    Drop photon weight inside the tissue (not glass)

    The photon is assumed not dead.
    Weight drop dw = w * mua / mut
    The dropped weight is assigned to the absorption array elements
    """
    x = photon.x
    y = photon.y
    layer = layers[photon.layer]

    # compute array indices
    iz = nb.uint(photon.z / inp.dz)
    iz = nb.uint(min(iz, inp.nz - 1))

    ir = nb.uint(np.sqrt(x * x + y * y) / inp.dr)
    ir = nb.uint(min(ir, inp.nr - 1))

    # update photon weight
    mua = layer.mua
    mus = layer.mus
    dw = photon.w * mua / (mua + mus)
    photon.w -= dw

    # assign dw to the absorption array element
    a_rz[ir][iz] += dw


@njit
def r_fresnel(
    n1: float,  # incident refractive index
    n2: float,  # transmit refractive index
    ca1: float,  # cosine of the incident angle.
) -> tuple[float, float]:
    """
    Compute the Fresnel reflectance

    Make sure ca1 is positive
    """
    ca2 = 0
    r = 0.0
    if n1 == n2:
        # matched boundary
        ca2 = ca1
        r = 0.0
    elif np.abs(1.0 - ca1) < 1e-8:
        # normal incidence
        ca2 = ca1
        r = (n2 - n1) / (n2 + n1)
        r *= r
    elif np.abs(ca1) < 1e-8:
        # very slanted
        ca2 = 0.0
        r = 1.0
    else:
        # general
        sa1 = np.sqrt(1 - ca1 * ca1)
        sa2 = n1 * sa1 / n2
        if sa2 >= 1.0:
            # check total internal reflection
            ca2 = 0.0
            r = 1.0
        else:
            ca2 = np.sqrt(1 - sa2 * sa2)
            cap = ca1 * ca2 - sa1 * sa2  # c+ = cc - ss
            cam = ca1 * ca2 + sa1 * sa2  # c- = cc + ss
            sap = sa1 * ca2 + ca1 * sa2  # s+ = sc + cs
            sam = sa1 * ca2 - ca1 * sa2  # s- = sc - cs
            r = (
                0.5
                * sam
                * sam
                * (cam * cam + cap * cap)
                / (sap * sap * cam * cam + 1e-9)
            )

    return r, ca2


@njit
def record_r(refl: float, photon: Photon, inp: InputParams, rd_ra: np.ndarray):
    """
    Record the photon weight exiting the first layer (uz<0)
    to the reflection array

    Update photon weight
    """
    x = photon.x
    y = photon.y

    ir = nb.uint(np.sqrt(x * x + y * y) / inp.dr)
    ir = np.uint(min(ir, inp.nr - 1))

    ia = nb.uint(np.arccos(-photon.uz) / inp.da)
    ia = nb.uint(min(ia, inp.na - 1))

    # assign photon to the reflection array
    rd_ra[ir][ia] += photon.w * (1.0 - refl)

    photon.w *= refl


@njit
def record_t(refl: float, photon: Photon, inp: InputParams, tt_ra: np.ndarray):
    """
    Record the photon weight exiting the last layer (uz>0)
    to the transmittance array

    Update photon weight
    """
    x = photon.x
    y = photon.y

    ir = nb.uint(np.sqrt(x * x + y * y) / inp.dr)
    ir = nb.uint(min(ir, inp.nr - 1))

    ia = nb.uint(np.arccos(-photon.uz) / inp.da)
    ia = nb.uint(min(ia, inp.na - 1))

    # assign photon to the reflection array
    tt_ra[ir][ia] += photon.w * (1.0 - refl)

    photon.w *= refl


@njit
def cross_up_or_not(
    photon: Photon, inp: InputParams, layers: Layers, rd_ra: np.ndarray
):
    """
    Decide whether the photon will be transmitted or reflected in the
    upper boundary (uz < 0) of the current layer
    """
    uz = photon.uz
    uz1 = 0.0

    r = 0.0  # reflectance
    layer_idx = photon.layer
    ni = layers[layer_idx].n
    nt = layers[layer_idx - 1].n

    if (-uz) <= layers[layer_idx].cos_crit0:
        # total internal reflection
        r = 1.0
    else:
        r, uz1 = r_fresnel(ni, nt, -uz)

    if get_random() > r:
        # transmitted to layer - 1
        if layer_idx == 1:
            photon.uz = -uz1
            record_r(0.0, photon, inp, rd_ra)
            photon.dead = True
        else:
            photon.layer -= 1
            photon.ux *= ni / nt
            photon.uy *= ni / nt
            photon.uz = -uz1
    else:
        # reflected
        photon.uz = -uz


@njit
def cross_down_or_not(
    photon: Photon, inp: InputParams, layers: Layers, tt_ra: np.ndarray
):
    """
    Decide whether the photon will be transmitted or reflected in the
    lower boundary (uz > 0) of the current layer
    """
    uz = photon.uz
    uz1 = 0.0  # cosine of transmission alpha

    r = 0.0  # reflectance
    layer_idx = photon.layer
    ni = layers[layer_idx].n
    nt = layers[layer_idx + 1].n

    if uz <= layers[layer_idx].cos_crit1:
        # total internal reflection
        r = 1.0
    else:
        r, uz1 = r_fresnel(ni, nt, uz)

    if get_random() > r:
        # transmitted to layer + 1
        if layer_idx == inp.num_layers:
            photon.uz = uz1
            record_t(0.0, photon, inp, tt_ra)
            photon.dead = True
        else:
            photon.layer += 1
            photon.ux *= ni / nt
            photon.uy *= ni / nt
            photon.uz = uz1
    else:
        # reflected
        photon.uz = -uz


@njit
def cross_or_not(
    photon: Photon,
    inp: InputParams,
    layers: Layers,
    rd_ra: np.ndarray,
    tt_ra: np.ndarray,
):
    if photon.uz < 0.0:
        cross_up_or_not(photon, inp, layers, rd_ra)
    else:
        cross_down_or_not(photon, inp, layers, tt_ra)


@njit
def hop_in_glass(
    photon: Photon,
    inp: InputParams,
    layers: Layers,
    rd_ra: np.ndarray,
    tt_ra: np.ndarray,
):
    """
    Move the photon packet in glass layer
    Horizontal photons are killed because they'll never interact
    with tissue again
    """
    if photon.uz == 0.0:
        # horizontal photon in glass is killed
        photon.dead = True
    else:
        update_step_size_in_glass(photon, layers)
        hop(photon)
        cross_or_not(photon, inp, layers, rd_ra, tt_ra)


@njit
def hop_drop_spin_in_tissue(
    photon: Photon,
    inp: InputParams,
    layers: Layers,
    a_rz: np.ndarray,
    rd_ra: np.ndarray,
    tt_ra: np.ndarray,
):
    """
    Set a step size, move the photon, drop some weight,
    choose a new photon direction for propagation.

    If the step size is long enough to hit and interface,
    move the photon to the bondary free of absorption/scattering,
    then decide whether to reflect or transmit.
    Then move the photon in the current or transmission medium with the
    unfinished stepsize to interaction site.
    """
    update_step_size_in_tissue(photon, layers)

    if hit_boundary(photon, layers):
        hop(photon)
        cross_or_not(photon, inp, layers, rd_ra, tt_ra)
    else:
        hop(photon)
        drop(photon, inp, layers, a_rz)
        spin(layers[photon.layer].g, photon)


@njit
def hop_drop_spin(
    photon: Photon,
    inp: InputParams,
    layers: Layers,
    a_rz: np.ndarray,
    rd_ra: np.ndarray,
    tt_ra,
):
    layer = layers[photon.layer]
    if layer.mua == 0.0 and layer.mus == 0.0:
        hop_in_glass(photon, inp, layers, rd_ra, tt_ra)
    else:
        hop_drop_spin_in_tissue(photon, inp, layers, a_rz, rd_ra, tt_ra)

    if photon.w < inp.wth and not photon.dead:
        roulette(photon)
