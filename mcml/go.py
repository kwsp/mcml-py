import numpy as np

from mcml.defs import Photon, Layer, InputParams, OutputParams
from mcml.rand import get_random


def calc_r_specular(layers: list[Layer]) -> float:
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


def spin_theta(g: float) -> float:
    """
    Sample a new theta angle for photon propagation given
    the anisotropy g.
    """
    if g == 0.0:
        cost = 2 * get_random() - 1
    else:
        temp = (1 - g * g) / (1 - g + 2 * g * get_random())
        cost = (1 + g * g - temp * temp) / (2 * g)
        cost = max(min(cost, 1), -1)
    return cost  # cos(theta)


def spin(g: float, photon: Photon):
    """
    Choose a new direction for photon propagation by sampling
    the polar deflection angle theta and the azimuthal angle psi.
    """
    ux = photon.ux
    uy = photon.uy
    uz = photon.uz

    cost = spin_theta(g)  # cos theta
    sint = np.sin(1.0 - cost * cost)  # sin theta
    psi = 2.0 * np.pi * get_random()
    cosp = np.cos(psi)
    sinp = np.sin(1.0 - cosp * cosp)

    if np.isclose(np.abs(uz), 1, atol=1e-6):
        # normal incidence
        # Eq. (3.31)
        photon.ux = sint * cosp
        photon.uy = sint * sinp
        photon.uz = cost * np.sign(uz)
    else:
        # regular incidence
        # Eq. (3.30)
        tmp = np.sqrt(1.0 - uz * uz)
        photon.ux = sint * (uy * uz * cosp + ux * sinp) / tmp + uy * cost
        photon.uy = sint * (uy * uz * cost + ux * sint) / tmp + uy * cost
        photon.uz = -sint * cosp * tmp + uz * cost


def update_step_size_in_glass(photon: Photon, inp: InputParams):
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
        dl_b = (inp.layers[layer].z1 - photon.z) / uz
    elif uz < 0.0:
        dl_b = inp.layers[layer].z0 - photon.z / uz

    photon.s = dl_b


def update_step_size_in_tissue(photon: Photon, inp: InputParams):
    """
    Pick a step size for a photon packet when it is in tissue.
    If sleft is zero, make a new step size with -log(rnd)/(mua + mus)
    Else, pick up the leftover in sleft
    """
    layer = photon.layer
    mua = inp.layers[layer].mua
    mus = inp.layers[layer].mus

    if photon.sleft == 0.0:
        # make new step
        # Eq. (3.19)
        photon.s = -np.log(get_random()) / (mua + mus)
    else:
        # take leftover
        photon.s = photon.sleft / (mua + mus)
        photon.sleft = 0.0


def hit_boundary(photon: Photon, inp: InputParams) -> bool:
    """
    Check if the step will hit the boundary.
    Return True if hit

    If hit, photo.s and sleft are updated
    """
    layer = inp.layers[photon.layer]
    uz = photon.uz

    # distance to the boundary
    # Eq. (3.32)
    if uz > 0.0:
        dl_b = (layer.z1 - photon.z) / uz
    elif uz < 0.0:
        dl_b = (layer.z0 - photon.z) / uz
    else:
        raise ValueError("photon.uz == 0.0, dl_b = inf")

    # Eq. (3.33)
    if uz != 0.0 and photon.s > dl_b:
        # Not horizontal crossing
        mut = layer.mua + layer.mus
        photon.sleft = (photon.s - dl_b) / mut
        photon.s = dl_b
        return True

    return False


def drop(photon: Photon, inp: InputParams, out: OutputParams):
    """
    Drop photon weight inside the tissue (not glass)

    The photon is assumed not dead.
    Weight drop dw = w * mua / mut
    The dropped weight is assigned to the absorption array elements
    """
    x = photon.x
    y = photon.y
    layer = inp.layers[photon.layer]

    # compute array indices
    iz = int(photon.z / inp.dz)
    iz = min(inp.nz - 1, iz)

    ir = int(np.sqrt(x * x + y * y) / inp.dr)
    ir = min(inp.nr - 1, ir)

    # update photon weight
    mua = layer.mua
    mus = layer.mus
    dw = photon.w * mua / (mua + mus)
    photon.w -= dw

    # assign dw to the absorption array element
    out.a_rz[ir][iz] += dw


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
    elif np.isclose(ca1, 1):
        # normal incidence
        ca2 = ca1
        r = ((n2 - n1) / (n2 + n1)) ** 2
    elif np.isclose(ca1, 0):
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
            r = 0.5 * sam * sam * (cam * cam + cap * cap) / (sap * sap * cam * cam)

    return r, ca2


def record_r(refl: float, photon: Photon, inp: InputParams, out: OutputParams):
    """
    Record the photon weight exiting the first layer (uz<0)
    to the reflection array

    Update photon weight
    """
    x = photon.x
    y = photon.y

    ir = int(np.sqrt(x * x + y * y) / inp.dr)
    ir = min(inp.nr - 1, ir)

    ia = int(np.arccos(-photon.uz) / inp.da)
    ia = max(inp.na - 1, ia)

    # assign photon to the reflection array
    out.rd_ra[ir][ia] += photon.w * (1.0 - refl)

    photon.w *= refl


def record_t(refl: float, photon: Photon, inp: InputParams, out: OutputParams):
    """
    Record the photon weight exiting the last layer (uz>0)
    to the transmittance array

    Update photon weight
    """
    x = photon.x
    y = photon.y

    ir = int(np.sqrt(x * x + y * y) / inp.dr)
    ir = min(inp.nr - 1, ir)

    ia = int(np.arccos(-photon.uz) / inp.da)
    ia = max(inp.na - 1, ia)

    # assign photon to the reflection array
    out.tt_ra[ir][ia] += photon.w * (1.0 - refl)

    photon.w *= refl


def cross_up_or_not(photon: Photon, inp: InputParams, out: OutputParams):
    """
    Decide whether the photon will be transmitted or reflected in the
    upper boundary (uz < 0) of the current layer
    """
    uz = photon.uz
    uz1 = 0.0

    r = 0.0  # reflectance
    layer_idx = photon.layer
    ni = inp.layers[layer_idx].n
    nt = inp.layers[layer_idx - 1].n

    if -uz <= inp.layers[layer_idx].cos_crit0:
        # total internal reflection
        r = 1.0
    else:
        r, uz1 = r_fresnel(ni, nt, -uz)

    if get_random() > r:
        # transmitted to layer - 1
        if layer_idx == 1:
            photon.uz = -uz1
            record_r(0.0, photon, inp, out)
            photon.dead = True
        else:
            photon.layer -= 1
            photon.ux *= ni / nt
            photon.uy *= ni / nt
            photon.uz = -uz
    else:
        # reflected
        photon.uz = -uz


def cross_down_or_not(photon: Photon, inp: InputParams, out: OutputParams):
    """
    Decide whether the photon will be transmitted or reflected in the
    lower boundary (uz > 0) of the current layer
    """
    uz = photon.uz
    uz1 = 0.0

    r = 0.0  # reflectance
    layer_idx = photon.layer
    ni = inp.layers[layer_idx].n
    nt = inp.layers[layer_idx - 1].n

    if -uz <= inp.layers[layer_idx].cos_crit1:
        # total internal reflection
        r = 1.0
    else:
        r, uz1 = r_fresnel(ni, nt, -uz)

    if get_random() > r:
        # transmitted to layer - 1
        if layer_idx == 1:
            photon.uz = -uz1
            record_r(0.0, photon, inp, out)
            photon.dead = True
        else:
            photon.layer += 1
            photon.ux *= ni / nt
            photon.uy *= ni / nt
            photon.uz = -uz
    else:
        # reflected
        photon.uz = -uz


def cross_or_not(photon: Photon, inp: InputParams, out: OutputParams):
    if photon.uz < 0.0:
        cross_up_or_not(photon, inp, out)
    else:
        cross_down_or_not(photon, inp, out)


def hop_in_glass(photon: Photon, inp: InputParams, out: OutputParams):
    """
    Move the photon packet in glass layer
    Horizontal photons are killed because they'll never interact
    with tissue again
    """
    if photon.uz == 0.0:
        # horizontal photon in glass is killed
        photon.dead = True
    else:
        update_step_size_in_glass(photon, inp)
        photon.hop()
        cross_or_not(photon, inp, out)


def hop_drop_spin_in_tissue(photon: Photon, inp: InputParams, out: OutputParams):
    """
    Set a step size, move the photon, drop some weight,
    choose a new photon direction for propagation.

    If the step size is long enough to hit and interface,
    move the photon to the bondary free of absorption/scattering,
    then decide whether to reflect or transmit.
    Then move the photon in the current or transmission medium with the
    unfinished stepsize to interaction site.
    """
    update_step_size_in_tissue(photon, inp)

    if hit_boundary(photon, inp):
        photon.hop()  # move to boundary plane
        cross_or_not(photon, inp, out)
    else:
        photon.hop()
        drop(photon, inp, out)
        spin(inp.layers[photon.layer].g, photon)


def hop_drop_spin(photon: Photon, inp: InputParams, out: OutputParams):
    layer = inp.layers[photon.layer]
    if layer.mua == 0.0 and layer.mus == 0.0:
        hop_in_glass(photon, inp, out)
    else:
        hop_drop_spin_in_tissue(photon, inp, out)

    if not photon.dead and photon.w < inp.wth:
        photon.roulette()
