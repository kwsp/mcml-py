"""
This file defines the Photon data structure described in Ch 3.4.2

The only public function from this file is `make_Photon`, which constructs a photon object.

    "x", "y", "z" represents the coordinates of a photon packet,
    (x, y, z), respectively.

    "ux", "uy", and "uz" represent the direction cosines of the propagation 
    direction of the photon packet, {μ_x, μ_y, μ_z}, respectively. 

    "w" represents the weight of the photon packet, W.

    "dead" repreesnts the status of the photon. It is set to True when the 
    photon is terminated.

    "s" represents the dimensionless step size, defined as the integration 
    of the extinction coefficient μ_t over the trajectory of the photon packet.
    In a homogenous medium, "s" is simply the physical path length multiplied
    by the extinction coefficient

    "layer" is the index of the layer in which the photon packet resides; 
    it is updated only when the photon leaves the current layer.


"""
from __future__ import annotations
from numba import njit
from numba.core import types
from numba.experimental import structref

from mcml.defs import Layer


@njit
def make_Photon(
    r_sp: float, layers: list[Layer], x=0.0, y=0.0, z=0.0, ux=0.0, uy=0.0, uz=1.0
) -> Photon:
    """
    Constructor for a Photon object.
    This is the only public API of this module

    Params
    ------
    r_sp: float. Specular reflectance
    layers: list[Layer]. List of Layer objects.

    # cartesian coordinates of photon packet
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    # direction cosines of photon propagation
    ux: float = 0.0
    uy: float = 0.0
    uz: float = 1.0


    Returns
    -------
    Photon
    """
    photon = Photon(
        w=1 - r_sp,
        layer=1,
        s=0.0,
        sleft=0.0,
        x=float(x),
        y=float(y),
        z=float(z),
        ux=float(ux),
        uy=float(uy),
        uz=float(uz),
        dead=False,
    )
    # check if the first layer is clear
    if layers[1].mua == 0.0 and layers[1].mus == 0.0:
        photon.layer = 2
        photon.z = layers[2].z0
    return photon


"""
Implementation details beyond this point.

Photon is defined as a Numba "structref", currently the best mutable heterogenous
datastructure supported by Numba that acts like a dataclass.
"""


@structref.register
class PhotonType(types.StructRef):
    def preprocess_fields(self, fields):
        return tuple((name, types.unliteral(typ)) for name, typ in fields)


# @dataclass
# class Photon:
# w: float = 1  # weight
# layer: int = 1  # index of layer where photon packet resides
# # scatters: int = 0  # no. of scattering events experienced
# s: float = 0  # dimensionless step size
# sleft: float = 0  # dimensionless step size
# dead: bool = False  # False if photon is propagating, True if terminated

# # cartesian coordinates of photon packet
# x: float = 0.0
# y: float = 0.0
# z: float = 0.0

# # direction cosines of photon propagation
# ux: float = 0.0
# uy: float = 0.0
# uz: float = 1.0

# @classmethod
# def init(cls, r_sp: float, layers: list[Layer]) -> Photon:
# w = 1 - r_sp
# if layers[1].mua == 0.0 and layers[1].mus == 0.0:
# layer = 2
# z = layers[2].z0
# return cls(w=w, layer=layer, z=z)

# return cls(w=w)


class Photon(structref.StructRefProxy):
    def __new__(cls, x, y, z, ux, uy, uz, w, layer, s, sleft, dead):
        # overriding __new__ allows us to use keyword arguments
        return structref.StructRefProxy.__new__(
            cls, x, y, z, ux, uy, uz, w, layer, s, sleft, dead
        )

    @property
    def w(self):
        return Photon_get_w(self)

    @w.setter
    def w(self, val):
        return Photon_set_w(self, val)

    @property
    def layer(self):
        return Photon_get_layer(self)

    @layer.setter
    def layer(self, val):
        return Photon_set_layer(self, val)

    @property
    def s(self):
        return Photon_get_s(self)

    @s.setter
    def s(self, val):
        return Photon_set_s(self, val)

    @property
    def sleft(self):
        return Photon_get_sleft(self)

    @sleft.setter
    def sleft(self, val):
        return Photon_set_sleft(self, val)

    @property
    def dead(self):
        return Photon_get_dead(self)

    @dead.setter
    def dead(self, val):
        return Photon_set_dead(self, val)

    @property
    def x(self):
        return Photon_get_x(self)

    @x.setter
    def x(self, val):
        return Photon_set_x(self, val)

    @property
    def y(self):
        return Photon_get_y(self)

    @y.setter
    def y(self, val):
        return Photon_set_y(self, val)

    @property
    def z(self):
        return Photon_get_z(self)

    @z.setter
    def z(self, val):
        return Photon_set_z(self, val)

    @property
    def ux(self):
        return Photon_get_ux(self)

    @ux.setter
    def ux(self, val):
        return Photon_set_ux(self, val)

    @property
    def uy(self):
        return Photon_get_uy(self)

    @uy.setter
    def uy(self, val):
        return Photon_set_uy(self, val)

    @property
    def uz(self):
        return Photon_get_uz(self)

    @uz.setter
    def uz(self, val):
        return Photon_set_uz(self, val)


@njit
def Photon_get_w(self):
    return self.w


@njit
def Photon_set_w(self, val):
    self.w = val


@njit
def Photon_get_layer(self):
    return self.layer


@njit
def Photon_set_layer(self, val):
    self.layer = val


@njit
def Photon_get_s(self):
    return self.s


@njit
def Photon_set_s(self, val):
    self.s = val


@njit
def Photon_get_sleft(self):
    return self.sleft


@njit
def Photon_set_sleft(self, val):
    self.sleft = val


@njit
def Photon_get_dead(self):
    return self.dead


@njit
def Photon_set_dead(self, val):
    self.dead = val


@njit
def Photon_get_x(self):
    return self.x


@njit
def Photon_set_x(self, val):
    self.x = val


@njit
def Photon_get_y(self):
    return self.y


@njit
def Photon_set_y(self, val):
    self.y = val


@njit
def Photon_get_z(self):
    return self.z


@njit
def Photon_set_z(self, val):
    self.z = val


@njit
def Photon_get_ux(self):
    return self.ux


@njit
def Photon_set_ux(self, val):
    self.ux = val


@njit
def Photon_get_uy(self):
    return self.uy


@njit
def Photon_set_uy(self, val):
    self.uy = val


@njit
def Photon_get_uz(self):
    return self.uz


@njit
def Photon_set_uz(self, val):
    self.uz = val


structref.define_proxy(
    Photon,
    PhotonType,
    ["w", "layer", "s", "sleft", "dead", "x", "y", "z", "ux", "uy", "uz"],
)
