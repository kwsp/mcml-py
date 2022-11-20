import numpy as np

from defs import Photon, Layer, InputParams, OutputParams


def r_specular(layers: list[Layer]) -> float:
    """
    Compute the specular reflection

    If the first layer is a turbid medium, use the Fresnel reflection
    from the boundary of the first layer as the specular reflectance.

    If the first layer is glass, multiple reflections in the first layer
    is considered to get the specular reflectance.
    """

    # Use equation (3.8)
    r1 = ((layers[0].n - layers[1].n) / (layers[0].n + layers[1].n)) ** 2

    if layers[1].mua == 0.0 and layers[1].mus == 0.0:
        # First layer is a clear medium (glass layer)
        r2 = ((layers[1].n - layers[2].n) / (layers[1].n + layers[2].n)) ** 2

        r1 += (1 - r1) ** 2 * r2 / (1 - r1 * r2)
    return r1
