from mcml.defs import InputParams, OutputParams, Layer
import numpy as np


def _sum_2d_Rd(out: OutputParams):
    """
    Get 1D array elements by summing the 2D array elements.
    """
    out.rd_r = out.rd_ra.sum(axis=1)
    out.rd_a = out.rd_ra.sum(axis=0)
    out.rd = out.rd_r.sum()


def _iz_to_layer(Iz: int, dz: float, layers: list[Layer]) -> int:
    """
    Return the index to the layer according to the index
    to the grid line system in z direction (Iz).

    Use the center of box.

    """
    i = 1
    num_layers = len(layers) - 2  # 2 ambient layers

    while (Iz + 0.5) * dz >= layers[i].z1 and i < num_layers:
        i += 1

    return i


def _sum_2d_A(inp: InputParams, layers: list[Layer], out: OutputParams):
    """
    Get 1D array elements by summing the 2D array elements.
    """
    out.a_z = out.a_rz.sum(axis=0)

    out.a_l = np.zeros(inp.nz)
    for iz in range(0, inp.nz):
        out.a_l[_iz_to_layer(iz, inp.dz, layers)] += out.a_z[iz]

    out.a = out.a_z.sum()


def _sum_2d_Tt(out: OutputParams):
    """
    Get 1D array elements by summing the 2D array elements.
    """
    out.tt_r = out.tt_ra.sum(axis=1)
    out.tt_a = out.tt_ra.sum(axis=0)
    out.tt = out.tt_r.sum()


def _scale_rd_tt(inp: InputParams, out: OutputParams):
    """
    Scale Rd and Tt properly.

    "a" stands for angle alpha.

    Scale Rd(r,a) and Tt(r,a) by
    (area perpendicular to photon direction)x(solid angle)x(No. of photons).
    or [2*PI*r*dr*cos(a)]x[2*PI*sin(a)*da]x[No. of photons]

    Scale Rd(r) and Tt(r) by (area on the surface)x(No. of photons).
    Scale Rd(a) and Tt(a) by (solid angle)x(No. of photons).
    """
    nr = inp.nr
    na = inp.na
    dr = inp.dr
    da = inp.da
    n_photons = inp.num_photons

    scale1 = 4.0 * np.pi * np.pi * dr * dr * np.sin(da / 2) * n_photons
    for ir in range(nr):
        for ia in range(na):
            scale2 = 1.0 / ((ir + 0.5) * np.sin(2.0 * (ia + 0.5) * da) * scale1)
            out.rd_ra[ir][ia] *= scale2
            out.tt_ra[ir][ia] *= scale2

    scale1 = 2.0 * np.pi * dr * dr * n_photons
    out.rd_r = np.zeros(nr)
    out.tt_r = np.zeros(nr)
    for ir in range(nr):
        scale2 = 1.0 / ((ir + 0.5) * scale1)
        out.rd_r[ir] *= scale2
        out.tt_r[ir] *= scale2

    scale1 = 2.0 * np.pi * da * n_photons
    out.rd_a = np.zeros(na)
    out.tt_a = np.zeros(na)
    for ia in range(na):
        scale2 = 1.0 / (np.sin((ia + 0.5) * da) * scale1)
        out.rd_a[ia] *= scale2
        out.tt_a[ia] *= scale2

    scale2 = 1.0 / float(inp.num_photons)
    out.rd *= scale2
    out.tt *= scale2


def _scale_a(inp: InputParams, out: OutputParams):
    """
    Scale absorption arrays
    """
    nr = inp.nr
    nz = inp.nz
    dr = inp.dr
    dz = inp.dz
    n_photons = inp.num_photons

    scale1 = 2.0 * np.pi * dr * dr * dz * n_photons
    # volume is 2*pi*(ir+0.5)*dr*dr*dz
    # ir+0.5 to be added.
    for iz in range(nz):
        for ir in range(nr):
            out.a_rz[ir][iz] /= (ir + 0.5) * scale1

    #  Scale A_z.
    scale1 = 1.0 / (dz * n_photons)
    out.a_z = np.zeros(nz)
    for iz in range(nz):
        out.a_z[iz] *= scale1

    # Scale A_l. Avoid int/int.
    scale1 = 1.0 / float(n_photons)
    out.a_l = np.zeros(inp.num_layers + 2)
    for il in range(inp.num_layers + 2):
        out.a_l[il] *= scale1

    out.a *= scale1


def sum_scale_result(inp: InputParams, layers: list[Layer], out: OutputParams):
    """
    Sum and scale results of the current run
    """
    _sum_2d_Rd(out)
    _sum_2d_A(inp, layers, out)
    _sum_2d_Tt(out)

    _scale_rd_tt(inp, out)
    _scale_a(inp, out)
    return out
