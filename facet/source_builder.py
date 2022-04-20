from difflib import diff_bytes
import logging
import warnings
from copy import deepcopy
from collections import namedtuple

import numpy as np
import pandas as pd

from tqdm import tqdm

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


import numba as nb
from numba import jit, njit, prange


directions_to_azimuths = {
    "N": 0.0,
    "NNE": 22.5,
    "NE": 45.0,
    "ENE": 67.5,
    "E": 90.0,
    "ESE": 112.5,
    "S": 180.0,
    "W": 270.0,
    "NW": 315.0,
    "SE": 135.0,
    "SW": 225.0,
}


def get_elastic_elements(cell_df, yield_threshold=0.0):
    return cell_df.loc[(cell_df["plastic_yielding"] > yield_threshold)]


def get_high_strain_indices(strain_rates, strain_threshold, strain_max=None):
    if strain_max is None:
        high_strain_bool = strain_rates >= strain_threshold
    else:
        high_strain_bool = (strain_rates >= strain_threshold) * (
            strain_rates < strain_max
        )

    high_strain_idxs = np.arange(len(strain_rates))[high_strain_bool]

    return high_strain_bool, high_strain_idxs


def get_high_strain_df(cell_df, strain_threshold, strain_max=None):
    high_strain_bool, high_strain_idx = get_high_strain_indices(
        cell_df["strain_rate"], strain_threshold, strain_max=strain_max
    )

    return cell_df.loc[high_strain_bool]


def get_fault_df(
    cell_df, strain_threshold, strain_max=None, yield_threshold=0.0
):
    elastic_cells = get_elastic_elements(
        cell_df, yield_threshold=yield_threshold
    )

    fault_df = get_high_strain_df(
        elastic_cells, strain_threshold=strain_threshold, strain_max=strain_max
    )

    return fault_df


@jit
def check_array_equal(arr1, arr2):
    equal = True
    for i, val1 in enumerate(arr1):
        if val1 != arr2[i]:
            equal = False
    return equal


@jit
def are_we_touching(arr1, arr2):

    touching = False

    for row1 in arr1:
        for row2 in arr2:
            # if np.array_equal(row1, row2):
            # if np.all(row1 == row2):
            if check_array_equal(row1, row2):
                touching = True
                break
        else:
            continue
    return touching


@njit(parallel=True)
def calc_touch_matrix(cell_pts, adj_matrix, n):
    for i in prange(n):
        cell_pts_i = cell_pts[i]
        for j in range(i + 1, n):
            if are_we_touching(cell_pts_i, cell_pts[j]):
                adj_matrix[i, j] = 1
                adj_matrix[j, i] = 1

    return adj_matrix


def get_touching_matrix(cell_pt_arr):
    n = len(cell_pt_arr)

    adj_matrix = np.zeros((n, n), dtype=np.int8)

    adj_matrix = calc_touch_matrix(cell_pt_arr, adj_matrix, n)

    return adj_matrix


def make_typed_pt_list(pt_series):
    return nb.typed.List(pt_series.values.tolist())


def precompile_adjacence_matrix_functions():
    pts_0 = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    pts_1 = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 2.0]])
    pts_2 = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 3.0]])

    pts = nb.typed.List([pts_0, pts_1, pts_2])

    return get_touching_matrix(pts)


precompile_adjacence_matrix_functions()


def get_fault_groups(fault_df, del_pts=True):
    logging.info("calculating fault cell adjacence matrix")
    nb_pt_list = make_typed_pt_list(fault_df["pts"])
    fault_adj_matrix = get_touching_matrix(nb_pt_list)
    fault_adj_matrix = csr_matrix(fault_adj_matrix)
    logging.info("...done")

    n_components, conn_groups = connected_components(
        fault_adj_matrix, directed=False
    )

    fault_df["fault_group"] = conn_groups

    if del_pts:
        del fault_df["pts"]

    return  # fault_df


def fit_plane_to_pts(xs, ys, zs):
    pts = np.vstack((xs, ys, zs)).T

    centroid = np.mean(pts, axis=0)

    CX = pts - centroid
    U, S, V = np.linalg.svd(CX)
    N = V[-1]

    if N[2] < 0:
        N *= -1.0

    x0, y0, z0 = centroid
    a, b, c = N
    d = -(a * x0 + b * y0 + c * z0)

    return np.array([a, b, c]), centroid


def project_pt_to_plane(pt, plane_normal, plane_pt):
    return pt - np.dot(pt - plane_pt, plane_normal) * plane_normal


def get_strike_dip_from_normal(normal):
    strike = angle_to_az(np.arctan2(normal[1], normal[0])) - 90.0
    while strike > 360.0:
        strike -= 360.0
    while strike < 0:
        strike += 360.0

    dip = np.degrees(np.arccos(normal.dot([0.0, 0.0, 1.0])))

    return strike, dip


def get_strike_vector(strike):
    # modified from halfspace/projections.py , (c) R. Styron
    strike_rad = np.radians(strike)
    return np.array([np.sin(strike_rad), np.cos(strike_rad), 0.0])


def get_dip_vector(strike, dip):
    # modified from halfspace/projections.py , (c) R. Styron
    norm = get_normal_from_strike_dip(strike, dip)
    strike_vec = get_strike_vector(strike)
    return np.cross(norm, strike_vec)


def dip_shear_stress_from_tensor(strike, dip, T):
    # modified from halfspace/projections.py , (c) R. Styron
    N = get_normal_from_strike_dip(strike, dip)
    D = get_dip_vector(strike, dip)

    return D @ T @ N.T


def strike_shear_stress_from_tensor(strike, dip, T):
    # modified from halfspace/projections.py , (c) R. Styron
    N = get_normal_from_strike_dip(strike, dip)
    S = get_strike_vector(strike)

    return S @ T @ N.T


def get_normal_from_strike_dip(strike, dip):
    # modified from halfspace/projections.py , (c) R. Styron
    strike_rad = np.radians(strike)
    dip_rad = np.radians(dip)

    nE = np.sin(dip_rad) * np.cos(strike_rad)
    nN = -np.sin(dip_rad) * np.sin(strike_rad)
    nD = np.cos(dip_rad)

    return np.array([nE, nN, nD])


def get_rake_from_shear_components(strike_shear, dip_shear):
    # modified from halfspace/projections.py , (c) R. Styron
    rake = np.degrees(np.arctan2(dip_shear, -strike_shear))
    return rake


def get_fault_strike_dip(fault):
    normal, centroid = fit_plane_to_pts(fault["xc"], fault["yc"], fault["zc"])

    strike, dip = get_strike_dip_from_normal(normal)

    return strike, dip, normal, centroid


def fake_proj_m_to_deg(
    xs, ys, zs=None, x0=0.0, y0=0.0, z0=0.0, lon0=0.0, lat0=0.0
):

    deg_to_m = 111_319.5

    x_norm = xs - x0
    y_norm = ys - y0

    del_lons = x_norm / (deg_to_m * np.cos(lat0))
    del_lats = y_norm / deg_to_m

    lons = del_lons + lon0
    lats = del_lats + lat0

    if zs is None:
        return_vals = (lons, lats)

    else:
        depths_km = -(zs - z0)

        return_vals = (lons, lats, depths_km)

    return return_vals


def rotate_pts(xs, ys, normal):
    rot_angle = np.arctan2(normal[0], normal[1])

    xs = np.asarray(xs)
    ys = np.asarray(ys)

    xp = np.cos(rot_angle) * xs - np.sin(rot_angle) * ys
    yp = np.sin(rot_angle) * xs + np.cos(rot_angle) * ys

    return xp, yp


def get_upper_fault_trace(fault, normal, z_threshold=1.0):
    upper_pts = fault.loc[((fault.zc.max() - fault.zc) < z_threshold * 1000.0)]

    xp, yp = rotate_pts(upper_pts.xc, upper_pts.yc, normal)

    fit = np.polyfit(xp, yp, 1)
    slope, intercept = fit

    fit_poly = np.poly1d(fit)

    xp_start = np.min(xp)
    xp_end = np.max(xp)

    xp_trace = [xp_start, xp_end]
    yp_trace = fit_poly(xp_trace)

    x_trace, y_trace = rotate_pts(xp_trace, yp_trace, -normal)

    return [[x_trace[0], y_trace[0]], [x_trace[1], y_trace[1]]]


def get_fault_moment(fault):
    cell_moments = (
        fault.volume * fault.elastic_shear_modulus * fault.strain_rate
    )
    return cell_moments.sum()


def haversine_distance(
    lon_0: float = None,
    lat_0: float = None,
    lon_1: float = None,
    lat_1: float = None,
    r: float = 6.371e6,
) -> float:

    """
    Calculates the great circle distance between two points in lon, lat
    using the haversine formula.
    """

    r_lon_0, r_lon_1, r_lat_0, r_lat_1 = np.radians(
        (lon_0, lon_1, lat_0, lat_1)
    )

    term_1 = np.sin((r_lon_1 - r_lon_0) / 2.0) ** 2
    term_2 = np.cos(r_lat_0) * np.cos(r_lat_1)
    term_3 = np.sin((r_lat_1 - r_lat_0) / 2.0) ** 2

    return 2 * r * np.arcsin(np.sqrt(term_1 + term_2 * term_3))


def fault_trace_length(fault_trace):
    n_segs = len(fault_trace) - 1
    total_length = 0.0
    for i in range(n_segs):
        total_length += haversine_distance(
            fault_trace[i][0],
            fault_trace[i][1],
            fault_trace[i + 1][0],
            fault_trace[i + 1][1],
        )

    return total_length


def get_weighted_shear_modulus(fault):

    return (
        fault.elastic_shear_modulus * fault.volume
    ).sum() / fault.elastic_shear_modulus.sum()


def get_fault_info(fault, x0=0.0, y0=0.0, z0=0.0, lon0=0.0, lat0=0.0):
    fault_info = {}

    (
        fault_info["strike"],
        fault_info["dip"],
        fault_info["normal"],
        fault_info["centroid"],
    ) = get_fault_strike_dip(fault)

    fault_info["depth"] = (fault["zc"].max() - fault["zc"].min()) / 1000.0

    fault_info["trace_coords"] = [
        fake_proj_m_to_deg(tr[0], tr[1])
        for tr in get_upper_fault_trace(fault, fault_info["normal"])
    ]

    fault_info["moment"] = get_fault_moment(fault)

    fault_info["area"] = fault_trace_length(fault_info["trace_coords"]) * (
        fault_info["depth"] / np.sin(np.radians(fault_info["dip"]))
    )

    fault_info["slip_rate_from_moment"] = fault_info["moment"] / (
        fault_info["area"] * get_weighted_shear_modulus(fault) * 1000.0
    )

    fault_info["dip_dir"] = get_dip_dir(
        angle_to_az(
            np.arctan2(fault_info["normal"][1], fault_info["normal"][0])
        )
    )

    return fault_info


def angle_to_az(angle):
    az = -(np.degrees(angle) - 90.0)
    while az < 0.0:
        az += 360.0
    while az >= 360.0:
        az -= 360.0

    return az


def az_to_angle(az):
    return np.radians(90.0 - az)


def az_difference(az1, az2, return_abs=True):
    diff = az2 - az1

    while diff < -180:
        diff += 360.0
    while diff > 180.0:
        diff -= 360.0

    if return_abs:
        diff = np.abs(diff)

    return diff


def get_dip_dir(dip_az):

    # az_dir_inv = {v: k for k, v in directions_to_azimuths.items()}

    az_dists = {
        k: az_difference(dip_az, v) for k, v in directions_to_azimuths.items()
    }

    print(az_dists)

    min_dir = "N"
    min_az_dist = 360.0
    for dir, az_dist in az_dists.items():
        if az_dist < min_az_dist:
            min_az_dist = az_dist
            min_dir = dir

    return min_dir


def fault_info_to_gj(fault_info):

    fgj = {
        "type": "Feature",
        "geometry": {
            "type": "LineString",
            "coordinates": fault_info["trace_coords"],
        },
        "properties": {
            "dip": fault_info["dip"],
            "depth": fault_info["depth"],
            "slip_rate": fault_info["slip_rate_from_moment"],
        },
    }

    return fgj


def sorted_eigens(T):

    # modified from halfspace.py (c) R. Styron

    eig_vals, eig_vecs = np.linalg.eigh(T)
    idx = eig_vals.argsort()[::-1]

    eig_vals = eig_vals[idx]
    eig_vecs = np.array(eig_vecs[:, idx])

    return eig_vals, eig_vecs


def get_optimal_fault_angle_from_s1(friction_angle):
    return (np.pi / 2.0 - np.radians(friction_angle)) / 2.0


def make_stress_tensor(xx, yy, zz, xy, xz, yz):
    return np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]])


def get_fault_rake_from_tensor(strike, dip, T):
    strike_shear = strike_shear_stress_from_tensor(strike, dip, T)
    dip_shear = dip_shear_stress_from_tensor(strike, dip, T)
    rake = get_rake_from_shear_components(strike_shear, dip_shear)

    return rake


def get_optimal_fault_plane(T, friction_angle=30):
    # modified from halfspace.py (c) R. Styron
    beta = get_optimal_fault_angle_from_s1(friction_angle)

    vals, vecs = sorted_eigens(T)

    opt_plane_normal_vec_rot = np.array([np.cos(beta), 0.0, np.sin(beta)])

    opt_plane_norm = vecs @ opt_plane_normal_vec_rot

    strike, dip = get_strike_dip_from_normal(opt_plane_norm)

    rake = get_fault_rake_from_tensor(strike, dip, T)

    return strike, dip, rake


def get_fault_plane_from_pt_stress(row):
    T = make_stress_tensor(
        row.stress_xx,
        row.stress_yy,
        row.stress_zz,
        row.stress_xy,
        row.stress_xz,
        row.stress_yz,
    )

    strike, dip = get_optimal_fault_plane(T, row.current_friction_angles)
    rake = get_fault_rake_from_tensor(strike, dip, T)
    return strike, dip, rake


def make_stress_tensor_from_row(row):
    stress_tensor = make_stress_tensor(
        row.stress_xx,
        row.stress_yy,
        row.stress_zz,
        row.stress_xy,
        row.stress_xz,
        row.stress_yz,
    )

    return stress_tensor
