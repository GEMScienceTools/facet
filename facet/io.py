import os
import logging

from tqdm import tqdm

import vtk
import numpy as np
import pandas as pd

from scipy.spatial import ConvexHull
from vtk.util.numpy_support import vtk_to_numpy


def get_output_from_vtk(vtk_filename):

    if os.path.exists(vtk_filename):
        reader = vtk.vtkXMLPUnstructuredGridReader()
        reader.SetFileName(vtk_filename)
        reader.Update()
        output = reader.GetOutput()

        return output

    else:
        raise ValueError("VTK filename does not exist.")


def get_field_data(output):
    field_names = []
    n_fields = output.GetPointData().GetNumberOfArrays()

    for i in range(n_fields):
        field_names.append(output.GetPointData().GetArrayName(i))

    field_data = {
        fn: vtk_to_numpy(output.GetPointData().GetArray(i))
        for i, fn in enumerate(field_names)
    }

    return field_data


def get_element(cell, field_data, downsample_coords=False):
    el = {}
    el["pts"] = vtk_to_numpy(cell.GetPoints().GetData())
    el["pts_idx"] = [
        cell.GetPointIds().GetId(i)
        for i in range(cell.GetPointIds().GetNumberOfIds())
    ]

    # which do I average over and which do I sum over?
    el["properties"] = {
        f: np.mean(vals[el["pts_idx"]]) for f, vals in field_data.items()
    }

    if downsample_coords:
        hull = ConvexHull(el["pts"])
        el["pts"] = el["pts"][hull.vertices]
        el["pts_idx"] = np.array(el["pts_idx"])[hull.vertices]

    return el


def get_cell(cell_id, output, field_data, downsample_coords=False):
    return get_element(
        output.GetCell(cell_id),
        field_data=field_data,
        downsample_coords=downsample_coords,
    )


def make_cell_dataframe(
    output,
    keep_cols=(
        "viscosity",
        "elastic_shear_modulus",
        "strain_rate",
        "plastic_yielding",
    ),
):
    n_cells = output.GetNumberOfCells()

    keep_kols = ["xc", "yc", "zc", "volume"] + list(keep_cols)

    keep_data = {col: np.empty(n_cells) for col in keep_kols}
    keep_data["pts"] = np.empty(n_cells, dtype="object")

    field_data = get_field_data(output)

    def fill_cols(i):
        cell = get_cell(i, output, field_data, downsample_coords=True)
        xc, yc, zc = cell["pts"].mean(axis=0)
        volume = np.prod(cell["pts"].max(axis=0) - cell["pts"].min(axis=0))
        cell["properties"]["xc"] = xc
        cell["properties"]["yc"] = yc
        cell["properties"]["zc"] = zc
        cell["properties"]["volume"] = volume

        for col in keep_kols:
            keep_data[col][i] = cell["properties"][col]

        keep_data["pts"][i] = cell["pts"]

    logging.info("building field arrays")
    [fill_cols(i) for i in tqdm(range(n_cells))]
    logging.info("done")

    cell_df = pd.DataFrame(keep_data)

    return cell_df


def get_strain_rates_from_output(output, field_data):
    n_cells = output.GetNumberOfCells()
    strain_rates = [
        get_element(output.GetCell(i), field_data)["properties"]["strain_rate"]
        for i in range(n_cells)
    ]

    return strain_rates
