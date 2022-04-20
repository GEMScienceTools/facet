import sys
import logging
from importlib import reload

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

import numpy as np
import pandas as pd

import facet

reload(facet)

vtk_file = "/home/itchy/research/gem/aspect/gems_aspect_data_example/aspect_model/output_aspect_model/solution/solution-00008.pvtu"

# print("getting output")
# vtk_output = facet.io.get_output_from_vtk(vtk_file)
#
## field_data = facet.io.get_field_data(vtk_output)
#
# print("getting field df")
# field_df = facet.io.make_cell_dataframe(vtk_output)
#
# strain_cutoff = 0.1e-13

# fault_df = facet.source_builder.get_fault_df(field_df, strain_cutoff)

# facet.source_builder.get_fault_groups(fault_df)

fault_df = pd.read_csv("~/Desktop/fault_df.csv", index_col=0)

fault_groups = fault_df.groupby(["fault_group"])

fault_dicts = {}
for fault, group in fault_groups:
    fault_dicts[fault] = facet.source_builder.get_fault_info(group)
    print(fault_dicts[fault])

# f0 = fault_groups.get_group(0)
# ff = facet.source_builder.get_fault_info(f0)

# print(ff)


import matplotlib.pyplot as plt


def plot_fault(fault):
    plt.scatter(fault.xc, fault.yc, c=fault.zc)


def plot_fault_best_fit(fault):
    ff = fault_groups.get_group(fault)
    plot_fault(ff)
    coords = facet.source_builder.get_upper_fault_trace(
        ff, fault_dicts[fault]["normal"]
    )
    plt.plot([coords[0][0], coords[1][0]], [coords[0][1], coords[1][1]])
    plt.axis("equal")

    return


for fault, group in fault_groups:
    plot_fault_best_fit(fault)


plt.show()
