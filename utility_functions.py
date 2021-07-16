"""
general usage functions to create visualizations and save outputs
"""

import pandas as pd
import geopandas as gpd

import matplotlib
import matplotlib.pyplot as plt


def plot_district_map(df, label_column, title=None, output_path=None):
    """
    visualize districts corresponding to a given label_column in geopandas dataframe
    If output_path not provided, will just display image. Otherwise save to location
    """
    df.plot(column=label_column, cmap="tab20", figsize=(12, 8))
    plt.axis("off")
    if title:
        plt.title(title)
    if output_path:
        plt.savefig(output_path)
    else:
        plt.show()


def add_other_population_attribute(graph, total_column="VAP",
                                   target_population_column="BVAP",
                                   other_population_name="nBVAP"):
    """calculates other population value by finding difference between total
     population and target population. For example, return graph with
     non-BVAP attribute using VAP & BVAP values
    """

    for node in graph.nodes():
        graph.nodes[node][other_population_name] = graph.nodes[node][total_column] - \
                                                   graph.nodes[node][target_population_column]

    return graph

def convert_attributes_to_int(graph, attribute_list):
    """
    helper function for casting attributes from string to int
    """
    for n in graph.nodes():
        for attribute in attribute_list:
            graph.nodes[n][attribute] = int(float(graph.nodes[n][attribute]))

    return graph

def convert_attributes_to_float(graph, attribute_list):
    """
     helper function for casting attributes from string to float
     """
    for n in graph.nodes():
        for attribute in attribute_list:
            graph.nodes[n][attribute] = float(graph.nodes[n][attribute])
    return graph
