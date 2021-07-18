"""
general usage functions to create visualizations and save outputs
"""
import os

import json
import csv

import pandas as pd
import geopandas as gpd
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
from functools import reduce

from gerrychain import Election, Partition
from gerrychain.metrics import efficiency_gap, mean_median, polsby_popper, wasted_votes


def plot_district_map(df, assignment_dict, title=None, output_path=None):
    """
    visualize districts corresponding to a given assignment_dict
    mapping index in geopandas dataframe to desired districts per node
    If output_path not provided, will just display image. Otherwise save to location
    """
    df['district_assignment'] = df.index.map(assignment_dict)
    df.plot(column='district_assignment', cmap="tab20", figsize=(12, 8))
    plt.axis("off")
    if title:
        plt.title(title)
    if output_path:
        plt.savefig(output_path)
    else:
        plt.show()


def add_other_population_attribute(graph, total_column_list=["VAP"],
                                   target_population_column_list=["BVAP"],
                                   other_population_name="nBVAP"):
    """
     calculates other population value by finding difference between total
     population and target population. For example, return graph with
     non-BVAP attribute using VAP & BVAP values

    :param graph: networkx graph object
    :param total_column_list: list of all attributes to consider in the 'total population' (e.g. VAP)
    :param target_population_column_list: list of all attributes to consider in the 'target population' (e.g. BVAP)
    :param other_population_name: string - identifer for other population
    :return:
    """

    for node in graph.nodes():
        total_sum = reduce(lambda a, b: a+b, [graph.nodes[node][att]
                                              for att in total_column_list])
        target_sum = reduce(lambda a, b: a+b, [graph.nodes[node][att]
                                               for att in target_population_column_list])
        graph.nodes[node][other_population_name] = total_sum - target_sum

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


def export_all_metrics_per_chain(chain, output_path,
                                 buffer_length=2000,
                                 include_election=True,
                                 exclude_metrics=['cut_edges', 'boundary_nodes',
                                                  'cut_edges_by_part',
                                                  'area', 'perimeter',
                                                  'interior_boundaries', 'exterior_boundaries'
                                                  ]
                                 ):

    """
    save all metrics for every k steps (k=buffer length) of chain in separate csv file
    Additionally, saves partition assignment at kth step of chain as json

    :param chain: GerryChain chain generator object
    :param output_path: folder to save results in (separate dir per chain)
    :param buffer_length: frequency at which data is written to file
    :param include_election: boolean - specifies whether election related metrics should be saved
    :param exclude_metrics: list - metrics to exclude from output files. All metrics not at district level
                            should be excluded.
    :return:
    """

    os.makedirs(output_path, exist_ok=True)

    all_res = []
    election_res = []
    for i, part in enumerate(chain):
        results = export_all_metrics_per_partition(part, include_election, exclude_metrics)
        all_res.append(results[0].values)

        if include_election:
            election_res.append(results[1].values)

        # save results every i = buffer_length steps
        if (i+1) % buffer_length == 0:
            print(i+1)

            # save non-election related metric results
            for n, metric_name in enumerate(results[0].columns):
                with open(os.path.join(output_path, f"{metric_name}_{str(i+1)}.csv"), 'w') as f:
                    writer = csv.writer(f, lineterminator="\n")
                    writer.writerows(np.array(all_res)[:, :, n])

            all_res = []

            if include_election:
                # save election related metric results
                for n, metric_name in enumerate(results[1].columns):
                    if metric_name == 'percent': #different format - save per election in sep file
                        for row, election_name in enumerate(results[1].index):
                            with open(os.path.join(output_path, f"{election_name}_{metric_name}_{str(i+1)}.csv"), 'w') as f:
                                writer = csv.writer(f, lineterminator='\n')
                                writer.writerows(np.array(election_res)[:, row, n])
                    else: # all other election metrics saved in single file
                        with open(os.path.join(output_path, f"{metric_name}_{str(i+1)}.csv"), 'w') as f:
                            writer = csv.writer(f, lineterminator='\n')
                            writer.writerows(np.array(election_res)[:, :, n])

                election_res = []

            # save assignment for every i steps
            with open(os.path.join(output_path, f'assignment_{str(i+1)}.json'), "w") as f:
                json.dump(dict(part.assignment), f)




def export_all_metrics_per_partition(partition, include_election=True,
                                     exclude_metrics=['cut_edges']):
    """
    returns tuple of dataframes with common metrics calculated at district level
    (split by nonelection & election related)

    :param partition:
    :param include_election: boolean
    :param exclude_metrics: metrics tracked in updater to exclude from export
    :return: returns tuple of dataframe - (nonelection related metrics, election related metrics)
    """
    metric_results = {}
    for name, func in partition.updaters.items():
        if type(func) == Election or name in exclude_metrics:
            continue
        metric_results[name] = partition[name]

    metric_results = pd.DataFrame.from_dict(metric_results)

    if include_election:
        election_results=export_election_metrics_per_partition(partition)
        return (metric_results, election_results)

    return (metric_results,)


def export_election_metrics_per_partition(partition):
    """
    returns dataframe with following metrics for all elections tracked in updater
    using the given partition

    * percent won by first party
    * mean_median score
    * efficiency gap
    * seats won

    :param partition:
    :return: pd dataframe election_results
    """
    elections_info = {}

    # identify elections related updaters being tracked per partition
    for name, func in partition.updaters.items():
        if type(func) == Election:
            elections_info[name] = func.parties

    election_results = {name: {} for name in elections_info.keys()}

    # store stats for each election
    for election_name, parties in elections_info.items():
        election_results[election_name]['percent'] = partition[
            election_name].percents(parties[0])
        election_results[election_name]['efficiency_gap'] = efficiency_gap(
            partition[election_name])
        election_results[election_name]['mean_median'] = mean_median(
            partition[election_name])
        election_results[election_name]['wins'] = partition[
            election_name].wins(parties[0])

    return pd.DataFrame.from_dict(election_results, orient='index')