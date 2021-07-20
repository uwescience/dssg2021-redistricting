#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Virginia Case Study

"""

import os
import sys
import re

import csv
import json
import random
import math
import numpy as np

from functools import partial
import pandas as pd
import geopandas as gpd

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from gerrychain import (
    Election,
    Graph,
    MarkovChain,
    Partition,
    accept, #always_accept, acceptance functions
    constraints,
    updaters,
)
from gerrychain.metrics import efficiency_gap, mean_median, polsby_popper, wasted_votes
from gerrychain.proposals import recom, propose_random_flip
from gerrychain.updaters import cut_edges
from gerrychain.tree import recursive_tree_part, bipartition_tree_random

sys.path.insert(0, os.getenv("REDISTRICTING_HOME"))
import utility_functions as uf

plt.style.use('seaborn-whitegrid')

#--- IMPORT DATA
try:
    os.chdir(os.path.join(os.getenv("REDISTRICTING_HOME", default=""),
                          "Virginia"))
except OSError:
    os.mkdir(os.path.join(os.getenv("REDISTRICTING_HOME", default=""),
                          "Virginia"))
    os.chdir(os.path.join(os.getenv("REDISTRICTING_HOME", default=""),
                          "Virginia"))

graph = Graph.from_json("Data/VA_Chain.json")
df = gpd.read_file("Data/VA_precincts.shp")

#--- CREATE SHORTCUTS

state_abbr="VA"
housen="CON"
num_districts=11
pop_col="TOTPOP"
num_elections= 3

#Make an output directory to dump files in
newdir = "./Outputs/"+state_abbr+housen+"_Precincts/"
print(newdir)
os.makedirs(os.path.dirname(newdir), exist_ok=True)

# Visualize districts for existing plans
uf.plot_district_map(df, df['CD_12'].to_dict(), "2012 Congressional District Map")
uf.plot_district_map(df, df['CD_16'].to_dict(), "2016 Congressional District Map")


#--- DATA CLEANING
graph = uf.convert_attributes_to_int(graph, ["G18DSEN", "G18RSEN", "G16DPRS", "G16RPRS"])

# calculate non-BVAP
graph = uf.add_other_population_attribute(graph)

#--- GENERIC UPDATERS

updater = {
    "population": updaters.Tally("TOTPOP", alias="population"), #can only take in partitions
    "cut_edges": cut_edges,
    #"PP":polsby_popper
            }

#--- ELECTION UPDATERS

#BVAP - Black Voting Age Population
#G18DSEN - 2018 Democratic senate candidate
#G18RSEN - 2018 Republican senate candidate
#G16DPRS: 2016 Democratic presidential candidate
#G16RPRS: 2016 Republican presidential candidate

election_names=[
    "BVAP", #BVAP, nBVAP
    "G16PRS", #G17DGOV, G17RGOV
    "G18SEN" #G18DSEN, G18RSEN
    ]

election_columns=[
    ["BVAP", "nBVAP"], #First is BVAP, Second is NOT BVAP
    ["G16DPRS", "G16RPRS"], #First is Democrats, Second is Republicans
    ["G18DSEN", "G18RSEN"] #First is Democrats, Second is Republicans
    ]

elections = [
    Election(
        election_names[i], #Name of election
        {"First": election_columns[i][0], "Second": election_columns[i][1]},
    )#Take two columns of the election_columns, using first and second aligned to those column assignments
    for i in range(num_elections)
]


election_updaters = {election.name: election for election in elections}
updater.update(election_updaters)


#--- STARTING PLAN (SEED PLAN)

totpop = df.TOTPOP.sum()
cddict = recursive_tree_part(graph, #graph object
                              range(num_districts), #How many districts
                              totpop/num_districts, #population target
                              "TOTPOP", #population column, variable name
                              .01, #epsilon value
                              1)

file_name = os.path.join(newdir, "initial_plan.png")
uf.plot_district_map(df, cddict, "Seed Plan: Recursive Partitioning Tree",
                  output_path=file_name)


# --- PARTITION

initial_partition = Partition(graph,
                              cddict, #initial plan (this is our recurisive_tree_part)
                              updater)

ideal_population = sum(initial_partition["population"].values()) / len(initial_partition)

with open(newdir+"init.json", 'w') as jf1:
        json.dump(cddict, jf1)

#--- LOOK AT STATS OF STARTING PLAN
stats_df = uf.export_election_metrics_per_partition(initial_partition)
stats_df.percent.apply(pd.Series).plot()
plt.show()

# --- PROPOSAL

proposal = partial(#All the functions inside gerrychain want to take partition, but recom wants more functions than that
                   #Partial takes main functions and prefill all the objects until it becomes a partition
    recom,
    pop_col="TOTPOP",
    pop_target=ideal_population,
    epsilon=0.05,
    node_repeats=1,
    method=bipartition_tree_random
)

#--- CREATE CONSTRAINTS

compactness_bound = constraints.UpperBound(
    lambda p: len(p["cut_edges"]), 1.5 * len(initial_partition["cut_edges"])
)

#--- ACCEPTANCE FUNCTIONS


#--- MCMC CHAINS

recom_chain = MarkovChain( #recom automatically does contiguity
    proposal=proposal,
    constraints=[
        constraints.within_percent_of_ideal_population(initial_partition, 0.05),
        compactness_bound,
    ],
    accept=accept.always_accept, #put acceptance function later?
    initial_state=initial_partition,
    total_steps=2000
)

flip_chain = MarkovChain(
    proposal=propose_random_flip,
    constraints=[
        constraints.within_percent_of_ideal_population(initial_partition, 0.05),
        constraints.single_flip_contiguous,
        compactness_bound
    ],
    accept=accept.always_accept,
    initial_state=initial_partition,
    total_steps=20000
)



# #--- RUN RECOMBINATION & FLIP BOUNDARY PROPOSALS & SAVE RESULTS


uf.export_all_metrics_per_chain(recom_chain,
                                output_path=os.path.join(newdir, 'recom_chain'),
                                buffer_length=200)


uf.export_all_metrics_per_chain(flip_chain,
                                output_path=os.path.join(newdir, 'flip_chain'),
                                buffer_length=2000
                                )

for file in os.listdir(os.path.join(newdir, 'recom_chain')):
    m = re.search('assignment_(\d+).json', file)
    if m:
        assignment = json.load(open(os.path.join(newdir,
                                                 'recom_chain/' + file)))
        assignment = {int(k): v for k, v in assignment.items()}
        title = f'Recom Proposal: {m.groups()[0]} Steps'
        uf.plot_district_map(df, assignment, title=title,
                             output_path=os.path.join(newdir, f'recom_chain/plot_{m.groups()[0]}.png')
                             )

for file in os.listdir(os.path.join(newdir, 'flip_chain')):
    m = re.search('assignment_(\d+).json', file)
    if m:
        assignment = json.load(open(os.path.join(newdir,
                                                 'flip_chain/' + file)))
        assignment = {int(k): v for k, v in assignment.items()}
        title = f'Flip Proposal: {m.groups()[0]} Steps'
        uf.plot_district_map(df, assignment, title=title,
                             output_path=os.path.join(newdir, f'flip_chain/plot_{m.groups()[0]}.png')
                             )



#--- BUILD VISUALIZATIONS

# sns.set_style('darkgrid')
sns.set_style("darkgrid", {"axes.facecolor": ".97"})

datadir = "./Outputs/VACON_Precincts/"

#Build partitions for 2012/2016 maps to calculate comparison metrics

partition_2012 = Partition(graph,
                           df["CD_12"],
                           updater)

partition_2016 = Partition(graph,
                           df["CD_16"],
                           updater)

#--- VISUALIZATION FUNCTIONS

def comparison_hist(df_proposal_metric, title, election, gc_metric):
    plt.hist(df_proposal_metric)
    plt.title(title)
    plt.vlines(x=sum(df_proposal_metric)/len(df_proposal_metric),
               ymin=0,
               ymax=(np.histogram(df_proposal_metric)[0]).max(),
               colors="blue",
               linestyles="solid",
               label="Ensemble Mean")
    plt.vlines(x=gc_metric(partition_2012[election]),
               ymin=0,
               ymax=(np.histogram(df_proposal_metric)[0]).max(),
               colors="red",
               linestyles="dashed",
               label="2012 Plan")
    plt.vlines(x=gc_metric(partition_2016[election]),
               ymin=0,
               ymax=(np.histogram(df_proposal_metric)[0]).max(),
               colors="orange",
               linestyles="dashed",
               label="2016 Plan")
    plt.legend(bbox_to_anchor=(.8, 1),
               loc='upper left', borderaxespad=0.)
    plt.show()

def comparison_plot(df_proposal_metric, title, election, gc_metric):
    plt.plot(df_proposal_metric)
    plt.title(title)
    plt.hlines(y=sum(df_proposal_metric)/len(df_proposal_metric),
           xmin=0,
           xmax=len(df_proposal_metric),
           colors="blue",
           linestyles="solid",
           label="Ensemble Mean")
    plt.hlines(y=gc_metric(partition_2012[election]),
           xmin=0,
           xmax=len(df_proposal_metric),
           colors="red",
           linestyles="dashed",
           label="2012 Plan")
    plt.hlines(y=gc_metric(partition_2016[election]),
           xmin=0,
           xmax=len(df_proposal_metric),
           colors="orange",
           linestyles="dashed",
           label="2016 Plan")
    plt.legend(bbox_to_anchor=(.8, 1),
           loc='upper left', borderaxespad=0.)
    plt.show()

#--- RECOM PROPOSAL VISUALIZATION

recom_hmss = []
for file in os.listdir(os.path.join(newdir, 'recom_chain')):
    m = re.search('wins_(\d+).csv', file)
    if m:
        recom_hmss.append(pd.read_csv(os.path.join(newdir, 'recom_chain/' + file), header=None))
df_recom_seats  = pd.concat(recom_hmss)
df_recom_seats.columns = election_names
df_recom_seats = df_recom_seats.reset_index(drop=False)

recom_mms = []
for file in os.listdir(os.path.join(newdir, 'recom_chain')):
    m = re.search('mean_median_(\d+).csv', file)
    if m:
        recom_mms.append(pd.read_csv(os.path.join(newdir, 'recom_chain/' + file), header=None))
df_recom_mms  = pd.concat(recom_mms)
df_recom_mms.columns = election_names
df_recom_mms = df_recom_mms.reset_index(drop=False)

recom_egs = []
for file in os.listdir(os.path.join(newdir, 'recom_chain')):
    m = re.search('efficiency_gap_(\d+).csv', file)
    if m:
        recom_egs.append(pd.read_csv(os.path.join(newdir, 'recom_chain/' + file), header=None))
df_recom_egs = pd.concat(recom_egs)
df_recom_egs.columns = election_names
df_recom_egs = df_recom_egs.reset_index(drop=False)

#Mean-Median

comparison_hist(df_recom_mms["G16PRS"],
                "ReCom: Mean-Median",
                "G16PRS",
                mean_median)

comparison_plot(df_recom_mms["G16PRS"],
                "ReCom: Mean-Median",
                "G16PRS",
                mean_median)

#Efficiency Gap

comparison_hist(df_recom_egs["G16PRS"],
                "ReCom: Efficiency Gap",
                "G16PRS",
                efficiency_gap)

comparison_plot(df_recom_egs["G16PRS"],
                "ReCom: Efficiency Gap",
                "G16PRS",
                efficiency_gap)

#--- FLIP PROPOSAL VISUALIZATION

max_steps = 20000
step_size = 2000

ts = [x * step_size for x in range(1, int(max_steps / step_size) + 1)]
#[2000, 4000, 6000, 8000, 10000... 20000]

flip_seats = []
flip_mms = []
flip_egs = []

for t in ts:
    temp = np.loadtxt(datadir + f"flip_chain/wins_{str(t)}.csv", delimiter=",")
    for s in range(step_size):
        flip_seats.append(temp[s, :])

    temp = np.loadtxt(datadir + f"flip_chain/mean_median_{str(t)}.csv", delimiter=",")
    for s in range(step_size):
        flip_mms.append(temp[s, :])

    temp = np.loadtxt(datadir + f"flip_chain/efficiency_gap_{str(t)}.csv", delimiter=",")
    for s in range(step_size):
        flip_egs.append(temp[s, :])

df_flip_seats = pd.DataFrame(flip_seats,
                    columns=election_names)

df_flip_mms = pd.DataFrame(flip_mms,
                    columns=election_names)

df_flip_egs = pd.DataFrame(flip_egs,
                    columns=election_names)

#Mean-Median

comparison_hist(df_flip_mms["G16PRS"],
                "Flip: Mean-Median",
                "G16PRS",
                mean_median)

comparison_plot(df_flip_mms["G16PRS"],
                "Flip: Mean-Median",
                "G16PRS",
                mean_median)

#Efficiency Gap

comparison_hist(df_flip_egs["G16PRS"],
                "Flip: Efficiency Gap",
                "G16PRS",
                efficiency_gap)

comparison_plot(df_flip_egs["G16PRS"],
                "Flip: Efficiency Gap",
                "G16PRS",
                efficiency_gap)
