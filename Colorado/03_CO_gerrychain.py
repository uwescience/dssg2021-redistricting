#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Colorado Case Study
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
    accept, 
    constraints,
    updaters,
)
from gerrychain.metrics import efficiency_gap, mean_median, polsby_popper, wasted_votes, partisan_bias
from gerrychain.proposals import recom, propose_random_flip
from gerrychain.updaters import cut_edges, county_splits
from gerrychain.tree import recursive_tree_part, bipartition_tree_random

sys.path.insert(0, os.getenv("REDISTRICTING_HOME"))
import utility_functions as uf

#plt.style.use('seaborn-whitegrid')

#--- IMPORT DATA

try:
    os.chdir(os.path.join(os.getenv("REDISTRICTING_HOME", default=""),
                          "Colorado"))
except OSError:
    os.mkdir(os.path.join(os.getenv("REDISTRICTING_HOME", default=""),
                          "Colorado"))
    os.chdir(os.path.join(os.getenv("REDISTRICTING_HOME", default=""),
                          "Colorado"))

graph = Graph.from_json("Data/co_precincts.json")
df = gpd.read_file("Data/co_precincts.shp")

#--- CREATE SHORTCUTS

state_abbr="CO"
housen="CON"
num_districts=7
pop_col="TOTPOP"
num_elections=2

newdir = "./Outputs/"+state_abbr+housen+"_Precincts/"
print(newdir)
os.makedirs(os.path.dirname(newdir), exist_ok=True)

uf.plot_district_map(df, df['CD116FP'].to_dict(), "2012 Congressional District Map")

#--- DATA CLEANING

df["POC_VAP"] = (df["HVAP"] + df["BVAP"] + df["AMINVAP"] + df["ASIANVAP"] 
                 + df["NHPIVAP"] + df["OTHERVAP"] + df["OTHERVAP"])

for node in graph.nodes():
    graph.nodes[node]["POC_VAP"] = (graph.nodes[node]["HVAP"] + graph.nodes[node]["BVAP"] 
                                    + graph.nodes[node]["AMINVAP"] + graph.nodes[node]["ASIANVAP"] 
                                    + graph.nodes[node]["NHPIVAP"] + graph.nodes[node]["OTHERVAP"] 
                                    + graph.nodes[node]["OTHERVAP"])
    graph.nodes[node]["nPOC_VAP"] = graph.nodes[node]["VAP"] - graph.nodes[node]["POC_VAP"]

#--- GENERIC UPDATERS

updater = {
    "population": updaters.Tally("TOTPOP", alias="population"), 
    "cut_edges": cut_edges,
    "PP":polsby_popper 
            }

#--- ELECTION UPDATERS

election_names=[
    "POC_VAP", 
    "USH18"
    ]

election_columns=[
    ["POC_VAP", "nPOC_VAP"], 
    ["USH18D", "USH18R"] 
    ]

elections = [
    Election(
        election_names[i], 
        {"First": election_columns[i][0], "Second": election_columns[i][1]},
    )
    for i in range(num_elections)
]

election_updaters = {election.name: election for election in elections}
updater.update(election_updaters)

totpop = df.TOTPOP.sum()

#--- ENACTED PLAN BASELINE STATS

partition_2012 = Partition(graph,
                           df["CD116FP"],
                           updater)


stats_2012_df = uf.export_election_metrics_per_partition(partition_2012)

stats_2012_df.loc[:, "cut_edges_value"] = len(partition_2012["cut_edges"])
stats_2012_df.loc[:, "ideal_population"] = sum(partition_2012["population"].values()) / len(partition_2012)

sum(i < 0.55 for i in stats_2012_df["percent"][1]) #number of "competitive" districts


#--- STARTING PLAN (SEED PLAN)

plan_seed = recursive_tree_part(graph, #graph object
                                range(num_districts), #How many districts
                                totpop/num_districts, #population target
                                "TOTPOP", #population column, variable name
                                .01, #epsilon value
                                1)

    
# --- PARTITION (SEED PLAN)

partition_seed = Partition(graph,
                           plan_seed, 
                           updater)

#--- STATS (SEED PLAN)

stats_seed_df = uf.export_election_metrics_per_partition(partition_2012)

uf.plot_district_map(df, 
                     plan_seed, 
                     "Random Seed Plan Map") 

# --- PROPOSAL

ideal_population = sum(partition_seed["population"].values()) / len(partition_seed)

proposal = partial(
    recom,
    pop_col="TOTPOP",
    pop_target=ideal_population,
    epsilon=0.01,
    node_repeats=1,
    method=bipartition_tree_random
)

#--- CREATE CONSTRAINTS

#Minimize County Splits, Compactness Score 

popbound = constraints.within_percent_of_ideal_population(partition_seed, 0.01)

""" COUNTY SPLITS
def num_splits(partition, df=df):
    df["current"] = df['node_names'].map(partition.assignment)
    return sum(df.groupby('City/Town')['current'].nunique()) - df['City/Town'].nunique()

county_bound = county_splits(partition_seed, "COUNTYFP")

county_bound = gerrychain.constraints.refuse_new_splits("COUNTYFP")

"""

compactness_bound = constraints.UpperBound(
    lambda p: len(p["cut_edges"]), 1.5 * len(partition_seed["cut_edges"])
)

#--- ACCEPTANCE FUNCTIONS

def competitive_accept(partition):
    new_score = 0 
    old_score = 0 
    for i in range(7):
        if .45 < partition.parent['USH18'].percents("First")[i] <.55:
            old_score += 1
            
        if .45 < partition['USH18'].percents("First")[i] <.55:
            new_score += 1
            
    if new_score >= old_score:
        return True
    
    elif random.random() < .05:
        return True
    else:
        return False

#--- MCMC CHAINS

chain = MarkovChain(
    proposal=proposal,
    constraints=[
        constraints.within_percent_of_ideal_population(partition_seed, 0.01),
        compactness_bound,
    ],
    accept=competitive_accept, 
    initial_state=partition_seed,
    total_steps=2000
)

#--- RUN CHAINS

dem_seats = []
comps = []

chain_loop = pd.DataFrame(columns=[],
                         index=df.index)
n=0
for part in chain: 
    df['current'] = df.index.map(dict(part.assignment))
    #df.plot(column='current',cmap='tab20')
    #plt.axis('off')
    
    chain_loop['current'] = df['current']
    chain_loop=chain_loop.rename(columns={'current': 'step_' + str(n)})    

    dem_seats.append(part['USH18'].wins('First'))
    comps.append(sum([.45<x<.55 for x in part['USH18'].percents('First')]))
    n+=1

plt.hist(comps)
plt.hist(dem_seats)

#--- BUILD VISUALIZATIONS