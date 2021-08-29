#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 3. Applying GerryChain
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

sns.set_theme(style="whitegrid")

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
from gerrychain.tree import recursive_tree_part, bipartition_tree_random, PopulatedGraph, \
                            find_balanced_edge_cuts_memoization, random_spanning_tree

sys.path.insert(0, os.getenv("REDISTRICTING_HOME"))
import utility_functions as uf

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

uf.plot_district_map(df, 
                     df['CD116FP'].to_dict(), 
                     title="2012 Colorado Congressional District Map",
                     map_colors="Dark2")
uf.plot_district_map(df, df['COUNTYFP'].to_dict(), 
                     title="Colorado County Map",
                     map_colors="Set3")

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

def num_splits(partition, df=df):
    df["current"] = df.index.map(partition.assignment)
    return sum(df.groupby('COUNTYFP')['current'].nunique() > 1)

updater = {
    "population": updaters.Tally("TOTPOP", alias="population"), 
    "cut_edges": cut_edges,
    "count_splits": num_splits
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

plan_2012_stats = uf.export_election_metrics_per_partition(partition_2012)

plan_2012_stats = plan_2012_stats.drop(labels="POC_VAP", axis=0)
plan_2012_stats = plan_2012_stats.rename({'wins':'dem_wins'}, axis=1)

plan_2012_names = plan_2012_stats.index.values
plan_2012_comp=[]
for n in range(1):
    plan_2012_comp.append(sum([.45<x<.55 for x in partition_2012[plan_2012_names[n]].percents('First')]))

plan_2012_stats['comp_dist'] = np.array(plan_2012_comp)

print(partition_2012["count_splits"])

# --- PROPOSAL

def get_spanning_tree_u_w(G):
    node_set=set(G.nodes())
    x0=random.choice(tuple(node_set))
    x1=x0
    while x1==x0:
        x1=random.choice(tuple(node_set))
    node_set.remove(x1)
    tnodes ={x1}
    tedges=[]
    current=x0
    current_path=[x0]
    current_edges=[]
    while node_set != set():
        next=random.choice(list(G.neighbors(current)))
        current_edges.append((current,next))
        current = next
        current_path.append(next)
        if next in tnodes:
            for x in current_path[:-1]:
                node_set.remove(x)
                tnodes.add(x)
            for ed in current_edges:
                tedges.append(ed)
            current_edges = []
            if node_set != set():
                current=random.choice(tuple(node_set))
            current_path=[current]
        if next in current_path[:-1]:
            current_path.pop()
            current_edges.pop()
            for i in range(len(current_path)):
                if current_edges !=[]:
                    current_edges.pop()
                if current_path.pop() == next:
                    break
            if len(current_path)>0:
                current=current_path[-1]
            else:
                current=random.choice(tuple(node_set))
                current_path=[current]
    #tgraph = Graph()
    #tgraph.add_edges_from(tedges)
    return G.edge_subgraph(tedges)

def my_uu_bipartition_tree_random(graph, pop_col, pop_target, epsilon, node_repeats=1, 
                                  spanning_tree=None, choice=random.choice):
    county_weight = 20
    populations = {node: graph.nodes[node][pop_col] for node in graph}
    possible_cuts = []
    if spanning_tree is None:
        spanning_tree = get_spanning_tree_u_w(graph)
    while len(possible_cuts) == 0:
        for edge in graph.edges():
            if graph.nodes[edge[0]]["COUNTYFP"] == graph.nodes[edge[1]]["COUNTYFP"]:
                graph.edges[edge]["weight"] = county_weight * random.random()
            else:
                graph.edges[edge]["weight"] = random.random()
        spanning_tree = get_spanning_tree_u_w(graph)
        h = PopulatedGraph(spanning_tree, populations, pop_target, epsilon)
        possible_cuts = find_balanced_edge_cuts_memoization(h, choice=choice)
    return choice(possible_cuts).subset

ideal_population = sum(partition_2012["population"].values()) / len(partition_2012)

proposal = partial(
    recom,
    pop_col="TOTPOP",
    pop_target=ideal_population,
    epsilon=0.01,
    node_repeats=1,
    method=my_uu_bipartition_tree_random
)

#--- CREATE CONSTRAINTS

popbound = constraints.within_percent_of_ideal_population(partition_2012, 0.01)

compactness_bound = constraints.UpperBound(
    lambda p: len(p["cut_edges"]), 1.5 * len(partition_2012["cut_edges"])
)

#--- ACCEPTANCE FUNCTIONS

def competitive_nudge_county_accept(partition):
    nudge_old = 0 
    nudge_new = 0 
    
    for i in range(7):
        if .40 < partition.parent['USH18'].percents("First")[i] <.60:
            nudge_old += 1
            
        if .40 < partition['USH18'].percents("First")[i] <.60:
            nudge_new += 1
    
    band_old = 0 
    band_new = 0 
    for i in range(7):
        if .45 < partition.parent['USH18'].percents("First")[i] <.55:
            band_old += 1
            
        if .45 < partition['USH18'].percents("First")[i] <.55:
            band_new += 1
            
    if (nudge_new >= nudge_old) and (band_new >= band_old) \
    and (partition["count_splits"] < partition.parent["count_splits"]):
        return True
    
    elif (band_new >= band_old) \
    and (partition["count_splits"] < partition.parent["count_splits"]):
        return True
    
    elif (nudge_new >= nudge_old) and (band_new >= band_old) \
    and (random.random() < .10):
        return True
        
    elif (random.random() < .10) \
    and (partition["count_splits"] < partition.parent["count_splits"]):
        return True
    
    else:
        return False
    
#--- MCMC CHAINS

chain = MarkovChain(
    proposal=proposal,
    constraints=[
        popbound,
        compactness_bound
    ],
    accept=competitive_nudge_county_accept,
    initial_state=partition_2012,
    total_steps=20000
)

#--- RUN CHAINS

comps = []
splits = []
t=0

for part in chain:
    splits.append(part["count_splits"])
    comps.append(sum([.45<x<.55 for x in part['USH18'].percents('First')]))
    t += 1
    if t % 200 == 0:
        print(t)
    if t%20000==0:
        t=0

#--- BUILD VISUALIZATIONS

plt.hist(comps)
plt.hist(splits)