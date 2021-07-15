#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Virginia Case Study

"""

#--- IMPORT LIBRARIES

import os

import csv
import json
import random

from functools import partial
import pandas as pd
import geopandas as gpd

import matplotlib
import matplotlib.pyplot as plt
#import matplotlib.style as style
#style.use('dark_background')

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

#--- IMPORT DATA

#os.chdir("/Volumes/GoogleDrive/My Drive/2021/DSSG/github/Virginia/")

graph = Graph.from_json("VA_Chain.json")
df = gpd.read_file("VA_precincts.shp")

#Eyeball the state
df.plot(column="CD_12", cmap="tab20", figsize=(12,8))
plt.axis("off")
plt.title("2012 Congressional District Map")
plt.show()

df.plot(column="CD_16", cmap="tab20", figsize=(12,8))
plt.axis("off")
plt.title("2016 Congressional District Map")
plt.show()

#--- CREATE SHORTCUTS

state_abbr="VA"
housen="CON"
num_districts=11
pop_col="TOTPOP"
num_elections= 3

#Make an output directory to dump files in

newdir = "./Outputs/"+state_abbr+housen+"_Precincts/"
print(newdir)

#!!!! WARNING: This code deletes folder without warning and even if there are files in it
#!!!! CHECK newdir BEFORE RUNNING!
#import shutil
#shutil.rmtree(os.path.dirname(newdir)) #WARNING
#!!!! CHECK newdir BEFORE RUNNING!

os.makedirs(os.path.dirname(newdir), exist_ok=True)
#with open(newdir + "init.txt", "w") as f:
#    f.write("Created Folder")

#--- GENERIC UPDATERS

updater = {
    "population": updaters.Tally("TOTPOP", alias="population"), #can only take in partitions
    "cut_edges": cut_edges,
    "PP":polsby_popper
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

for node in graph.nodes():
    graph.nodes[node]["nBVAP"] = graph.nodes[node]["VAP"] - graph.nodes[node]["BVAP"]

election_updaters = {election.name: election for election in elections}

updater.update(election_updaters)

#--- DATA CLEANING

totpop = 0
for n in graph.nodes():
    totpop+= graph.nodes[n]["TOTPOP"]
    graph.nodes[n]["G18DSEN"] = int(float(graph.nodes[n]["G18DSEN"]))
    graph.nodes[n]["G18RSEN"] = int(float(graph.nodes[n]["G18RSEN"]))
    graph.nodes[n]["G16DPRS"] = int(float(graph.nodes[n]["G16DPRS"]))
    graph.nodes[n]["G16RPRS"] = int(float(graph.nodes[n]["G16RPRS"]))

#--- STARTING PLAN (SEED PLAN)

cddict =  recursive_tree_part(graph, #graph object
                              range(num_districts), #How many districts
                              totpop/num_districts, #population target
                              "TOTPOP", #population column, variable name
                              .01, #epsilon value
                              1)

df['initial'] = df.index.map(cddict)

df.plot(column="initial",cmap="tab20", figsize=(12,8))
plt.axis("off")
plt.title("Seed Plan: Recursive Partitioning Tree")
plt.show()
#plt.savefig(newdir+"initial.png")

#--- PARTITION

initial_partition = Partition(graph,
                              cddict, #initial plan (this is our recurisive_tree_part)
                              updater)

ideal_population = sum(initial_partition["population"].values()) / len(initial_partition)

with open(newdir+"init.json", 'w') as jf1:
        json.dump(cddict, jf1)

#--- LOOK AT STATS OF STARTING PLAN

stats_df = pd.DataFrame(index=["d1_pct",
                               "d2_pct",
                               "d3_pct",
                               "d4_pct",
                               "d5_pct",
                               "d6_pct",
                               "d7_pct",
                               "d8_pct",
                               "d9_pct",
                               "d10_pct",
                               "d11_pct",
                               "mean_median",
                               "efficiency_gap",
                               "seat_num",
                               #"cut_edges"
                               ],
                        columns=election_names)

for i in range(3):
    current_column = stats_df.columns[i]
    dist_i = initial_partition[election_names[i]].percents("First")
    stats_df[current_column] = [float(dist_i[0]),
                                float(dist_i[1]),
                                float(dist_i[2]),
                                float(dist_i[3]),
                                float(dist_i[4]),
                                float(dist_i[5]),
                                float(dist_i[6]),
                                float(dist_i[7]),
                                float(dist_i[8]),
                                float(dist_i[9]),
                                float(dist_i[10]),
                                float(mean_median(initial_partition[election_names[i]])) ,
                                #What percent of the vote over 50% would you need to get 50% of the seats, e.g., 3, need 53% to get half of seats
                                #Close to 0 then how symmetrical/fair is the votes
                                float(efficiency_gap(initial_partition[election_names[i]])),
                                float(initial_partition[election_names[i]].wins("First")),
                                #float(len(initial_partition["cut_edges"]))
                                ]

stats_inital_df = stats_df.transpose()

stats_inital_df.plot(y=['d1_pct','d2_pct','d3_pct','d4_pct','d5_pct','d6_pct',
                        'd7_pct','d8_pct','d9_pct','d10_pct','d11_pct']
                     )

stats_inital_df.plot(y=['mean_median','efficiency_gap']
                     )
"""
with open(newdir + "Start_Values.txt", "w") as f:
    f.write("Values for Starting Plan: Tree Recursive\n \n ")
    f.write("Initial Cut: " + str(len(initial_partition["cut_edges"])))
    f.write("\n")
    f.write("\n")

    for elect in range(num_elections):
        f.write(
            election_names[elect]
            + "_District Percentages"
            + str(
                sorted(initial_partition[election_names[elect]].percents("First"))
            )
        )
        f.write("\n")
        f.write("\n")

        f.write(
            election_names[elect]
            + "_Mean-Median :"
            + str(mean_median(initial_partition[election_names[elect]]))
        )

        f.write("\n")
        f.write("\n")

        f.write(
            election_names[elect]
            + "_Efficiency Gap :"
            + str(efficiency_gap(initial_partition[election_names[elect]]))
        )

        f.write("\n")
        f.write("\n")

        f.write(
            election_names[elect]
            + "_How Many Seats :"
            + str(initial_partition[election_names[elect]].wins("First"))
        )

        f.write("\n")
        f.write("\n")
"""

#--- PROPOSAL

proposal = partial(#All the functions inside gerrychain want to take partition, but recom wants more functions than that
                   #Partial takes main functions and prefill all the objects until it becomes a partition
    recom,
    pop_col="TOTPOP",
    pop_target=ideal_population,
    epsilon=0.05,
    node_repeats=1,
    method = bipartition_tree_random
)

#--- CREATE CONSTRAINTS

compactness_bound = constraints.UpperBound(
    lambda p: len(p["cut_edges"]), 1.5 * len(initial_partition["cut_edges"])
)

#--- ACCEPTANCE FUNCTIONS

"""
def sixty_accept(partition):
#competitiveness comparison between old and new chains
#how many of these districts have percentages under 60%
    new = sum(x<.6 for x in partition["G18SEN"].percents("First"))
    #Looking at each district and if the votes are less than 60% and summing all districts

    old = sum(x<.6 for x in partition.parent["G18SEN"].percents("First"))
    #Parition parent provides us access to parent and evaluate in comparison to the next step

    if new > old:  #Newer chain have fewer districts not meeting the 60% than old chain
        return True

    else:
        return False

def unpack_accept(part):
    if max(part["PRES16"].percents("D")) < .65:
        return True

    elif max(part["PRES16"].percents("D")) < max(part.parent["PRES16"].percents("D")):
        return True

    elif random.random() < max(part.parent["PRES16"].percents("D"))/max(part["PRES16"].percents("D")):
        return True
    else:
        return False

"""
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

"""
mix_chain = MarkovChain(
    proposal= #check out how to build mix
    constraints=[ #constraints for mix
    ],
    accept=accept.always_accept,
    initial_state=initial_partition,
    total_steps= #ideal num of steps?
)
"""

#--- RUN RECOMBINATION PROPOSAL

"""
#Testing whether 60% acceptance function is feasible
sixty = []

for part in recom_chain:

    sixty.append(sum(x<.6 for x in part["G18SEN"].percents("First"))) #

plt.plot(sixty) #number of competitive districts below 60%

df['recom'] = df.index.map(dict(part.assignment)) #new plan
df.plot(column="recom",cmap='tab20')
"""
recom_pop_vec = [] #total population
recom_cut_vec = [] #cut edges
recom_votes = [[], [], [], [],[],[]] #number of votes
recom_mms = [] #mean-median score
recom_egs = [] #efficiency gap score
recom_hmss = [] #how many seats score

t = 0 #keep track on how long we make it in the chain

for repart in recom_chain:

    recom_pop_vec.append(sorted(list(repart["population"].values())))
    recom_cut_vec.append(len(repart["cut_edges"]))
    recom_mms.append([])
    recom_egs.append([])
    recom_hmss.append([])

    for elect in range(num_elections):
        recom_votes[elect].append(sorted(repart[election_names[elect]].percents("First")))
        recom_mms[-1].append(mean_median(repart[election_names[elect]]))
        recom_egs[-1].append(efficiency_gap(repart[election_names[elect]]))
        recom_hmss[-1].append(repart[election_names[elect]].wins("First"))

    t += 1
    if t % 200 == 0:
        #sorted by score and by column aligned with the election_columns
        print(t)

        df["plot" + str(t)] = df.index.map(dict(repart.assignment))
        df.plot(column="plot" + str(t), cmap="tab20", edgecolor="face")
        plt.axis("off")
        plt.title(str(t) + " Steps")
        plt.savefig(newdir + "recom_plot" + str(t) + ".png")
        plt.close()

df["recom"] = df.index.map(dict(repart.assignment))

df.plot(column="recom",cmap='tab20', figsize=(12,8))
plt.axis("off")
plt.title("ReCom Proposal: 2,000 Steps")
plt.show()

#--- RUN FLIP BOUNDARY PROPOSAL

flip_pop_vec = [] #total population
flip_cut_vec = [] #cut edges
flip_votes = [[], [], [], [],[],[]] #number of votes
flip_mms = [] #mean-median score
flip_egs = [] #efficiency gap score
flip_hmss = [] #how many seats score

t = 0

for fpart in flip_chain:

    flip_pop_vec.append(sorted(list(fpart["population"].values())))
    flip_cut_vec.append(len(fpart["cut_edges"]))
    flip_mms.append([])
    flip_egs.append([])
    flip_hmss.append([])

# Write results from flip_chain

    for elect in range(num_elections):
        flip_votes[elect].append(sorted(fpart[election_names[elect]].percents("First")))
        flip_mms[-1].append(mean_median(fpart[election_names[elect]]))
        flip_egs[-1].append(efficiency_gap(fpart[election_names[elect]]))
        flip_hmss[-1].append(fpart[election_names[elect]].wins("First"))

    t += 1

    if t % 2000 == 0:

        print(t)

        with open(newdir + "flip_mms" + str(t) + ".csv", "w") as tf1:
            writer = csv.writer(tf1, lineterminator="\n")
            writer.writerows(flip_mms) #want to see ~0.015 for something reasonable

        with open(newdir + "flip_egs" + str(t) + ".csv", "w") as tf1:
            writer = csv.writer(tf1, lineterminator="\n")
            writer.writerows(flip_egs)

        with open(newdir + "flip_hmss" + str(t) + ".csv", "w") as tf1:
            writer = csv.writer(tf1, lineterminator="\n")
            writer.writerows(flip_hmss)

        with open(newdir + "flip_pop" + str(t) + ".csv", "w") as tf1:
            writer = csv.writer(tf1, lineterminator="\n")
            writer.writerows(flip_pop_vec)

        #with open(newdir + "flip_cuts" + str(t) + ".csv", "w") as tf1:
        #    writer = csv.writer(tf1, lineterminator="\n")
        #    writer.writerows(flip_cut_vec)

        with open(newdir + "flip_assignment" + str(t) + ".json", "w") as jf1:
            json.dump(dict(fpart.assignment), jf1)

        for elect in range(num_elections):
            with open(
                newdir + election_names[elect] + "_flip_" + str(t) + ".csv", "w"
            ) as tf1:
                writer = csv.writer(tf1, lineterminator="\n")
                writer.writerows(flip_votes[elect])

        df["plot" + str(t)] = df.index.map(dict(fpart.assignment))
        df.plot(column="plot" + str(t), cmap="tab20")
        plt.axis("off")
        plt.title(str(t) + " Steps")
        plt.savefig(newdir + "flip_plot" + str(t) + ".png")
        plt.close()

        votes = [[], [], [], [],[],[]]
        mms = []
        egs = []
        hmss = []
        pop_vec = []
        #cut_vec = []

df["flip"] = df.index.map(dict(fpart.assignment))

df.plot(column="flip",cmap='tab20', figsize=(12,8))
plt.axis("off")
plt.title("Flip Proposal: 20,000 Steps")
plt.show()

#--- BUILD VISUALIZATIONS

#import os
import math
#import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# sns.set_style('darkgrid')
sns.set_style("darkgrid", {"axes.facecolor": ".97"})

datadir = "./Outputs/VACON_Precincts/"

#--- RECOM PROPOSAL VISUALIZATION

df_recom_seats = pd.DataFrame(recom_hmss,
                    columns=election_names)

df_recom_cuts = pd.DataFrame(recom_cut_vec)

df_recom_mms = pd.DataFrame(recom_mms,
                    columns=election_names)

df_recom_egs = pd.DataFrame(recom_egs,
                    columns=election_names)

#Mean-Median

plt.hist(df_recom_mms["G16PRS"])
plt.title("ReCom: Mean-Median")
plt.vlines(x=sum(df_recom_mms["G16PRS"])/len(df_recom_mms["G16PRS"]),
           ymin=0,
           ymax=450,
           colors="red",
           linestyles="solid")
plt.show()

plt.plot(df_recom_mms["G16PRS"])
plt.title("ReCom: Mean-Median")
plt.hlines(y=sum(df_recom_mms["G16PRS"])/len(df_recom_mms["G16PRS"]),
           xmin=0,
           xmax=2100,
           colors="red",
           linestyles="solid")
plt.show()

#Efficiency Gap

plt.hist(df_recom_egs["G16PRS"])
plt.title("ReCom: Efficiency Gap")
plt.vlines(x=sum(df_recom_egs["G16PRS"])/len(df_recom_egs["G16PRS"]),
           ymin=0,
           ymax=1200,
           colors="red",
           linestyles="solid")
plt.show()

plt.plot(df_recom_egs["G16PRS"])
plt.title("ReCom: Efficiency Gap")
plt.hlines(y=sum(df_recom_egs["G16PRS"])/len(df_recom_egs["G16PRS"]),
           xmin=0,
           xmax=2100,
           colors="red",
           linestyles="solid")
plt.show()

#TO DO: Calculate mean-median and efficiency gap using the 2012 and 2016 plans
#sum(df["G16DPRS"].astype(float)), sum(df["G16RPRS"].astype(float))

#--- FLIP PROPOSAL VISUALIZATION

max_steps = 20000
step_size = 2000

ts = [x * step_size for x in range(1, int(max_steps / step_size) + 1)]
#[2000, 4000, 6000, 8000, 10000... 20000]

flip_seats = []
flip_mms = []
flip_egs = []
#flip_cut_vec=[]

for t in ts:
    temp = np.loadtxt(datadir + "flip_hmss" + str(t) + ".csv", delimiter=",")
    for s in range(step_size):
        flip_seats.append(temp[s, :])

    temp = np.loadtxt(datadir + "flip_mms" + str(t) + ".csv", delimiter=",")
    for s in range(step_size):
        flip_mms.append(temp[s, :])

    temp = np.loadtxt(datadir + "flip_egs" + str(t) + ".csv", delimiter=",")
    for s in range(step_size):
        flip_egs.append(temp[s, :])

    #temp_c = np.loadtxt(datadir + "flip_cuts" + str(t) + ".csv", delimiter=",")
    #temp_c = temp_c.transpose()
    #for s in range(step_size):
    #    flip_cut_vec.append(temp_c[s])


df_flip_seats = pd.DataFrame(flip_seats,
                    columns=election_names)

df_flip_mms = pd.DataFrame(flip_mms,
                    columns=election_names)

df_flip_egs = pd.DataFrame(flip_egs,
                    columns=election_names)

#Mean-Median

plt.hist(df_flip_mms["G16PRS"])
plt.title("Flip: Mean-Median")
plt.vlines(x=sum(df_flip_mms["G16PRS"])/len(df_flip_mms["G16PRS"]),
           ymin=0,
           ymax=3500,
           colors="red",
           linestyles="solid")
plt.show()

plt.plot(df_flip_mms["G16PRS"])
plt.title("Flip: Mean-Median")
plt.hlines(y=sum(df_flip_mms["G16PRS"])/len(df_flip_mms["G16PRS"]),
           xmin=0,
           xmax=20000,
           colors="red",
           linestyles="solid")
plt.show()

#Efficiency Gap

plt.hist(df_flip_egs["G16PRS"])
plt.title("Flip: Efficiency Gap")
plt.vlines(x=sum(df_flip_egs["G16PRS"])/len(df_flip_egs["G16PRS"]),
           ymin=0,
           ymax=20000,
           colors="red",
           linestyles="solid")
plt.show()

plt.plot(df_flip_egs["G16PRS"])
plt.title("Flip: Efficiency Gap")
plt.hlines(y=sum(df_flip_egs["G16PRS"])/len(df_flip_egs["G16PRS"]),
           xmin=0,
           xmax=20000,
           colors="red",
           linestyles="solid")
plt.show()
