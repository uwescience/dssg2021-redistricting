#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Colorado Case Study: Converting shapefile to json 
"""

import os
import sys

import geopandas as gpd
from gerrychain import Graph
import networkx as nx
import matplotlib.pyplot as plt

sys.path.insert(0, os.getenv("REDISTRICTING_HOME"))
try:
    os.chdir(os.path.join(os.getenv("REDISTRICTING_HOME", default=""),
                          "Colorado"))
except OSError:
    os.mkdir(os.path.join(os.getenv("REDISTRICTING_HOME", default=""),
                          "Colorado"))
    os.chdir(os.path.join(os.getenv("REDISTRICTING_HOME", default=""),
                          "Colorado"))

#This loads in the shapefile as a geodataframe so we can add the centroids to the nodes for plotting
df = gpd.read_file("Data/co_precincts.shp")

# This line builds the dual graph directly from the shapefile - just ignore the warning for now
graph = Graph.from_file("Data/co_precincts.shp")

centroids = df.centroid
df["C_X"] = centroids.x
df["C_Y"] = centroids.y

graph.add_data(df,columns=["C_X","C_Y"])

#This converts the labels of the current districts to numeric types
for node in graph.nodes():
    graph.nodes[node]["CD116FP"] = int(graph.nodes[node]["CD116FP"])

#Plot the graph to make sure it looks reasonable
nx.draw(graph,pos = {node:(graph.nodes[node]["C_X"],graph.nodes[node]["C_Y"]) 
                     for node in graph.nodes()},node_color=[graph.nodes[node]["CD116FP"] 
                                                            for node in graph.nodes()],node_size=10,cmap='tab20')

                                                            #Write the graph to a .json file so it is easier to load in the future
graph.to_json("Data/co_precincts.json")
