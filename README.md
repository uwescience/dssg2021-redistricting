# Developing Ensemble Methods for Initial Districting Plan Evaluation
This repository collects associated Python scripts and data to support the 2021 UW Data Science for Social Good project, "Developing Ensemble Methods for Initial Districting Plan Evaluation".

More details can be found on our [website](https://uwescience.github.io/DSSG2021-redistricting-website/).

# Project Details

Broadly, this project has two core aims: to understand how computational tools can be used to assist in the drawing of maps and to evaluate the fairness of proposed and enacted plans. The data used in this project come from [MGGG Redistricting Lab](https://github.com/mggg-states). Details about original data source and the processing steps can be found on the state's associated directory. 

We conducted three different use cases of GerryChain across three different state contexts to demonstrate the process of computational redistricting. The purpose of this exercise is to illustrate the decision process associated with different types of analysis across differing state contexts. These case studies complement the guide’s outline of GerryChain steps from start to finish, following the application of each stage alongside example code found in the relevant state directory.

* Georgia: Using GerryChain’s Built in Metrics
  
  Focus: Partisan Metrics

* Colorado: Using GerryChain to Support the Map Drawing Process

Focus: Political Competitiveness

* Texas: Using GerryChain to Evaluate Proposed or Enacted Maps
 
Focus: Voting Rights Act and Opportunity Districts



# Reproducibility

To reproduce case study results and to follow along with our [User's Guide](https://uwescience.github.io/DSSG2021-redistricting-website/gerrychain/), please ensure the following steps are taken first:

* Set REDISTRICTING_HOME environment variable to base location of project using `export REDISTRICTING_HOME=[your_path]`.
