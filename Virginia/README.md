## Virginia 

During the process of developing our stakeholder guide, the DSSG team conducted an exploratory analysis of Virginia to familiarize ourselves with modeling decisions and GerryChain components. `VA_Case_Study.py` outlines a sequenced GerryChain script that beginning users can use to consider how to write a GerryChain analysis from start to finish. `VA_MeanSeatTest.ipynb` document modeling decisions considered when interpreting proposed legislative text, which can serve as a template for exploratory data analysis and modeling for a state.

#### Modeling: Legislative Proposals for Partisan Tests

Virginia has a new bipartisan commission for the upcoming cycle and the state has a long history of racial and partisan gerrymandering. A potential test that state legislatures might consider to assess partisan gerrymandering is:

"No state shall draw the districts for a state legislative body thereof so as to unduly favor a political party such that the political party is able to consistently maintain majority control of the body with a minority of the popular vote or maintain supermajority control of the body, as defined by the laws of the state and procedures of the body, with a bare majority of the popular vote. 

It shall create a rebuttable presumption that a state has violated section (a) by unduly favoring a political party if the mean number of seats of the legislative body won by the favored party when the results of a representative sample of statewide elections are applied across the challenged districting plan deviates more than 1 standard deviation away from the mean number of seats won by that party when the same election results are applied across a robust neutral ensemble of districting plans.

A representative sample of statewide elections means a collection of no fewer than 4 contests from the preceding 4 general election cycles which featured candidates from the two political parties whose candidates for president garnered the most votes in the most recent presidential election in the state. 

A robust neutral ensemble means a statistically significant collection of districting plans generated using only the following limiting factors to constrain the acceptance of plans into the collection ensemble:
(A) contiguity;
(B) adherence to established one person-one vote requirements; and
(C) a compactness requirement justified by comparison to historically enacted plans in the state."

