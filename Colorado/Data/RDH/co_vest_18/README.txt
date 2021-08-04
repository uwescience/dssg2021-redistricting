2018 Colorado precinct and election shapefile.

## RDH Date retrieval
12/07/2020

## Sources
Election results from the Colorado Secretary of State (https://www.sos.state.co.us/pubs/elections/Results/Archives.html)
Precinct shapefiles primarily from the U.S. Census Bureau's 2020 Redistricting Data Program Phase 2 release. The following counties used shapefiles sourced from the respective county governments instead: Adams, Arapahoe, Boulder, Delta, Denver, Douglas, El Paso, Fremont, Garfield, Gilpin, Jefferson, La Plata, Larimer, Mesa, Pitkin, Pueblo, Summit, Weld.

## Fields metadata

Vote Column Label Format
------------------------
Columns reporting votes follow a standard label pattern. One example is:
G16PREDCli
The first character is G for a general election, P for a primary, S for a special, and R for a runoff.
Characters 2 and 3 are the year of the election.
Characters 4-6 represent the office type (see list below).
Character 7 represents the party of the candidate.
Characters 8-10 are the first three letters of the candidate's last name.

Office Codes
A## - Ballot amendment, where ## is an identifier
AGR - Commissioner of Agriculture
ATG - Attorney General
AUD - Auditor
CFO - Chief Financial Officer
CHA - Council Chairman
COC - Corporation Commissioner
COM - Comptroller
CON - State Controller
COU - City Council Member
CSC - Clerk of the Supreme Court
DEL - Delegate to the U.S. House
GOV - Governor
H## - U.S. House, where ## is the district number. AL: at large.
HOD - House of Delegates, accompanied by a HOD_DIST column indicating district number
HOR - U.S. House, accompanied by a HOR_DIST column indicating district number
INS - Insurance Commissioner
LAB - Labor Commissioner
LND - Commissioner of Public/State Lands
LTG - Lieutenant Governor
MAY - Mayor
MNI - State Mine Inspector
PSC - Public Service Commissioner
PUC - Public Utilities Commissioner
RGT - State University Regent
SAC - State Appeals Court
SBE - State Board of Education
SOC - Secretary of Commonwealth
SOS - Secretary of State
SPI - Superintendent of Public Instruction
SPL - Commissioner of School and Public Lands
SSC - State Supreme Court
TAX - Tax Commissioner
TRE - Treasurer
UBR - University Board of Regents/Trustees/Governors
USS - U.S. Senate

Party Codes
D and R will always represent Democrat and Republican, respectively.
See the state-specific notes for the remaining codes used in a particular file; note that third-party candidates may appear on the ballot under different party labels in different states.

## Fields
G18GOVDPOL - Jared Polis (Democratic Party)
G18GOVRSTA - Walker Stapleton (Republican Party)
G18GOVLHEL - Scott Helker (Libertarian Party)
G18GOVOHAM - Bill Hammons (Unity Party)

G18ATGDWEI - Phil Weiser (Democratic Party)
G18ATGRBRA - George Brauchler (Republican Party)
G18ATGLROB - William F. Robinson III (Libertarian Party)

G18SOSDGRI - Jena Griswold (Democratic Party)
G18SOSRWIL - Wayne Williams (Republican Party)
G18SOSCCAM - Amanda Campbell (American Constitution Party)
G18SOSOHUB - Blake Huber (Approval Voting Party)

G18TREDYOU - Dave Young (Democratic Party)
G18TRERWAT - Brian Watson (Republican Party)
G18TRECKIL - Gerald F. Kilpatrick (American Constitution Party)

G18RGTDSMI - Lesley Smith (Democratic Party)
G18RGTRMON - Ken Montera (Republican Party)
G18RGTLTRE - James K. Treibert (Libertarian Party)
G18RGTOOTW - Christopher E. Otwell (Unity Party)

## Processing Steps
Las Animas County precinct assignments in the voter file differ markedly from both the Census VTD boundaries and from maps received from the county. All precincts were revised to match the geocoded voter file and the list of districts assigned to precinct splits. As appropriate, precinct boundaries were revised using Census blocks, the Trinidad municipal boundary shapefile, school district or fire district boundaries, and the parcel shapefile from the Las Animas County Assessor.

The following additional revisions were made to match the 2018 precinct boundaries:

Logan: Align Sterling City precincts with city limits
Montezuma: Precincts renumbered to match county maps
Otero: Align La Junta City precincts with county maps
Prowers: All precincts adjusted to match county maps
Rio Grande: Adjust Precincts 2/3 to match county maps

