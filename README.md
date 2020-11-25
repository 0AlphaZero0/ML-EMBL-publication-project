<p align="center"><img width=30% src="https://www.ebi.ac.uk/eva/img/dbSNP/EMBL_EBI-logo.png"></p>
<p align="center"><img width=20% src="https://www.infodocket.com/wp-content/uploads/2020/06/2020-06-02_09-17-36.png"></p>



<h1 align="center"> ML-EMBL-publication-project </h1> <br>
<p align="center">
  Repository of ML models  and algorithm built for the EMBL publication project
</p>

![Python](https://img.shields.io/badge/Python-v3.5%2B-blue)
![Confluence](https://img.shields.io/badge/Confluence-FREYA%2FEMBL%20project-green)

## Clone
```bash
git clone https://github.com/0AlphaZero0/ML-EMBL-publication-project.git
```

## Table of Contents

- [Introduction](#introduction)
- [Running](#running)
- [Details](#details)
- [is_EMBL](#is_embl)
- [get_geoloc_from](#get_geoloc_from)


## Introduction
This prototype is made to detect every EMBL paper within a list of PMIDs. An EMBL paper is a paper where there is at least one affiliation to either an EMBL site or an EMBL partnership.

At this time there are 6 different sites and 2 partnerships : 

- Australia (partnership)
- Barcelona
- Hinxton (EMBL-EBI Cambridge)
- Grenoble
- Hamburg
- Heidelberg
- Nordic (partnership)
- Rome

The prototype used two machine learning models and two vectorizer built during the ***[FREYA project](https://www.project-freya.eu/en/about/mission)***. This work has been made for the [deliverable 4.6](https://www.project-freya.eu/en/resources/project-output).

All this project is described in **[`Confluence`](https://img.shields.io/badge/Confluence-FREYA%2FEMBL%20project-green)**

## Running
As described before, the prototype needs a list of PMIDs. This list can be in a file provide by the user or a string directly wrote in the script.
Results are organized by searches, each new search require a file *.csv* or *.txt* containing PMIDs. Each search have its corresponding folder in the searches folder. The results are a list of file, one for each site and one for all EMBL PMIDs detected.
For each site the file will be a *.csv* table like the following :

| PMIDs | EMBL | Member states | Worldwide | Partnership |
| --------------- | --------------- | --------------- | --------------- | --------------- |
| 30537516 | TRUE| TRUE | TRUE | TRUE |
| 30496853 | TRUE | TRUE | FALSE | FALSE |
| 29330484 | TRUE | FALSE | TRUE | FALSE |

To run this prototype just use the following command :
```bash
python .\detect_EMBL.py
```

In the script, 3 variables are necessary to run your search: 
```python
search_name="test"
search_file="test_pmid_EPMC.txt"
directory="./searches/"+search_name+"/"
```
The *search_name* corresponds to a name you choose and the directory name in the *searches* directory. Then the *search_file* corresponds to the file in your directory where the PMIDs you want to process are located.

***This algorithm uses multiprocessing to be able to process huge amount of PMIDs, it is, therefore, possible that the machine where this algorithm run could be slowed.***

## Details
This prototype remains on two algorithms :

### is_EMBL
This algorithm take a an affiliation string and will return a dictionary with prediction scores and methods of prediction. It uses a combination of exact matches and predictions either on the whole string or sub parts of this string.

### get_geoloc_from
This algorithm take a an affiliation string and will return a dictionary with corresponding geolocation information found in this string. This algorithm is not the best one to extract geolocation from a string and thus to improve the EMBL detection this is one algorithm to think about.
