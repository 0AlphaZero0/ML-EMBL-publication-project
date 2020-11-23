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

The prototype used two machine learning models and two vectorizer built during the FREYA project. All this project is described in **[Confluence](https://img.shields.io/badge/Confluence-FREYA%2FEMBL%20project-green)**

## Running
As described before, the prototype needs a list of PMIDs
