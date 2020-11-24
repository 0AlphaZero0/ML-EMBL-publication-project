#!/usr/bin/env python
#-*- coding: utf-8 -*-
# THOUVENIN Arthur athouvenin@outlook.fr
# 26/02/2020
########################
import geonamescache # Allows to use data from Geonames database (http://www.geonames.org/)
import joblib # Used to save and load machine learning models in files (https://joblib.readthedocs.io/en/latest/)
import json # Used to load json from url response (https://docs.python.org/3/library/json.html)
import numpy # 
import os
import pycountry# Allows to load a dictionnary of iso-2 iso3 country codes
import re
import requests
import spacy # Allows to use pre-trained models for NER (Name Entity Recognition)
import time
import tqdm
from multiprocessing import Pool,Manager,cpu_count

#############################                   VARIABLES                   #############################

search_name="test"
search_file="test_pmid_EPMC.txt"
directory="./searches/"+search_name+"/"

####    MODELS    ####
EMBL_ID_Vecto=joblib.load("./models/EMBL_ID_Vecto.joblib") # TfidfVectorizer train to EMBL detection
EMBL_ID_clf=joblib.load("./models/EMBL_ID_clfLR.joblib") # Logistic Regression train to EMBL detection
EMBL_Sites_ID_Vecto=joblib.load("./models/EMBL_Sites_ID_Vecto.joblib") # TidfVectorizer train to EMBL-sites detection
EMBL_Sites_ID_clfLR=joblib.load("./models/EMBL_Sites_ID_clfLR.joblib") # Logistic Regression train to EMBL-sites detection
EMBL_sites={ ### Dictionary of classes (1 site/1 int)
    0:"EMBL Australia",
    1:"EMBL Barcelona",
    2:"EMBL-EBI",
    3:"EMBL Grenoble",
    4:"EMBL Hamburg",
    5:"EMBL Heidelberg",
    6:"EMBL Nordic",
    7:"EMBL Rome"}
replacements=[ #List of tuples for string preparation (regex pattern, replacement)
    (r'[^\s]+@[^\s]+',' '),
    (r'\\',' '),
    (r'\/',' '),
    (r'-',' '),
    (r'\n',' '),
    (r'\t',' ',),
    (r'^\s*',' '),
    (r'"',"'"),
    (r',\s*$',' '),
    (r'\s.\s',' '),
    (r'^[0-9]+\s|^\s[0-9]+\s',''),
    (r'^\s+',''),
    (r'Electronic address\s*:',''),
    (r'Current address\s*:','')]
EMBL_RORs=[ #List of ROR IDs for the different EMBL sites
    "https://ror.org/00yx5cw48", # Australia 1st
    "https://ror.org/050589e39", # Hamburg
    "https://ror.org/03mstc592", # Heidelberg
    "https://ror.org/010jaxs89", # Barcelona
    "https://ror.org/01zjc6908", # Grenoble
    "https://ror.org/01yr73893", # Rome
    "https://ror.org/052c7a780", # Australia 2nd
    "https://ror.org/02catss52"] # EBI
member_states=[ #List of EMBL member states
    "Austria",
    "Belgium",
    "Croatia",
    "Czech Republic",
    "Denmark",
    "Finland",
    "France",
    "Germany",
    "Greece",
    "Hungary",
    "Iceland",
    "Ireland",
    "Israel",
    "Italy",
    "Lithuania",
    "Luxembourg",
    "Malta",
    "Montenegro",
    "The Netherlands",
    "Norway",
    "Portugal",
    "Slovakia",
    "Spain",
    "Sweden",
    "Switzerland",
    "United Kingdom"]
associate_member_states=[ #List of EMBL associate member states
    "Argentina",
    "Australia"]

#############################                   DEFINITIONS                   #############################

def chunkIt(seq, num): #### Create a list of sublist
    """This function will create from a list, a list of chunks
    Description : 
            This function will create different sublists or chunks from a bigger list. All chunks generated will be stored in a list
    Args :
            seq (list-str) : 
                    Sequence to divide in sublists
            num (int) :
                    Size of chunks to return at the end
    Return :
            out (list) :
                    List of chunks  
    """
    avg = len(seq) / float(num)
    out = []
    last = 0.0
    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg
    return out

def gen_list_extract(var, key):
    if isinstance(var, dict):
        for k, v in var.items():
            if k == key:
                yield v
            if isinstance(v, (dict, list)):
                yield from gen_list_extract(v, key)
    elif isinstance(var, list):
        for d in var:
            yield from gen_list_extract(d, key)

def main(): #### Create a queue of process, each process will process a chunk of PMIDs
    """This function process a list of chunk containing diverse PMIDs
    Description :
            This function will use mutliprocessing to process a huge list of PMIDs by splitting it in different chunks and create different process for each chunk.
    Args :
            Nor args
    Return : 
            No return
    """
    global EMBL_pmids
    global Sites
    manager=Manager()
    q=manager.Queue()
    pool=Pool(cpu_count()+2)
    for i in tqdm.tqdm(pool.imap_unordered(process,PMIDs),total=len(PMIDs)):
        EMBL_pmids+=i[0]
        for si in i[1]:
            Sites[si]+=i[1][si]
    q.put(str(EMBL_pmids))
    q.put('kill')
    pool.close()
    pool.join()

def process(sublist): #### Extract PMIDs from a sublist
    """This function extract EMBL pmid thanks to the affiliation and the algorithm to detect EMBL affiliation (is_EMBL())
    Description : 
            This function will first make a POST request to the EuropePMC's REST API, this request is based on sublist of PMIDs gave as argument.
            Then the function goes through all results and for each PMID go through each affiliation and predict if it's an EMBL affiliation or not.
            If the PMID contains an EMBL affiliation in the end it returns a list of PMIDs affiliated to EMBL and corresponding sites.
    Args : 
            sublist (list) : 
                    A list of pmid in this format : ['24929366', '28316114', '26078129']
    Return :
            affiliated (list) :
                    A list of PMIDs affiliated to EMBL
            sub_sites(dict) :
                    A dictionnary of EMBL Sites and their associated (detected) PMIDs
    """
    sub_sites={
        "EMBL Australia":[],
        "EMBL Barcelona":[],
        "EMBL-EBI":[],
        "EMBL Grenoble":[],
        "EMBL Hamburg":[],
        "EMBL Heidelberg":[],
        "EMBL Nordic":[],
        "EMBL Rome":[]}
    affiliated=[]
    postm={
        "query":"EXT_ID:"+str(" OR EXT_ID:".join(sublist)),
        "resultType":"core",
        "pageSize":1000,
        "format":"json"}
    url="https://www.ebi.ac.uk/europepmc/webservices/rest/searchPOST"
    req=requests.post(url,data=postm)
    query=json.loads(req.text)
    for result in query["resultList"]["result"]:
        if "pmid" in result:
            if str(result["pmid"]) in sublist:
                aff=False
                PMID_sites={
                    "EMBL Australia":False,
                    "EMBL Barcelona":False,
                    "EMBL-EBI":False,
                    "EMBL Grenoble":False,
                    "EMBL Hamburg":False,
                    "EMBL Heidelberg":False,
                    "EMBL Nordic":False,
                    "EMBL Rome":False}
                pmid=result["pmid"]
                try:
                    for author in result["authorList"]["author"]:
                        try:
                            if "affiliation" in author: # OLD VERSION of affiliation within EuropePMC (one affiliation/author)
                                is_embl=is_EMBL(author["affiliation"],site=True,proba=True)
                                if is_embl["choose"]:
                                    aff=True
                                    PMID_sites[is_embl["site"]]=True
                            elif "authorAffiliationDetailsList" in author: # NEW VERSION of multiple affiliations/author within EuropePMC
                                for aff in author["authorAffiliationDetailsList"]["authorAffiliation"]:
                                    is_embl=is_EMBL(aff["affiliation"],site=True,proba=True)
                                    if is_embl["choose"]:
                                        aff=True
                                        PMID_sites[is_embl["site"]]=True
                        except KeyError:
                            continue
                except (KeyError, TypeError, IndexError) as error:
                    print(str(error))
                    pass
                if aff==True:
                    affiliated.append(pmid)
                for si in PMID_sites:
                    if PMID_sites[si]:
                        sub_sites[si].append(pmid)
    return affiliated,sub_sites

def is_EMBL(request,site=False,proba=False): #### Predict if request is EMBL and the site or not
    """This function will predict if the affiliation is EMBL or not and return some information about the prediction
    Description :
            This function will first prepare the request (str) for the prediction, then it predict on prepared request.
            If the score is superior to 0.9 then it returns different informations like the result dictionnary, if the proba or the site is asked in arguments it return also thos information.
            But if the probability is below 0.9 but above 0.6 the algorithm check if the request is not splitted by the ";" character. If it is, it will predict on substrings with the treshold=0.9.
            If the EMBL affiliation is still ot predict as "choose" it will check if first words of EMBL (EMBL EBI or European) are present in the string. If it's the case then it will take 6 words after
            this first word and predict on this sequence of 7 words. If it reach 0.9 then results are returned. In the end if it's still not predict as EMBL it check if the full name of EMBL is in the string.
            If results are not returned during the prediction then it return the result of the prediction but as not choose etc..
    Args :
            request (string) : 
                    A string to predict if the string is an EMBL one or not
            site (boolean) : 
                    A boolean corresponding to the result wanted, if the site is wanted, it returns it
            proba (boolean) :
                    A boolean corresponding to the result wanted, if the proba is wanted, it returns it
    Return :
            result (dict) :
                        result={
                            "choose":False,    (boolean)
                            "method":"",       (string)
                            "string":request,  (string)
                            "score_EMBL":0.0,  (float)
                            "site":"",         (string)
                            "score_site":0.0,  (float)
                            "substring":""}    (string)
    """
    patterns=[";","EMBL","EBI","European","European Molecular Biology","European Bioinformatics Institute"]
    # EMBL_obvious=r"[^0-9a-zA-Z]*(EMBL)[^0-9a-zA-Z]*|[^0-9a-zA-Z]*(EBI)[^0-9a-zA-Z]*|[^0-9a-zA-Z]*(European Molecular Biology Laboratory)[^0-9a-zA-Z]*|[^0-9a-zA-Z]*(European Bioinformatics Institute)[^0-9a-zA-Z]*"
    result={
        "choose":False,
        "method":"",
        "string":request}
    for old, new in replacements: # String preparation
        request=re.sub(old,new,request)
    X_test_tfidf=EMBL_ID_Vecto.transform([request])
    y_pred=EMBL_ID_clf.predict_proba(X_test_tfidf)
    ## Default value score & site
    if proba:
        result["score_EMBL"]=y_pred[0][1]
    if site:
        X_test_tfidf=EMBL_Sites_ID_Vecto.transform([request])
        y_pred_site=EMBL_Sites_ID_clfLR.predict_proba(X_test_tfidf)
        result["site"]=EMBL_sites[numpy.argmax(y_pred_site[0])]
        if proba:
            result["score_site"]=y_pred_site[0][numpy.argmax(y_pred_site[0])]
    ## Proba > 0.9
    if y_pred[0][1]>0.9:
        result["method"]="Complete sentence"
        result["choose"]=True
        if site:
            X_test_tfidf=EMBL_Sites_ID_Vecto.transform([request])
            y_pred_site=EMBL_Sites_ID_clfLR.predict_proba(X_test_tfidf)
            result["site"]=EMBL_sites[numpy.argmax(y_pred_site[0])]
            if proba:
                result["score_site"]=y_pred_site[0][numpy.argmax(y_pred_site[0])]
        if proba:
            result["score_EMBL"]=y_pred[0][1]
        return result
    ## Proba > 0.6 or patterns
    elif y_pred[0][1]>0.6 or any(char for char in patterns):
        if ";" in request: ## split in sub strings
            for aff in request.split(";"):
                is_embl=is_EMBL(aff,site=True,proba=True)
                if is_embl["choose"]:
                    is_embl["method"]="Substring ';'"
                    is_embl["substring"]=aff
                    is_embl["string"]=request
                    if not site:
                        del is_embl["site"]
                        del is_embl["score_site"]
                    if not proba:
                        del is_embl["score_EMBL"]
                    return is_embl
        for patt in ["European","EMBL","EBI"]:
            if patt in request:
                sent=re.findall(r'[\w]+',request)
                indices=[i for i,x in enumerate(sent) if x==patt]
                for indice in indices:
                    limit=6
                    if indice+limit>len(sent):
                        limit=-1
                    else:
                        limit+=indice
                    sub_EU=" ".join(sent[indice:limit])
                    X_test_tfidf=EMBL_ID_Vecto.transform([sub_EU])
                    y_pred=EMBL_ID_clf.predict_proba(X_test_tfidf)
                    if y_pred[0][1]>0.9:
                        result["method"]="Substring '"+patt+"'"
                        result["choose"]=True
                        result["substring"]=sub_EU
                        if site:
                            X_test_tfidf=EMBL_Sites_ID_Vecto.transform([request])
                            y_pred_site=EMBL_Sites_ID_clfLR.predict_proba(X_test_tfidf)
                            result["site"]=EMBL_sites[numpy.argmax(y_pred_site[0])]
                            if proba:
                                result["score_site"]=y_pred_site[0][numpy.argmax(y_pred_site[0])]
                        if proba:
                            result["score_EMBL"]=y_pred[0][1]
                        return result
                    elif "European Bioinformatics Institute" in sub_EU:
                        result["method"]="Substring 'European Bioinformatics Institute'"
                        result["choose"]=True
                        result["substring"]='European Bioinformatics Institute'
                        if site:
                            result["site"]="EMBL-EBI"
                        return result
                    elif "European Molecular Biology Laboratory" in sub_EU:
                        result["method"]="Substring 'European Molecular Biology Laboratory'"
                        result["choose"]=True
                        result["substring"]='European Molecular Biology Laboratory'
                        if site:
                            X_test_tfidf=EMBL_Sites_ID_Vecto.transform([request])
                            y_pred_site=EMBL_Sites_ID_clfLR.predict_proba(X_test_tfidf)
                            result["site"]=EMBL_sites[numpy.argmax(y_pred_site[0])]
                        return result
    return result

def save(file,obj): #### Save object in a txt file
    """This function will save an object into a a text file
    Description :
            Here based on the directory defined in VARIABLES the function will write the object in a text file
    Args :
            file (str) :
                    The file name
            obj (list-str-dict) :
                    An object to save in a file
    Return:
            No return
    """
    with open(directory+file+".txt","w",encoding="utf-8") as f:
        f.write(str(obj))
        f.write("\n")
        f.close()

### !!! Not the best way to check countries NEED IMPROVEMENTS
def get_geoloc_from(request,cities=False,other=False,all_mention=False): #### Extraction of geolocation information from a sentence
    """This function will extract countries/cities or others geolocation information
    Description :
            Here the function will use spacy as Named Entity Recognition (NER) to catch potential countries or cities or even other geolocation information in the request (string).
            Then for each entity detect by spacy if the entitiy as not already been met it will check if the entity is a country or city (if pass as an argument) or other (if pass as an argument).
            In the end the function will return a dictionnary with mention of the countries (if cities and other = False), the cities (if cities = True), other geolocation information (if other = True).
    Args :
            request (string) : 
                    A string to search geolocation inside
            cities (boolean) :
                    This argument is used to know if cities are wanted in the output instead of countries
            other (boolean) :
                    This argument is used to know if other geolocation information are wanted in the output instead of countries or cities
            all_mention (boolean) : 
                    The dictionnary returned will have the number of mention for each geolocation information found
    Return :
            geoloc_dict (dictionary) :
                    This dictionnary will be in the following format (here for a request mentionning 1 time Australia and Botswana and 4 time USA, with cities=False, other=False, all_mention=True) :
                        {
                            "Australia":1,
                            "United States":4,
                            "Botswana"1
                        }  
    """
    
    def check_dict(request,exact_mention,real_name,geoloc_dict): #### Save in the dictionnary the real name instead of the exact mention that could be wrong
        """This function is used save in a dictionary the real name of a city or country instead of the exact mention
        Description :
                This function will save in the dictionary gave as argument the exact mention of he geolocation information and if the argument all_mention is True it will save the number of times it appear in the request(string)
        Args :
                request (string) :
                        A string to search exact_mention in
                exact_mention (string) : 
                        The exact mention of the entity
                real_name (string) :
                        The real name of the entity
                geoloc_dict (dictionary) :
                        A dictionary to store the information
        Return :
                No return
        """
        if all_mention:
            nb_of_mention=len(re.findall(r'^'+exact_mention+'[^a-zA-Z0-9]|[^a-zA-Z0-9]'+exact_mention+'[^a-zA-Z0-9]|[^a-zA-Z0-9]'+exact_mention+'$',request))
        else:
            nb_of_mention=1
        if real_name not in geoloc_dict:
            geoloc_dict[real_name]=nb_of_mention
        else:
            geoloc_dict[real_name]+=nb_of_mention

    def check_in_dict(dico): #### Use to look for key in a dict and then save it if key in it
        """This function will go through a dictionary (dico) to find if there is any ey in the request
        Description :
                Here the function will go through the dictionary pass as an argument to check if any (abreviation/iso2/iso3) are in the request (string).
                If it is, it calls check_dict() function to save the real name (value) based on the exact mention (key)
        Args : 
                dico (dictionary) :
                        A dictionary with a pattern as a key and real name as value
        Return :
                No return
        """
        for abrev in dico:
            if abrev not in geoloc_met and re.search(r'^'+abrev+'[^a-zA-Z0-9]|[^a-zA-Z0-9]'+abrev+'[^a-zA-Z0-9]|[^a-zA-Z0-9]'+abrev+'$',request):
                check_dict(request,abrev,dico[abrev],geoloc_dict)
                geoloc_met.append(abrev)
                if not all_mention:
                    return

    geoloc={
        "Countries":countries_list,
        "Cities":cities_list,
        "iso2":countries_iso2,
        "iso3":countries_iso3}
    geoloc_dict={}
    if cities:
        geoloc_geoname=geoloc["Cities"]
    else:
        geoloc_geoname=geoloc["Countries"]
    geoloc_met=[]
    geoloc_info=[]
    spacy_aff=nlp(request)
    
    for entities in spacy_aff.ents:          
        if entities.label_ =='GPE' and entities.text not in geoloc_met and not re.search(r'[^0-9a-zA-Z\s]',entities.text):
            geoloc_info.append(entities.text)
            if other and entities.text not in geoloc["Countries"] and entities.text not in geoloc["Cities"] and entities.text not in geoloc["iso2"] and entities.text not in geoloc["iso3"]:
                check_dict(request,entities.text,entities.text,geoloc_dict)
                geoloc_met.append(entities.text)
            elif not other:
                entity=entities.text[0].upper()+entities.text[1:].lower()
                if entities.text in geoloc_geoname:
                    check_dict(request,entities.text,entities.text,geoloc_dict)
                    geoloc_met.append(entities.text)
                elif entity in geoloc_geoname:
                    check_dict(request,entities.text,entity,geoloc_dict)
                    geoloc_met.append(entities.text)
                else:
                    for w in entities.text.split():
                        if w in geoloc_geoname:
                            check_dict(request,w,w,geoloc_dict)
                            geoloc_met.append(w)
    
    if not cities and not other:
        check_in_dict(abrevs)
        # check_in_dict(countries_iso3)
        # check_in_dict(countries_iso2)

    if cities and len(geoloc_met)<1:
        for w in re.findall(r'[\w]+',request):
            if w in geoloc["Cities"]:
                check_dict(request,w,w,geoloc_dict)
                geoloc_met.append(w)
    
    if not cities and not other:
        if len(geoloc_met)>0 and not all_mention:
            return geoloc_dict
        for info in geoloc_info:
            if info.lower().title() in countries_list:
                check_dict(request,info,info.lower().title(),geoloc_dict)
                geoloc_met.append(info)
                return geoloc_dict
        for w in re.findall(r'[\w]+',request):
            if w in geoloc_geoname:
                check_dict(request,w,w,geoloc_dict)
                geoloc_met.append(w)
        if len(geoloc_met)>0 and not all_mention:
            return geoloc_dict
    return geoloc_dict

#############################                   MAIN                   #############################

if __name__=='__main__':
    abrevs={ ### Dictionary of countries abreviations often met 
        'UK':'United Kingdom',
        'USA':'United States',
        "US":'United States',
        'Czech':"Czech Republic"}
    gc=geonamescache.GeonamesCache() # load data from geonamescache
    countries_list = [*gen_list_extract(gc.get_countries(), 'name')] # Creation of Countries list
    countries_iso2 = {}
    countries_iso3 = {}
    countries_to_iso2 = {}
    for country in pycountry.countries:
        countries_to_iso2[country.name] = country.alpha_2
        countries_iso2[country.alpha_2] = country.name
        countries_iso3[country.alpha_3] = country.name
    cities_list = [*gen_list_extract(gc.get_cities(), 'name')] # Creation of Cities list
    nlp = spacy.load("en_core_web_sm") # Model from spacy for NER (Name Entity Recognition)
    Sites={
        "EMBL Australia":[],
        "EMBL Barcelona":[],
        "EMBL-EBI":[],
        "EMBL Grenoble":[],
        "EMBL Hamburg":[],
        "EMBL Heidelberg":[],
        "EMBL Nordic":[],
        "EMBL Rome":[]}
    # File reading
    if ".txt" in search_file or ".csv" in search_file:
        with open(directory+search_file,"r",encoding="utf-8") as file:
            PMID_list=re.findall(r'([0-9]+)',file.read())
            file.close()
    else:
        PMID_list=re.findall(r'([0-9]+)',search_file)
    PMIDs=chunkIt(PMID_list,len(PMID_list)/1000)
    EMBL_pmids=[]
    start=time.time()
    main()
    print("Number of PMIDs to process :"+str(len(PMID_list)))
    end=time.time()
    print("Computing time: "+str(end-start))
    print("Number of EMBL publications found: "+str(len(EMBL_pmids)))
    for si in Sites:
        file=si.replace(" ","_").replace("-","_")
        x=0
        NA=[]
        PMID_EMBL_only=[]
        PMID_member_states=[]
        PMID_others=[]
        PMID_deleted=[]
        RESULTS_PMIDS={}
        for PMID in Sites[si]:
            RESULTS_PMIDS[PMID]={
                "EMBL":True,
                "Member states":False,
                "Worldwide":False,
                "Partnership":False}
            x+=1
            req=requests.get("https://www.ebi.ac.uk/europepmc/webservices/rest/search/query=ext_id:"+PMID+"&resultType=core&format=json")
            try:
                req_PMID=json.loads(req.text)["resultList"]["result"][0]["authorList"]["author"]
            except IndexError:
                PMID_deleted.append(PMID)
                continue
            for result in req_PMID:
                if PMID==result:
                    req_PMID=result["authorList"]["author"]
                    break
            affiliations=[]
            for author in req_PMID:
                try:
                    if "affiliation" in author:
                        affiliations.append(author["affiliation"])
                    elif "authorAffiliationDetailsList" in author:
                        for aff in author["authorAffiliationDetailsList"]["authorAffiliation"]:
                            affiliations.append(aff["affiliation"])
                except KeyError:
                    continue
            nb_EMBL=0
            EMBL_aff=[]
            for affiliation in affiliations:
                is_embl=is_EMBL(affiliation,site=True,proba=True)
                if is_embl["choose"]:
                    if is_embl["site"] == "EMBL Nordic" or is_embl["site"] == "EMBL Australia":
                        RESULTS_PMIDS[PMID]["Partnership"]=True
                else:
                    C=get_geoloc_from(affiliation)
                    if any(country in C for country in member_states+associate_member_states):
                        RESULTS_PMIDS[PMID]["Member states"]=True
                    else:
                        RESULTS_PMIDS[PMID]["Worldwide"]=True
            if RESULTS_PMIDS[PMID]["Member states"]:
                PMID_member_states.append(PMID)
            elif RESULTS_PMIDS[PMID]["Worldwide"]:
                PMID_others.append(PMID)
            elif RESULTS_PMIDS[PMID]["Partnership"]:
                NA.append(PMID)

        with open("./searches/"+search_name+"/"+file+"_categories.csv","w",encoding="utf-8") as f:
            f.write("PMID\tEMBL\tMember states\tWorldwide\tPartnership\n")
            for pmid in RESULTS_PMIDS:
                f.write(str(pmid))
                f.write("\t")
                f.write(str(RESULTS_PMIDS[pmid]["EMBL"]))
                f.write("\t")
                f.write(str(RESULTS_PMIDS[pmid]["Member states"]))
                f.write("\t")
                f.write(str(RESULTS_PMIDS[pmid]["Worldwide"]))
                f.write("\t")
                f.write(str(RESULTS_PMIDS[pmid]["Partnership"]))
                f.write("\n")
    save("EMBL_PMIDs",EMBL_pmids)