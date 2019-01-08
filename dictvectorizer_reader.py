#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 00:21:26 2018

@author: dt
"""

from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from tqdm import tqdm
import pandas as pd
from glob import glob
import random
import numpy as np
#os.chdir('/Volumes/SD/UChicago/Literature/9/')

def iter_wf(wordcount_list, normalize=True):
#    count_not_exist = 0
    
    for fn in wordcount_list:
#        if not os.path.exists(fn):
#            count_not_exist += 1
#            continue

        series = pd.read_csv(fn,index_col=0)['WEIGHT']
        series = series[~series.index.isnull()]
        sum_weight = sum(series)
        if not sum_weight:
            continue
        if normalize:
            yield series/sum_weight
        else:
            yield series

def path_to_id(fn):
    return fn.split('wordcounts_')[1].replace('.CSV','').replace('_','/')

def get_wordcount_list(shrinked=None, nlim=100):
    wordcount_list = glob('raw_data_quiet_trans/*/wordcounts/*.CSV') + glob('raw_data_quiet_trans/*/*/wordcounts/*.CSV')
    if nlim:
        wordcount_list = wordcount_list[:nlim]
    if shrinked is not None:
        wordcount_list = list(filter(lambda fn: path_to_id(fn) in shrinked, wordcount_list))
    return wordcount_list 
   
def get_pubdate_relation(wordcount_list):
    pubyears = []
    pubids = []
    citations = get_citation_info()
    for i, fn in enumerate(wordcount_list):
        fnid = path_to_id(fn)
        pubyears.append(citations[fnid])
        pubids.append(i)
    print('pubdate relation done!')
    return {'id':pubids, 'year':pubyears}
    
def get_tf(wordcount_list): # output array. row: Doc, col: term
    vec = DictVectorizer()
    tf = vec.fit_transform(iter_wf(wordcount_list))
    return tf

def iter_wf_to_passage(wordcount_list):
    for fn in wordcount_list:
        series = pd.read_csv(fn,index_col=0)['WEIGHT']
        series = series[~series.index.isnull()]
        sum_weight = sum(series)
        if not sum_weight:
            continue
        
#    for i,fn in tqdm(enumerate(wordcount_list)):
#        wcdf = pd.read_csv(fn)
#        sum_weight = sum(series)
#        if not sum_weight:
#            continue

        yield ''.join([(str(word)+' ')*count for word, count in series.iteritems()])

def get_counter(wordcount_list, n_feature=None): 
    vec = CountVectorizer(max_df=0.98, min_df=0.05)
    tf = vec.fit_transform(iter_wf_to_passage(wordcount_list))
    tf /= np.sum(tf, axis=0)
    return tf

def get_citation_info():
    cit_list = glob('raw_data_quiet_trans/*/citations.CSV') + glob('raw_data_quiet_trans/*/*/citations.CSV')
    citations = pd.DataFrame({})
    for fn in cit_list:
        df = pd.read_csv(fn, index_col=False)[['id','pubdate']]
        df['pubdate'] = df['pubdate'].apply(lambda x: int(x.split('-')[0]))
        citations = pd.concat([citations, df])
    
    citations = citations.set_index('id')
    return citations['pubdate']

def shrink_corpus(new_total=10000):
    cites = get_citation_info()
    count = cites.groupby(cites).count()
    total = sum(count)
    keep = round(count/total*new_total,0).astype(int)
    shrinked = pd.Series({},name='pubdate')
    for year in cites.drop_duplicates().sort_values():
        new_selection = list(range(count[year]))
        random.shuffle(new_selection)
        new_selection = sorted(new_selection[:keep[year]])
        year_slice = cites[cites==year]
        shrinked = shrinked.append(year_slice.iloc[new_selection])
    return shrinked
        