#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 18 04:38:29 2018

@author: dt
"""
import random
import os
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances
from dictvectorizer_reader import *
import numpy as np
from scipy.sparse import csr_matrix
from tqdm import tqdm
from sklearn.decomposition import LatentDirichletAllocation
import time
import pickle

random.seed(1229)

# high frequency word
N = None
new_total = 3000
alpha = 0.2
shrinked = shrink_corpus(new_total)
wc = get_wordcount_list(shrinked, nlim=N)

pub_relation = get_pubdate_relation(wc)
pubyear = pd.DataFrame(pub_relation)

tf = get_counter(wc)
print('tf:',tf.shape)
# lda weight

t0=time.time()
if os.path.isfile('doc_topic_distr2.pkl'): # use cache
    doc_topic_distr = pickle.load(open('doc_topic_distr2.pkl', 'rb'))
else:
    vec = DictVectorizer()
    tc = vec.fit_transform(iter_wf(wc, normalize=False))

    lda = LatentDirichletAllocation(n_components=500,random_state=1,max_iter=5)
    doc_topic_distr = lda.fit_transform(tc)
    pickle.dump(doc_topic_distr, open('doc_topic_distr2.pkl', 'wb'))
    
print('done in %0.3fs.' % (time.time()-t0), '\a')
print('doc_topic_distr:',doc_topic_distr.shape)

#tflda = np.concatenate((tf, doc_topic_distr), axis=1)
tflda = np.hstack((tf, doc_topic_distr))

D = euclidean_distances(tflda)
del tflda
print('Euclidean dist finish...')


for i,r in enumerate(D):
    rf = r[r>0]
#    r[r > rf.std()] = 0
    r[r > alpha * rf.std()] = 0
    D[i] = r

D=np.triu(D)

print('Filtering finish...')

s = csr_matrix(D)
del D
row, col = s.nonzero()
print('Sparse matrix finish...')


df = pd.DataFrame({'Source':row,'Target':col,'Weight':s.data})
print('df finish...')
#df['Id'] = df.index

nlim = len(shrinked)
print('Edge keep number:', len(df))
print('Edge keep pct:', len(df)/((nlim-2)*(nlim-1)/2))
print('Distance range:', min(df.Weight), '-', max(df.Weight))
print('Average:', sum(df.Weight)/len(df.Weight))

#ans=input('proceed to save to csv? (y/n)')
ans='n'
if ans == 'y':
    pubyear.to_csv('gephi/Node_{}_{}.csv'.format(new_total, alpha), index=False)
    df['Weight'] = df['Weight'].apply(lambda x: '{:0.5f}'.format(x))
    df.to_csv('gephi/Edges_{}.csv'.format(new_total),index=False)

id_year = {x:y for x,y in zip(pub_relation['id'], pub_relation['year'])}
df.Source = df.Source.apply(lambda x: id_year[x])
df.Target = df.Target.apply(lambda x: id_year[x])
df['year_diff'] = df.Target - df.Source
df = df[df.year_diff > 0]

import statsmodels.api as sm
X = sm.add_constant(df['year_diff'])
model = sm.OLS(df['Weight'], X).fit()
print(model.summary())



