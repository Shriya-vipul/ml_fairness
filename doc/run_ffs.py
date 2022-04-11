#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 10 00:48:34 2022

@author: christiedu
"""

import pandas as pd
import numpy as np
import sys

sys.path.insert(1, '../lib')
from ffs_utils import compute_shapley

print("Reading in data..........")

compas = pd.read_csv("../output/ffs_data_v3.csv",
                     header=0,
                     dtype = {'age_cat': int,'race': int,
                              'priors_count': int,'length_of_stay': int})

# data processing
print("Processing data..........")

compas['c_charge_degree'] = pd.get_dummies(compas['c_charge_degree'])['F'] # 1 if felony, 0 if misdemeanor
compas['sex'] = pd.get_dummies(compas['sex'])['Male'] # 1 if male, 0 if female

juv_cond = (compas['juv_fel_count'] > 0) | (compas['juv_misd_count'] > 0) | (compas['juv_other_count'] > 0)
compas['has_juv'] = np.where(juv_cond, 1, 0)
compas['has_prior'] = np.where(compas['priors_count'] > 0, 1, 0)

X = compas[['age', 'c_charge_degree', 'sex', 'length_of_stay', 
            'has_juv', 'has_prior']] # 6 features
y = compas['two_year_recid']
protected = compas['race']

# get shapley scores

print("Features:", X.columns)

print("Computing Shapley Scores..........")
ffs_df = compute_shapley(X.to_numpy(),
                         y.to_numpy(),
                         protected.to_numpy(),
                         X.columns.tolist())

print("Saving FFS scores..........")
# saving ffs DF
ffs_df.to_csv("../output/ffs_scores.csv", header=True, index=False)