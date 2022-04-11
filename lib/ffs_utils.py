#!/usr/bin/env python
# coding: utf-8

from itertools import chain, combinations, product
import numpy as np
import pandas as pd
import math
import time


def powerset(iterable):
    """
    Source: https://stackoverflow.com/questions/1482308/how-to-get-all-subsets-of-a-set-powerset
    powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)
    """
    
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))


def unique_per_col(arr):
    val_list = []
    for col in arr.T:     
        val_list.append(np.unique(col))
        
    return val_list


def compute_shapley(X, y, protected, features):
    """
    Computes accuracy and discrimination scores for features
    Arguments:
        X: DF with features to be selected
        y: labels
        protected: protected feature (ex: race)
    Returns:
        DataFrame of features in X and computed acc and disc scores
    """
    
    scores_dict = {'Feature': features, 'Accuracy': [], 'Discrimination': []}
    
    for i in range(X.shape[1]):
        start = time.time()
        print("starting feature", i)
        acc_tmp, disc_tmp = compute_acc_disc(X, y, protected, i)

        print(acc_tmp, disc_tmp)
        scores_dict['Accuracy'].append(acc_tmp)
        scores_dict['Discrimination'].append(disc_tmp)
        
        print("\ntime:", time.time()-start)
    
    assert len(scores_dict['Feature']) == len(scores_dict['Accuracy']), 'Feature and Accuracy lengths unequal!'
    assert len(scores_dict['Feature']) == len(scores_dict['Discrimination']), 'Feature and Discrimination lengths unequal!'
    
    return pd.DataFrame.from_dict(scores_dict)


def compute_acc_disc(X, y, protected, i):
    """
    Computes accuracy + discrimination score for given feature
    as proposed by Definition 5 in FFS paper.
    Arguments:
        X: DF with features to be selected
        y: labels
        protected: protected feature (ex: race)
        i: feature index
    Returns:
        Shapley accuracy score
    """

    num_features = X.shape[1]
    idx = list(range(num_features))
    del idx[i]
    subsets = [list(x) for x in powerset(idx) if len(x) > 0] # exclude the empty set

    acc_shapley = disc_shapley = 0
    for subset in subsets:
        
        print(".", end="")
        # See Definition 5 in FFS paper

        # leading coefficient
        subset_len = len(subset)
        fact1 = math.factorial(subset_len)
        fact2 = math.factorial(num_features - subset_len - 1)
        fact3 = math.factorial(num_features)
        coef = fact1 * fact2 / fact3
        
        # v(T U {i}): includes the feature
        Xs_incl_idx = subset[:]
        Xs_incl_idx.append(i)
        Xsc_incl_idx = list(set(range(num_features)) - (set(Xs_incl_idx)))
        acc_incl = v_acc(X[:, Xs_incl_idx],
                                X[:, Xsc_incl_idx],
                                y.reshape(-1, 1),
                                protected.reshape(-1, 1))
        
        disc_incl = v_d(X[:, Xs_incl_idx],
                                  y.reshape(-1, 1), 
                                  protected.reshape(-1, 1))
        
        # v(T): excludes the feature
        Xsc_excl_idx = list(set(idx) - (set(subset)))
        acc_excl = v_acc(X[:, subset], 
                                X[:, Xsc_excl_idx],
                                y.reshape(-1, 1), 
                                protected.reshape(-1, 1))
        
        disc_excl = v_d(X[:, subset],
                                  y.reshape(-1, 1), 
                                  protected.reshape(-1, 1))
        
        # acc
        acc_marginal = acc_incl - acc_excl
        acc_shapley += coef * acc_marginal
        
        #disc
        disc_marginal = disc_incl - disc_excl
        disc_shapley += coef * disc_marginal
    
    return acc_shapley, disc_shapley


def v_acc(Xs, Xsc, y, protected):
    cond = np.concatenate((Xsc, protected), axis=1)
    I = comp_cond_dep(y, Xs, cond)
    return I


def v_d(Xs, y, protected):
    Xsa = np.concatenate((Xs, protected), axis=1)
    SI = comp_dep(y, Xsa)
    I1 = comp_dep(Xs, protected)
    I2 = comp_cond_dep(Xs, protected, y)
    return SI * I1 * I2


def comp_dep(left, right):
    
    #start = time.time()

    num_rows = left.shape[0]
    num_left_cols = left.shape[1]
        
    concat_arr = np.concatenate((left, right), axis=1)
    concat_unique = unique_per_col(concat_arr)

    concat_cart = list(product(*concat_unique))
    p_total = 0
    for vec in concat_cart:
        p_r1_r2 = len(np.where((concat_arr == vec).all(axis=1))[0]) / num_rows
        
        if p_r1_r2 == 0:
            p_event = 0
        else:
            p_r1 = len(np.where((left == vec[:num_left_cols]).all(axis=1))[0]) / num_rows
            if p_r1 == 0:
                p_event = 0
            else:
                p_r2 = len(np.where((right == vec[num_left_cols:]).all(axis=1))[0]) / num_rows
                
                if p_r2 == 0:
                    p_event = 0
                else:
                    p_event = p_r1_r2 * np.log(p_r1_r2 / p_r1) / p_r1

        p_total += np.abs(p_event)
        
    #print("dep:", (time.time() - start) / len(concat_cart))
    return p_total


def comp_cond_dep(left, right, conditional):

    #start = time.time()
    
    num_rows = left.shape[0]
    num_left_cols = left.shape[1]
    num_right_cols = right.shape[1]

    concat_arr = np.concatenate((left, right, conditional), axis=1)    
    concat_unique = unique_per_col(concat_arr)
    concat_cart = list(product(*concat_unique))
    p_total = 0

    for vec in concat_cart:
        
        p_r1_r2 = len(np.where((concat_arr == vec).all(axis=1))[0]) / num_rows
        if p_r1_r2 == 0:
            p_event = 0
        else:
            p_r1 = len(np.where((left == vec[:num_left_cols]).all(axis=1))[0]) / num_rows
            if p_r1 == 0:
                p_event = 0
            else:
                num = len(np.where((concat_arr[:, num_left_cols: -num_right_cols] == 
                            vec[num_left_cols: -num_right_cols]).all(axis=1))[0])
                p_r2 = num / num_rows
                if p_r2 == 0:
                    p_event = 0
                else:
                    denom = len(np.where((concat_arr[:, -num_right_cols:] == 
                                          vec[-num_right_cols:]).all(axis=1))[0])
                    if denom != 0:
                        
                        cond1 = (concat_arr[:, :num_left_cols] == vec[:num_left_cols]).all(axis=1)
                        cond2 = (concat_arr[:, -num_right_cols:] == vec[-num_right_cols:]).all(axis=1)
                        p_r1_given_r3 = len(np.where(cond1 & cond2)[0]) / denom
                    
                    else:
                        p_r1_given_r3 = 0
                        
                    if p_r1_given_r3 == 0:
                        p_event = 0
                    else:
                        p_event = p_r1_r2 * np.log(p_r1_r2 / p_r2) / p_r1_given_r3
        
        p_total += np.abs(p_event)

    #print("cond_dep", (time.time() - start) / len(concat_cart))
    return p_total





