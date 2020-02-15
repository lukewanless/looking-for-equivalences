from itertools import combinations
from tqdm import tqdm
import numpy as np


def get_ecdf(series_):
    return lambda x: (series_.sort_values() < x).astype(int).mean()

def get_ks_diff(boots, bound=10):
    n, m =  boots.shape
    ks = np.full((n,n), np.nan)
    combs = list(combinations(range(n), 2))

    for i,j in tqdm(combs):
        ecdf1 =  get_ecdf(boots.iloc[i])
        ecdf2 =  get_ecdf(boots.iloc[j])
        ecdf1 = np.vectorize(ecdf1)
        ecdf2 = np.vectorize(ecdf2)    
        x = np.linspace(-bound, bound, m)
        y1 = ecdf1(x)
        y2 = ecdf2(x)
        diff = np.max(np.abs(y1 - y2)) 
        ks[j,i] = diff

    for i in range(n):
        ks[i,i] = 0
        
    return ks