import scipy
import numpy as np

def get_groups(dataframe, group_col, value_col):

    groups = []
    normality_pvals = []
    for val in dataframe[group_col].unique():
        g = dataframe[dataframe[group_col] == val][value_col]
        groups.append(g)
        _, p = (scipy.stats.shapiro(g) if len(g) <= 2500 
                else scipy.stats.normaltest(g))
        normality_pvals.append(p)
    return groups, np.array(normality_pvals)

def select_group_test(groups, normality_pvals, alpha):

    is_normal = bool((normality_pvals > alpha).all())
    n = len(groups)
    
    if n == 2:
        if is_normal and min(len(g) for g in groups) > 30:
            _, p = scipy.stats.ttest_ind(*groups, equal_var=False)
        else:
            _, p = scipy.stats.mannwhitneyu(*groups, alternative='two-sided')
    elif n >= 3:
        if is_normal and min(len(g) for g in groups) > 30:
            _, p_levene = scipy.stats.levene(*groups)
            if p_levene > 0.05:
                _, p = scipy.stats.f_oneway(*groups)
            else:
                _, p = scipy.stats.kruskal(*groups)
        else:
            _, p = scipy.stats.kruskal(*groups)
    return p