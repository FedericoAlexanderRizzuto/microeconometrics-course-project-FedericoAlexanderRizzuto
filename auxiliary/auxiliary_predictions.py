# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 23:35:07 2020

@author: FedericoAlexander
"""
import numpy as np
import pandas as pd
import statsmodels as sm
import statsmodels.formula.api as smf
from linearmodels import IV2SLS

from auxiliary_plots import *
from auxiliary_tables import *

def get_coeff_se(data,model,formula,cov_type,cov_kwds):
    if model == smf.ols:
        result = model(formula,data).fit(cov_type=cov_type, cov_kwds=cov_kwds)
        coeff = result.params['clsize_ols']
        se = result.bse['clsize_ols']
    elif model == IV2SLS.from_formula:
        result = model(formula,data).fit()
        coeff = result.params['clsize_ols']
        se = result.std_errors['clsize_ols']
    else:
        print('Function only for smf.ols and IV2SLS.from_formula')
    return np.array([[coeff], [se]])