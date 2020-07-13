# -*- coding: utf-8 -*-
"""
Date: July 17, 2020
Author: Federico Alexander Rizzuto
Content: Code producing tables needed to replicate Angrist et al. (2017) for the 
Microeconometrics project
"""
#import numpy as np
#import pandas as pd
#import statsmodels as sm
import statsmodels.formula.api as smf
from linearmodels import IV2SLS
from linearmodels.iv._utility import annihilate, proj
from linearmodels.utility import (InvalidTestStatistic, WaldTestStatistic, _ModelComparison,_SummaryStr, _str, pval_format)

from auxiliary.auxiliary_plots import *
from auxiliary.auxiliary_tables import *

def summ_reg(df,model,formula,cov_type,cov_kwds):
    if model == smf.ols:
        result = model(formula,df).fit(cov_type=cov_type, cov_kwds=cov_kwds)
        se = "{:.3f}".format(result.bse['clsize_ols'])
    elif model == IV2SLS.from_formula:
        result = model(formula,df).fit(cov_type='clustered',clusters=df.clu)
        se = "{:.3f}".format(result.std_errors['clsize_ols'])
    else:
        print('Function only for smf.ols and IV2SLS.from_formula')
    nobs = "{:.0f}".format(result.nobs)
    coeff = "{:.3f}".format(result.params['clsize_ols'])
    if 'students' in formula:
        enr = 'X'
    if 'students2' in formula:
        enr2 = 'X'
    if 'students*C(segment)' in formula:
        inter = 'X'
    else:
        inter = ''
    return [coeff,(se),enr,enr2,inter,nobs]
