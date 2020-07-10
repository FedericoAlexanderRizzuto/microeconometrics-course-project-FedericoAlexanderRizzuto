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

def sargan(self):
        """
        Sargan test of overidentifying restrictions

        Returns
        -------
        t : WaldTestStatistic
            Object containing test statistic, p-value, distribution and null

        Notes
        -----
        Requires more instruments than endogenous variables

        Tests the ratio of re-projected IV regression residual variance to
        variance of the IV residuals.

        .. math ::

          n(1-\hat{\epsilon}^{\prime}M_{Z}\hat{\epsilon}/
          \hat{\epsilon}^{\prime}\hat{\epsilon})\sim\chi_{v}^{2}

        where :math:`M_{z}` is the annihilator matrix where z is the set of
        instruments and :math:`\hat{\epsilon}` are the residuals from the IV
        estimator.  The degree of freedom is the difference between the number
        of instruments and the number of endogenous regressors.

        .. math ::

          v = n_{instr} - n_{exog}
        """
        z = self.model.instruments.ndarray
        nobs, ninstr = z.shape
        nendog = self.model.endog.shape[1]
        name = 'Sargan\'s test of overidentification'
        if ninstr - nendog == 0:
            return InvalidTestStatistic('Test requires more instruments than '
                                        'endogenous variables.', name=name)

        eps = self.resids.values[:, None]
        u = annihilate(eps, self.model._z)
        stat = nobs * (1 - (u.T @ u) / (eps.T @ eps)).squeeze()
        null = 'The model is not overidentified.'

        return WaldTestStatistic(stat, null, ninstr - nendog, name=name)
    
