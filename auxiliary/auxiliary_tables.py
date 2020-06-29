# -*- coding: utf-8 -*-
"""
This module contains auxiliary functions for the creation of tables in the main notebook.
"""

import numpy as np
import pandas as pd
import statsmodels as sm
import statsmodels.formula.api as smf
from linearmodels import IV2SLS

from auxiliary.auxiliary_regressions import *
from auxiliary.auxiliary_plots import *

def prepare_data(df):
    df['north_center'] = 1
    df.loc[df.south == 1, 'north_center'] = 2
    
    return df

def prepare_data_table1(df):
    df['placeholder'] = 0
    var_cl = df[['grade','north_center','classid','m_female','m_origin',
                        'm_dad_edu','m_mom_occ','female','immigrants_broad',
                        'dad_midedu','mom_employed','answers_math_pct',
                        'answers_ital_pct','clsize_snv','our_CHEAT_math',
                        'our_CHEAT_ital','schoolid','survey','plessoid',
                        'enrol_sch_snv','enrol_ins_snv','o_math',
                        'placeholder']]
    pd.set_option('mode.chained_assignment', None)
    var_cl.loc[var_cl.m_female != 0, 'female'] = np.nan
    var_cl.loc[var_cl.m_origin != 0, 'immigrants_broad'] = np.nan
    var_cl.loc[var_cl.m_mom_occ != 0, 'mom_employed'] = np.nan
    var_cl.loc[var_cl.m_dad_edu != 0, 'dad_midedu'] = np.nan
    var_sch = var_cl.groupby(['north_center','schoolid','survey','plessoid','grade','enrol_sch_snv','placeholder'],as_index=False)['classid'].count()
    var_ins = var_cl.groupby(['north_center','schoolid','survey','grade','enrol_ins_snv','o_math','placeholder'],as_index=False)['classid'].count()
    
    return var_cl, var_sch, var_ins

def format_table1(reg,nreg,it,nit):
    table_reg = pd.concat([reg,nreg], axis=1, sort=False)
    table_it = pd.concat([it,nit], axis=1, sort= False)
    table = pd.concat([table_reg,table_it], axis = 0, sort = False)
    table = table.transpose()
    table.columns.set_levels(['Grade 5','Grade 2'],level=0,inplace=True)
    table.columns.set_levels(['Italy','North/Center','South'],level=1,inplace=True)
    table = table.reindex(sorted(table.columns),axis=1).round(2)
    table.rename(index={'female':'Female*','immigrants_broad':'Immigrant*','dad_midedu':'Father HS*',
                          'mom_employed':'Mother employed*','answers_math_pct':'Pct correct: Math',
                          'answers_ital_pct':'Pct correct: Language','clsize_snv':'Class size',
                          'our_CHEAT_math':'Score manipulation: Math',
                          'our_CHEAT_ital':'Score manipulation: Language',
                          'classid':'Number of classes','enrol_sch_snv':'Enrollment',
                          'schoolid':'Number of schools','enrol_ins_snv':'Enrollment',
                          'o_math':'External monitor'},inplace=True)
    
    return table
    
def create_table1a(df):
    var_cl, var_sch, var_ins = prepare_data_table1(df)
    grouped = ['female','immigrants_broad','dad_midedu','mom_employed','answers_math_pct',
            'answers_ital_pct','clsize_snv','our_CHEAT_math','our_CHEAT_ital']
    t1a_reg = var_cl.groupby(['grade','north_center'])[grouped].agg(['mean','std'])
    ncl_reg = df[~df.isnull().any(axis=1)].groupby(['grade','north_center'])[['classid']].agg(['count'])
    t1a_it = var_cl.groupby(['grade','placeholder'])[grouped].agg(['mean','std'])
    ncl_it = df[~df.isnull().any(axis=1)].groupby(['grade','placeholder'])[['classid']].agg(['count'])
    table1a = format_table1(t1a_reg,ncl_reg,t1a_it,ncl_it)
    
    return table1a

def create_table1b(df):
    var_cl, var_sch, var_ins = prepare_data_table1(df)
    t1b_reg = var_sch.groupby(['grade','north_center'])[['classid','enrol_sch_snv']].agg(['mean','std'])
    nsch_reg = var_sch.groupby(['grade','north_center'])[['schoolid']].agg(['count'])
    t1b_it = var_sch.groupby(['grade','placeholder'])[['classid','enrol_sch_snv']].agg(['mean','std'])
    nsch_it = var_sch.groupby(['grade','placeholder'])[['schoolid']].agg(['count'])
    table1b = format_table1(t1b_reg,nsch_reg,t1b_it,nsch_it)
    
    return table1b
   
def create_table1c(df):
    var_cl, var_sch, var_ins = prepare_data_table1(df)
    t1c_reg = var_ins.groupby(['grade','north_center'])[['schoolid','classid','enrol_ins_snv','o_math']].agg(['mean','std'])
    nins_reg = var_ins.groupby(['grade','north_center'])[['schoolid']].agg(['count'])
    t1c_it = var_ins.groupby(['grade','placeholder'])[['schoolid','classid','enrol_ins_snv','o_math']].agg(['mean','std'])
    nins_it = var_ins.groupby(['grade','placeholder'])[['schoolid']].agg(['count'])    
    table1c = format_table1(t1c_reg,nins_reg,t1c_it,nins_it)
    
    return table1c

def create_table2and3(df,outcomes,panels):
    df['clsize_ols'] = df['clsize_snv'] / 10
    df_nc = df.loc[df.north_center == 1]
    df_nc.region = df_nc.region.cat.remove_unused_categories()
    df_s = df.loc[df.north_center == 2]
    df_s.region = df_s.region.cat.remove_unused_categories()
    datasets = [df, df_nc, df_s]
    model = [smf.ols,IV2SLS.from_formula,IV2SLS.from_formula]
    outcomes = df[outcomes]
    output = []
    columns= pd.MultiIndex.from_product([['OLS', 'IV/2SLS',' IV/2SLS'],
                                         ['Italy', 'North/Center', 'South']])
    table = pd.DataFrame()
    for i, outcome in enumerate(outcomes):
        clmn = 0
        idx = pd.MultiIndex.from_product([[panels[i]],
                                          ['Class size - coefficient', 
                                           'Class size - standard errors', 'Enrollment',
                                           'Enrollment squared','Interactions',
                                           'Observations']])
        newtable = pd.DataFrame(index=idx,columns=columns)
        formula1 = outcome +'~ clsize_ols + female+ m_female+ immigrants_broad+ m_origin+ dad_lowedu+ dad_midedu+ dad_highedu+ mom_unemp+ mom_housew+ mom_employed+ m_mom_edu + students+ students2 + C(survey) + C(grade) + enrol_ins_snv*region'
        formula2 = outcome +'~ [clsize_ols ~ clsize_hat] + female+ m_female+ immigrants_broad+ m_origin+ dad_lowedu+ dad_midedu+ dad_highedu+ mom_unemp+ mom_housew+ mom_employed+ m_mom_edu + students+ students2 + C(survey) + C(grade) + enrol_ins_snv*region'
        formula3 = outcome +'~ [clsize_ols ~ clsize_hat] + female+ m_female+ immigrants_broad+ m_origin+ dad_lowedu+ dad_midedu+ dad_highedu+ mom_unemp+ mom_housew+ mom_employed+ m_mom_edu + C(survey) + C(grade) + enrol_ins_snv*region + students*C(segment) + students2*C(segment)'
        formulas = [formula1,formula2,formula3]
        for j, formula in enumerate(formulas):
            for k, data in enumerate(datasets):
                data = data[data[outcome].notna()]
                cov_type = 'cluster'
                cov_kwds = {'groups': data['clu']}
                reg_res = summ_reg(data, model[j], formula, cov_type, cov_kwds)
                this_column = newtable.columns[clmn]
                newtable[this_column] = reg_res
                clmn = clmn + 1
        table = table.append(newtable,ignore_index=False)
        
    return table