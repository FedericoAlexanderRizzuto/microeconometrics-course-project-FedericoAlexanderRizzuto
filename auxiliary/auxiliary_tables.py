# -*- coding: utf-8 -*-
"""
Date: July 17, 2020
Author: Federico Alexander Rizzuto
Content: Code producing tables needed to replicate Angrist et al. (2017) for the 
Microeconometrics project
"""

import numpy as np
import pandas as pd
#import statsmodels as sm
import statsmodels.formula.api as smf
from linearmodels import IV2SLS

#import statsmodels.formula.api as smf
#from linearmodels import IV2SLS
from linearmodels.iv._utility import annihilate, proj
from linearmodels.utility import (InvalidTestStatistic, WaldTestStatistic, _ModelComparison,_SummaryStr, _str, pval_format)


#from auxiliary.auxiliary_regressions import *
#from auxiliary.auxiliary_plots import *

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


def create_table2and3(df,outcomes,panels):
    df['clsize_ols'] = df['clsize_snv'] / 10
    df_nc = df.loc[df.north_center == 1]
    df_nc.region = df_nc.region.cat.remove_unused_categories()
    df_s = df.loc[df.north_center == 2]
    df_s.region = df_s.region.cat.remove_unused_categories()
    datasets = [df, df_nc, df_s]
    model = [smf.ols,IV2SLS.from_formula,IV2SLS.from_formula]
    outcomes = df[outcomes]
    columns= pd.MultiIndex.from_product([['OLS', 'IV/2SLS',' IV/2SLS'],
                                         ['Italy', 'North/Center', 'South']])
    table = pd.DataFrame()
    for i, outcome in enumerate(outcomes):
        clmn = 0
        idx = pd.MultiIndex.from_product([[panels[i]],
                                          ['Class size - coefficient', 
                                           'Class size - SE', 'Enrollment',
                                           'Enrollment squared','Interactions',
                                           'Observations']])
        newtable = pd.DataFrame(index=idx,columns=columns)
        formula1 = outcome +'~ d + clsize_ols + female+ m_female+ immigrants_broad+ m_origin+ dad_lowedu+ dad_midedu+ dad_highedu+ mom_unemp+ mom_housew+ mom_employed+ m_mom_edu + students+ students2 + C(survey) + C(grade) + enrol_ins_snv*region'
        formula2 = outcome +'~ [clsize_ols ~ clsize_hat] + d + female+ m_female+ immigrants_broad+ m_origin+ dad_lowedu+ dad_midedu+ dad_highedu+ mom_unemp+ mom_housew+ mom_employed+ m_mom_edu + students+ students2 + C(survey) + C(grade) + enrol_ins_snv*region'
        formula3 = outcome +'~ [clsize_ols ~ clsize_hat] + d + female+ m_female+ immigrants_broad+ m_origin+ dad_lowedu+ dad_midedu+ dad_highedu+ mom_unemp+ mom_housew+ mom_employed+ m_mom_edu + C(survey) + C(grade) + enrol_ins_snv*region + students*C(segment) + students2*C(segment)'
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

def create_table4a(df):
    df['placeholder'] = 0
    table = pd.DataFrame()
    vars_stats = {'clsize_snv':['mean', 'std'],'enrol_sch_snv':['mean', 'std'],'ratio_clsize':['mean', 'std'],
                          'ratio_sch_enrol' :['mean', 'std'],'ratio_ins_enrol':['mean', 'std'],'female':['mean', 'std'],'immigrants_broad':['mean', 'std'],
                          'dad_midedu':['mean', 'std'],'mom_employed':['mean', 'std'],'m_dad_edu':['mean', 'std'],'m_mom_occ':['mean', 'std'],'m_origin':['mean', 'std'],'classid': ['count']}
    table = df.groupby(['placeholder','o_math']).agg(vars_stats).round(2)
    table = table.append(df.groupby(['north_center','o_math']).agg(vars_stats).round(2),ignore_index=False,sort=False)
    tablebo = df.groupby(['placeholder','o_math'])[['classid']].count()
    table = table.transpose()
    table.columns.set_levels([['Italy', 'North/Center','South'],['No Monitor', 'Monitor']],inplace=True)
    table.rename(index={'clsize_snv':'Class size','enrol_sch_snv':'Grade enrollment at school','ratio_clsize':'Percent in class sitting the test',
                          'ratio_sch_enrol':'Percent in school sitting the test','ratio_ins_enrol':'Percent in institution sitting the test',
                          'female':'Female students','immigrants_broad':'Immigrant students','dad_midedu':'Father HS',
                  'mom_employed':'Mother employed','m_dad_edu':'Missing data on father\'s education',
                  'm_mom_occ':'Missing data on mother\'s occupation','m_origin':'Missing data on country of origin','classid':'Observations'},inplace=True)
    
    return table

def tables_balance(df,outcomes,idx,CONTROLS,var_interest):
    df['clsize_ols'] = df['clsize_snv']/10
    df_nc = df.loc[df.north_center == 1]
    df_nc.region = df_nc.region.cat.remove_unused_categories()
    df_s = df.loc[df.north_center == 2]
    df_s.region = df_s.region.cat.remove_unused_categories()
    datasets = [df, df_nc, df_s]
    columns = [['Italy', 'North/Center','South']]
    table = pd.DataFrame(index=idx,columns=columns)
    for i, data in enumerate(datasets):
        results_interest = []
        for j, outcome in enumerate(outcomes):
            formula = outcome +' ~ '+CONTROLS+var_interest
            result = smf.ols(formula,data).fit(cov_type='cluster', cov_kwds={'groups': data['clu']})
            results_interest.append("{:.4f}".format(result.params[var_interest]))
            results_interest.append("{:.4f}".format(result.bse[var_interest]))
        this_column = table.columns[i]
        table[this_column] = results_interest
    
    return table

def create_table4b(df):
    idx = [['Class size - coefficient','Class size - SE','Grade enrollment at school - coefficient','Grade enrollment at school - SE',
            'Percent in class sitting the test - coefficient','Percent in class sitting the test - SE',
            'Percent in school sitting the test - coefficient','Percent in school sitting the test - SE',
            'Percent in institution sitting the test - coefficient','Percent in institution sitting the test - SE',
            'Female students - coefficient','Female students - SE',
            'Immigrant students - coefficient','Immigrant students - SE',
            'Father HS - coefficient','Father HS - SE',
            'Mother employed - coefficient','Mother employed - SE',
            'Missing data on father\'s education - coefficient','Missing data on father\'s education - SE',
            'Missing data on mother\'s occupation - coefficient','Missing data on mother\'s occupation - SE',
            'Missing data on country of origin - coefficient','Missing data on country of origin - SE']]
    outcomes = ['clsize_ols','enrol_sch_snv','ratio_clsize','ratio_sch_enrol','ratio_ins_enrol','female',
                'immigrants_broad','dad_midedu','mom_employed','m_dad_edu','m_mom_occ','m_origin']
    CONTROLS = 'C(grade) + C(survey) + enrol_ins_snv*C(region) +'
    var_interest = 'o_math'
    table = tables_balance(df,outcomes,idx,CONTROLS,var_interest)
    
    return table

def create_table5(df):
    panels = ['Panel A. Math', 'Panel B. Language']
    columns= pd.MultiIndex.from_product([['Score manipulation', 'Test scores'],
                                          ['Italy', 'North/Center', 'South']])
    subjects = ['math','ital']
    df_nc = df.loc[df.north_center == 1]
    df_nc.region = df_nc.region.cat.remove_unused_categories()
    df_s = df.loc[df.north_center == 2]
    df_s.region = df_s.region.cat.remove_unused_categories()
    datasets = [df, df_nc, df_s]
    table = pd.DataFrame()
    for i, subject in enumerate(subjects):
        outcomes = ['our_CHEAT_'+subject,'answers_'+subject+'_std']
        col = 0
        idx = pd.MultiIndex.from_product([[panels[i]],
                                          ['Monitor at institution - coefficient', 
                                            'Monitor at institution - SE', 
                                            'Dependent variable mean',
                                            'Dependent variable mean',
                                            'Observations']])
        newtable = pd.DataFrame()
        newtable = pd.DataFrame(index=idx,columns=columns)
        X = ' female+ m_female+ immigrants_broad+ m_origin+ dad_lowedu+ dad_midedu+ dad_highedu+ mom_unemp+ mom_housew+ mom_employed+ m_mom_edu + '
        POLY = ' students+ students2  + students*C(segment) + students2*C(segment) + '
        FIXED = ' C(survey) + C(grade) + enrol_ins_snv*C(region) + C(d) + '
        for j, outcome in enumerate(outcomes):
            formula = outcome +' ~ '+X+POLY+FIXED+' C(o_math) '
            for k, data in enumerate(datasets):
                results_interest = []
                data = data[data[outcome].notna()]
                result = smf.ols(formula,data).fit(cov_type='cluster', cov_kwds={'groups': data['clu']})
                results_interest.append("{:.3f}".format(result.params['C(o_math)[T.1]']) )
                results_interest.append("{:.3f}".format(result.bse['C(o_math)[T.1]']) )
                results_interest.append("{:.3f}".format(data[outcome].mean()))
                results_interest.append("{:.3f}".format(data[outcome].std()))
                results_interest.append("{:.0f}".format(result.nobs))
                this_column = newtable.columns[col]
                newtable[this_column] = results_interest
                col = col+1
        table = table.append(newtable,ignore_index=False)
    
    return table

    
def create_table6(df):
    df['clsize_ols'] = df['clsize_snv']/10
    df['clsize_monit_no'] = np.nan
    df['clsize_monit_no'] = df['clsize_ols']*(1-df['o_math'])
    df['clsize_monit'] = np.nan
    df['clsize_monit'] = df['clsize_ols']*df['o_math']
    df['clsize_hat_monitor'] = np.nan
    df['clsize_hat_monitor'] = df['clsize_hat']*df['o_math']
    columns= pd.MultiIndex.from_product([['Math', 'Language'],
                                         ['Italy', 'North/Center', 'South']])
    idx = [['Class size x Monitor - coefficient','Class size x Monitor - SE',
            'Class size x No monitor - coefficient','Class size x No monitor - SE',
            'Monitor - coefficient','Monitor - SE','Observations']]
    table = pd.DataFrame(index=idx,columns=columns)
    outcomes = ['answers_math_std','answers_ital_std']
    vars_interest = ['clsize_monit','clsize_monit_no','C(o_math)[T.1]']
    col = 0
    df_nc = df.loc[df.north_center == 1]
    df_nc.region = df_nc.region.cat.remove_unused_categories()
    df_s = df.loc[df.north_center == 2]
    df_s.region = df_s.region.cat.remove_unused_categories()
    datasets = [df, df_nc, df_s]
    for i, outcome in enumerate(outcomes):
        for j, data in enumerate(datasets):
            results_interest = []
            formula = outcome + '~ female+ m_female+ immigrants_broad+ m_origin+ dad_lowedu+ dad_midedu+ dad_highedu+ mom_unemp+ mom_housew+ mom_employed+ m_mom_edu + students+ students2 + C(survey) + C(grade) + enrol_ins_snv*C(region) + C(o_math) + C(d) + students*C(segment) + students2*C(segment) + [clsize_monit + clsize_monit_no ~ clsize_hat + clsize_hat_monitor]' 
            result = IV2SLS.from_formula(formula,data).fit(cov_type='clustered',clusters=data.clu)
            for k, var_interest in enumerate(vars_interest):
                results_interest.append("{:.3f}".format(result.params[vars_interest[k]]))
                results_interest.append("{:.3f}".format(result.std_errors[vars_interest[k]]))
            results_interest.append("{:.0f}".format(result.nobs))
            this_column = table.columns[col]
            table[this_column] = results_interest
            col = col+1
            
    return table

def create_table7(df):
    panels = ['Panel A. Score manipulation', 'Panel B. Class size']
    end_vars = [['our_CHEAT_math','our_CHEAT_ital'],['clsize_ols']]
    df['clsize_ols'] = df['clsize_snv'] / 10
    df_nc = df.loc[df.north_center == 1]
    df_nc.region = df_nc.region.cat.remove_unused_categories()
    df_s = df.loc[df.north_center == 2]
    df_s.region = df_s.region.cat.remove_unused_categories()
    datasets = [df, df_nc, df_s]
    table = pd.DataFrame()
    for i, panel in enumerate(panels):
        outcomes = end_vars[i]
        col = 0
        columns= pd.MultiIndex.from_product([['Math', 'Language'],['Italy', 'North/Center', 'South']])
        idx = []
        idx = pd.MultiIndex.from_product([[panels[i]],['Maimonides\' Rule - coefficient',
                                                       'Maimonides\' Rule - SE',
                                                       'Monitor at institution - coefficient',
                                                       'Monitor at institution - SE','Observations']])
        newtable = pd.DataFrame()
        newtable = pd.DataFrame(index=idx,columns=columns)
        X = ' female+ m_female+ immigrants_broad+ m_origin+ dad_lowedu+ dad_midedu+ dad_highedu+ mom_unemp+ mom_housew+ mom_employed+ m_mom_edu + '
        POLY = ' students+ students2  + students*C(segment) + students2*C(segment) + '
        FIXED = ' C(survey) + C(grade) + enrol_ins_snv*C(region) + C(d) + '
        for j, outcome in enumerate(outcomes):
            formula = outcome +' ~ '+X+POLY+FIXED+' C(o_math) + clsize_hat '
            for k, data in enumerate(datasets):
                results_interest = []
                data = data[data[outcome].notna()]
                result = smf.ols(formula,data).fit(cov_type='cluster', cov_kwds={'groups': data['clu']})
                results_interest.append("{:.4f}".format(result.params['clsize_hat']) )
                results_interest.append("{:.4f}".format(result.bse['clsize_hat']) )
                results_interest.append("{:.4f}".format(result.params['C(o_math)[T.1]']))
                results_interest.append("{:.4f}".format(result.bse['C(o_math)[T.1]']))
                results_interest.append("{:.0f}".format(result.nobs))
                this_column = newtable.columns[col]
                newtable[this_column] = results_interest
                col = col+1
        table = table.append(newtable,ignore_index=False)
    
    return table

def create_table8(df):
    outcomes = ['math','ital']
    panels = ['Panel A. Math', 'Panel B. Language']
    WEGOEXTRA = 'fuzzy2_d2 + fuzzy2_d3 + fuzzy2_d4 + fuzzy2_d5'
    df['clsize_ols'] = df['clsize_snv'] / 10
    df['inter_instr'] = df['clsize_hat']*df['o_math']
    for h, outcome in enumerate(outcomes):
        df['inter_'+outcome] = df['clsize_ols']*df['our_CHEAT_'+outcome]
    for l in range(2,6):
        df['inter_instr_over'+str(l)] = df['o_math']*df['fuzzy2_d'+str(l)]
    WEGOEXTRA_INTER = '+ inter_instr_over2 + inter_instr_over3 + inter_instr_over4 + inter_instr_over5'
    df_nc = df.loc[df.north_center == 1]
    df_nc.region = df_nc.region.cat.remove_unused_categories()
    df_s = df.loc[df.north_center == 2]
    df_s.region = df_s.region.cat.remove_unused_categories()
    datasets = [df, df_nc, df_s]
    columns= pd.MultiIndex.from_product([['IV/2SLS', 'IV/2SLS (overidentified)',' IV/2SLS (overidentified-interacted)'],
                                          ['Italy', 'North/Center', 'South']])
    table = pd.DataFrame()
    for i, outcome in enumerate(outcomes):
        col = 0
        idx = pd.MultiIndex.from_product([[panels[i]],
                                          ['Class size - coefficient', 
                                            'Class size - SE', 
                                            'Score manipulation - coefficient',
                                            'Score manipulation - SE',
                                            'Class size x Score manipulation - coefficient',
                                            'Class size x Score manipulation - SE',
                                            'Overid test p-value','Observations']])
        newtable = pd.DataFrame(index=idx,columns=columns)
        X = ' female+ m_female+ immigrants_broad+ m_origin+ dad_lowedu+ dad_midedu+ dad_highedu+ mom_unemp+ mom_housew+ mom_employed+ m_mom_edu + '
        POLY = ' students+ students2  + students*C(segment) + students2*C(segment) + '
        FIXED = ' C(survey) + C(grade) + enrol_ins_snv*C(region) + C(d) + '
        formula1 = 'answers_'+outcome+'_std ~'+X+FIXED+POLY+' [clsize_ols + our_CHEAT_'+outcome+' ~ clsize_hat + o_math]' 
        formula2 = 'answers_'+outcome+'_std ~'+X+FIXED+POLY+' [clsize_ols + our_CHEAT_'+outcome+' ~ clsize_hat + o_math +'+ WEGOEXTRA+']' 
        formula3 = 'answers_'+outcome+'_std ~'+X+FIXED+POLY+' [clsize_ols + our_CHEAT_'+outcome+' + inter_'+outcome+' ~ clsize_hat + o_math + inter_instr +'+ WEGOEXTRA + WEGOEXTRA_INTER + ']' 
        formulas = [formula1,formula2,formula3]
        vars_interest = ['clsize_ols','our_CHEAT_'+outcome,'inter_'+outcome]
        for j, formula in enumerate(formulas):
            for k, data in enumerate(datasets):
                results_interest = []
                data = data[data['our_CHEAT_'+outcome].notna()]
                result = IV2SLS.from_formula(formula,data).fit(cov_type='clustered',clusters=data.clu)
                for k in range(max(2,j+1)):
                    results_interest.append("{:.4f}".format(result.params[vars_interest[k]]))
                    results_interest.append("{:.4f}".format(result.std_errors[vars_interest[k]]))
                if j == 0:
                    results_interest.append('')
                    results_interest.append('')
                    results_interest.append('')
                elif j == 1:
                    results_interest.append('')
                    results_interest.append('')
                    results_interest.append("{:.3f}".format(result.sargan.pval))
                else:
                    results_interest.append("{:.3f}".format(result.sargan.pval))
                results_interest.append("{:.0f}".format(result.nobs))
                this_column = newtable.columns[col]
                newtable[this_column] = results_interest
                col = col+1
        table = table.append(newtable,ignore_index=False)
        
    return table

def create_table9(df):
    outcomes = ['ratio_clsize','ratio_sch_enrol','ratio_ins_enrol','female',
                'immigrants_broad','dad_midedu','mom_employed','m_dad_edu','m_mom_occ','m_origin']
    idx = [['Percent in class sitting the test - coefficient','Percent in class sitting the test - SE',
            'Percent in school sitting the test - coefficient','Percent in school sitting the test - SE',
            'Percent in institution sitting the test - coefficient','Percent in institution sitting the test - SE',
            'Female students - coefficient','Female students - SE',
            'Immigrant students - coefficient','Immigrant students - SE',
            'Father HS - coefficient','Father HS - SE',
            'Mother employed - coefficient','Mother employed - SE',
            'Missing data on father\'s education - coefficient','Missing data on father\'s education - SE',
            'Missing data on mother\'s occupation - coefficient','Missing data on mother\'s occupation - SE',
            'Missing data on country of origin - coefficient','Missing data on country of origin - SE']]
    POLY = ' students+ students2  + students*C(segment) + students2*C(segment) + '
    FIXED = ' C(survey) + C(grade) + enrol_ins_snv*C(region) + C(d) + '
    CONTROLS = POLY+FIXED
    var_interest = 'clsize_hat'
    table = tables_balance(df, outcomes, idx, CONTROLS, var_interest)
    
    return table    