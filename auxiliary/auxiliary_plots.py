# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 23:35:07 2020

@author: FedericoAlexander
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set(style="white", color_codes=True)
import statsmodels as sm
import statsmodels.formula.api as smf
from linearmodels import IV2SLS

from auxiliary.auxiliary_tables import *
from auxiliary.auxiliary_regressions import *

plt.style.use('seaborn')

def plot_count_classes(df):
    g = sns.catplot(x = 'survey', hue='grade',hue_order=[1,0],col='north_center',kind="count",data=df,palette='tab20c',legend=False)
    axes = g.axes.flatten()
    axes[0].set_title('North and Center')
    axes[1].set_title('South')
    g.set_axis_labels('Survey Year','Number of Classes')
    g.add_legend(title='Grade')
    new_labels = ['2', '5']
    for t, l in zip(g._legend.texts, new_labels): t.set_text(l)
    plt.show()

def prepare_data_fig2and3(df):
    grouped = df[['grade','students','d','clsize_snv','clsize_hat']]
    grouped = grouped[grouped['students']<=150]
    grouped = grouped.groupby(['grade','students','d'],as_index=False)[['clsize_snv','clsize_hat']].mean()
    
    return grouped

def create_fig2(df):
    grouped = prepare_data_fig2and3(df)
    fig, (ax1,ax2) = plt.subplots(2,1,sharex='col', sharey=True,figsize=(8,8))
    plt.ylim(10,30)
    plt.xticks([0,25,50,75,100,125,150])
    ax1.plot('students','clsize_snv',data=grouped[(grouped['grade']==1) & (grouped['d']=='All remaining grades/years')],linestyle='none',marker='o',alpha=0.6)
    ax1.plot('students','clsize_hat',data=grouped[(grouped['grade']==1) & (grouped['d']=='All remaining grades/years')],linestyle='-',marker='',color='firebrick')
    ax1.set_title('Panel A. Grade 2')
    ax1.title.set_position([0.1, 1])
    ax1.set(ylabel='Class size')
    ax2.plot('students','clsize_snv',data=grouped[(grouped['grade']==0) & (grouped['d']=='All remaining grades/years')],linestyle='none',marker='o',alpha=0.6)
    ax2.plot('students','clsize_hat',data=grouped[(grouped['grade']==0) & (grouped['d']=='All remaining grades/years')],linestyle='-',marker='',color='firebrick')
    ax2.set_title('Panel B. Grade 5')
    ax2.title.set_position([0.1, 1])
    fig.text(0.5, 0.0, 'FIGURE 2. CLASS SIZE BY ENROLLMENT IN PRE-REFORM YEARS', ha='center', va='center')
    ax2.set(xlabel="Enrollment",ylabel='Class size')
    #l = plt.legend()
    l1 = ax1.legend(bbox_to_anchor=[-0.01,0,1,0.27])
    l1.get_texts()[0].set_text('Actual classes')
    l1.get_texts()[1].set_text('Maimonides\' Rule')
    l2 = ax2.legend(bbox_to_anchor=[-0.01,0,1,0.27])
    l2.get_texts()[0].set_text('Actual classes')
    l2.get_texts()[1].set_text('Maimonides\' Rule')
    fig.tight_layout()
    
    return

def create_fig3(df):
    grouped = prepare_data_fig2and3(df)
    fig = plt.figure(figsize=(8,4))
    fig.suptitle('Grade 2',fontsize=12,x=0.14,y=1)
    plt.xlabel('Enrollment')
    plt.ylabel('Class size')
    plt.ylim(10,30)
    plt.xticks([0,25,50,75,100,125,150])
    plt.plot('students','clsize_snv',data=grouped[(grouped['grade']==1) & (grouped['d']!='All remaining grades/years')],linestyle='none',marker='o',alpha=0.6)
    plt.plot('students','clsize_hat',data=grouped[(grouped['grade']==1) & (grouped['d']!='All remaining grades/years')],linestyle='-',marker='',color='firebrick')
    l = fig.legend(bbox_to_anchor=[-0.05,0,1,0.41])
    l.get_texts()[0].set_text('Actual classes')
    l.get_texts()[1].set_text('Maimonides\' Rule')
    fig.text(0.5, 0.0, 'FIGURE 3. CLASS SIZE BY ENROLLMENT IN POST-REFORM YEARS', ha='center', va='center')
    fig.tight_layout()
    
    return

def cutoffs_center(df):
    df['dev'] = np.nan
    cutoffs = [[25,50,75,100,125],[27,54,81,108,135]]
    regimes = ['All remaining grades/years','Grade 2 from 2010']
    for i, regime in enumerate(regimes):
        cutoffs_subset = cutoffs[i]
        for j, cutoff in enumerate(cutoffs_subset):
            condition = ((df['students'].between(cutoff-12, cutoff+12, inclusive=True)) & (df['d'].isin([regimes[i]])))
            df['dev'] = np.where(condition,df['students']-int(cutoff),df['dev'])
            
    return df['dev']

def onesided_MAs(grouped_data,variable):
    grouped_data['MA'] = np.nan
    intervals = [[-12,0],[1,12]]
    for i in range(1,3):
        for j, interval in enumerate(intervals):
            grouped_data['temp'] = np.nan
            cond_ma = ((grouped_data['north_center']== i) & (grouped_data['dev']>=interval[0]) & (grouped_data['dev']<=interval[1]) )
            grouped_data['temp'] = np.where(cond_ma,grouped_data[variable],np.nan)
            grouped_data['MA'] = np.where(cond_ma,grouped_data['temp'].rolling(window=3,min_periods=1,center=True).mean(),grouped_data['MA'])
            
    return grouped_data['MA']

def create_fig4(df):
    df['dev'] = cutoffs_center(df)    
    subset = df[['clsize_snv', 'female', 'm_female', 'immigrants_broad','m_origin', 'dad_lowedu',
            'dad_midedu', 'dad_highedu', 'mom_unemp', 'mom_housew', 'mom_employed', 'm_mom_edu', 
            'survey', 'region', 'north_center', 'grade','dev','d']]
    subset.dropna(how='any', inplace=True)
    subset['clsize_res'] = np.nan
    formula = 'clsize_snv ~ d + female + m_female + immigrants_broad + m_origin + dad_lowedu + dad_midedu + dad_highedu + mom_unemp + mom_housew + mom_employed + m_mom_edu + C(survey) + C(region)'
    result = smf.ols(formula,data=subset[(subset['grade']==0)]).fit()
    subset['clsize_res'] = result.resid
    result = smf.ols(formula,data=subset[(subset['grade']==1)]).fit()
    subset['clsize_res'].update(result.resid)
    subset = subset.groupby(['grade','north_center','dev'],as_index=False)['clsize_res'].mean()
    subset['MA'] = onesided_MAs(subset,'clsize_res')
    fig, [[ax1,ax2],[ax3,ax4]] = plt.subplots(2,2,sharex='col', sharey=True,figsize=(10,10))
    plt.yticks([-5,-3,-1,1,3,5])
    ax1.axvline(color='r')
    ax2.axvline(color='r')
    ax3.axvline(color='r')
    ax4.axvline(color='r')
    plt.ylim(-5.5,5.5)
    ax1.plot('dev','MA',data=subset[(subset['north_center']==1) & (subset['grade']==1)],linestyle='none',marker='o',color='c' )
    ax1.set_title('North and Center')
    ax2.plot('dev','MA',data=subset[(subset['north_center']==2) & (subset['grade']==1)],linestyle='none',marker='o',color='c' )
    ax2.set_title('South')
    ax3.plot('dev','MA',data=subset[(subset['north_center']==1) & (subset['grade']==0)],linestyle='none',marker='o',color='c' )
    ax3.set_title('North and Center')
    ax3.set(xlabel='Enrollment')
    ax4.plot('dev','MA',data=subset[(subset['north_center']==2) & (subset['grade']==0)],linestyle='none',marker='o',color='c' )
    ax4.set_title('South')
    ax4.set(xlabel='Enrollment')
    fig.text(0.05,.98,'PANEL A. Grade 2')
    fig.text(0.05,.51,'PANEL B. Grade 5')
    fig.text(0.5, 0.0, 'FIGURE 4. CLASS SIZE AND ENROLLMENT, CENTERED AT MAIMONIDES\' CUTOFFS', ha='center', va='center')
    fig.tight_layout()


    return 

def create_fig5(df):
    df['dev'] = cutoffs_center(df)    
    subset = df[['clsize_snv', 'female', 'm_female', 'immigrants_broad','m_origin', 'dad_lowedu',
            'dad_midedu', 'dad_highedu', 'mom_unemp', 'mom_housew', 'mom_employed', 'm_mom_edu', 
            'survey', 'region', 'north_center', 'grade','dev','answers_math_std','answers_ital_std','d']]
    subset.dropna(how='any', inplace=True)
    subset['ans_math_res'] = np.nan
    subset['ans_ital_res'] = np.nan
    formula = 'answers_math_std ~ d + female + m_female + immigrants_broad + m_origin + dad_lowedu + dad_midedu + d + dad_highedu + mom_unemp + mom_housew + mom_employed + m_mom_edu + C(survey) + C(region) + C(grade)'
    result_math = smf.ols(formula,data=subset).fit()
    subset['ans_math_res'] = result_math.resid
    subset_math = subset.groupby(['north_center','dev'],as_index=False)['ans_math_res'].mean()
    subset_math['MA'] = onesided_MAs(subset_math,'ans_math_res')
    formula = 'answers_ital_std ~ d + female + m_female + immigrants_broad + m_origin + dad_lowedu + dad_midedu + d + dad_highedu + mom_unemp + mom_housew + mom_employed + m_mom_edu + C(survey) + C(region) + C(grade)'
    subset['ans_ital_res'] = smf.ols(formula,data=subset).fit().resid
    subset_ital = subset.groupby(['north_center','dev'],as_index=False)['ans_ital_res'].mean()
    subset_ital['MA'] = onesided_MAs(subset_ital,'ans_ital_res')
    fig, [[ax1,ax2],[ax3,ax4]] = plt.subplots(2,2,sharex='col', sharey=True,figsize=(10,10))
    #plt.yticks([-5,-3,-1,1,3,5])
    plt.yticks([-0.10,-0.08,-0.06,-0.04,-0.02,0,0.02,0.04,0.06,0.08,0.10])
    ax1.axvline(color='r')
    ax2.axvline(color='r')
    ax3.axvline(color='r')
    ax4.axvline(color='r')
    plt.ylim(-0.1,0.1)
    ax1.plot('dev','MA',data=subset_math[(subset_math['north_center']==1)],linestyle='none',marker='o',color='c' )
    ax1.set_title('North and Center')
    ax2.plot('dev','MA',data=subset_math[(subset_math['north_center']==2)],linestyle='none',marker='o',color='c' )
    ax2.set_title('South')
    ax3.plot('dev','MA',data=subset_ital[(subset_ital['north_center']==1)],linestyle='none',marker='o',color='c' )
    ax3.set_title('North and Center')
    ax3.set(xlabel='Enrollment')
    ax4.plot('dev','MA',data=subset_ital[(subset_ital['north_center']==2)],linestyle='none',marker='o',color='c' )
    ax4.set_title('South')
    ax4.set(xlabel='Enrollment')
    fig.text(0.05,.98,'PANEL A. Math score')
    fig.text(0.05,.51,'PANEL B. Language score')
    fig.text(0.5, 0.0, 'FIGURE 5. TEST SCORES AND ENROLLMENT, CENTERED AT MAIMONIDES\' CUTOFFS', ha='center', va='center')
    fig.tight_layout()


    return 
    
def create_fig6(df):
    df['dev'] = cutoffs_center(df)    
    subset = df[['clsize_snv', 'female', 'm_female', 'immigrants_broad','m_origin', 'dad_lowedu',
            'dad_midedu', 'dad_highedu', 'mom_unemp', 'mom_housew', 'mom_employed', 'm_mom_edu', 
            'survey', 'region', 'north_center', 'grade','dev','our_CHEAT_math','our_CHEAT_ital','d']]
    subset.dropna(how='any', inplace=True)
    subset['cheat_math_res'] = np.nan
    subset['cheat_ital_res'] = np.nan
    formula = 'our_CHEAT_math ~ female + m_female + immigrants_broad + m_origin + dad_lowedu + dad_midedu + d + dad_highedu + mom_unemp + mom_housew + mom_employed + m_mom_edu + C(survey) + C(region) + C(grade)'
    result_math = smf.ols(formula,data=subset).fit()
    subset['cheat_math_res'] = result_math.resid
    subset_math = subset.groupby(['north_center','dev'],as_index=False)['cheat_math_res'].mean()
    subset_math['MA'] = onesided_MAs(subset_math,'cheat_math_res')
    formula = 'our_CHEAT_ital ~ female + m_female + immigrants_broad + m_origin + dad_lowedu + dad_midedu + d + dad_highedu + mom_unemp + mom_housew + mom_employed + m_mom_edu + C(survey) + C(region) + C(grade)'
    subset['cheat_ital_res'] = smf.ols(formula,data=subset).fit().resid
    subset_ital = subset.groupby(['north_center','dev'],as_index=False)['cheat_ital_res'].mean()
    subset_ital['MA'] = onesided_MAs(subset_ital,'cheat_ital_res')
    fig, [[ax1,ax2],[ax3,ax4]] = plt.subplots(2,2,sharex='col', sharey=True,figsize=(10,10))
    plt.yticks([-0.04,-0.02,0,0.02,0.04])
    ax1.axvline(color='r')
    ax2.axvline(color='r')
    ax3.axvline(color='r')
    ax4.axvline(color='r')
    plt.ylim(-0.04,0.04)
    ax1.plot('dev','MA',data=subset_math[(subset_math['north_center']==1)],linestyle='none',marker='o',color='c' )
    ax1.set_title('North and Center')
    ax2.plot('dev','MA',data=subset_math[(subset_math['north_center']==2)],linestyle='none',marker='o',color='c' )
    ax2.set_title('South')
    ax3.plot('dev','MA',data=subset_ital[(subset_ital['north_center']==1)],linestyle='none',marker='o',color='c' )
    ax3.set_title('North and Center')
    ax3.set(xlabel='Enrollment')
    ax4.plot('dev','MA',data=subset_ital[(subset_ital['north_center']==2)],linestyle='none',marker='o',color='c' )
    ax4.set_title('South')
    ax4.set(xlabel='Enrollment')
    fig.text(0.05,.98,'PANEL A. Math score manipulation')
    fig.text(0.05,.51,'PANEL B. Language score manipulation')
    fig.text(0.5, 0.0, 'FIGURE 6. SCORE MANIPULATION AND ENROLLMENT, CENTERED AT MAIMONIDES\' CUTOFFS', ha='center', va='center')
    fig.tight_layout()

    return 