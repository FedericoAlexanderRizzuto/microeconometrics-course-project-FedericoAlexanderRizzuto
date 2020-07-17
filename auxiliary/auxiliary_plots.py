# -*- coding: utf-8 -*-
"""
Date: July 17, 2020
Author: Federico Alexander Rizzuto
Content: Code producing tables needed to replicate Angrist et al. (2017) for the 
Microeconometrics project
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set(style="white", color_codes=True)
import statsmodels.formula.api as smf

from auxiliary.auxiliary_tables import *
from auxiliary.auxiliary_regressions import *

plt.style.use('seaborn')

def plot_count_classes(df):
    g = sns.catplot(x = 'survey', hue='grade',hue_order=[1,0],col='area',kind="count",data=df,palette='tab20c',legend=False)
    axes = g.axes.flatten()
    axes[0].set_title('North')
    axes[1].set_title('Center')
    axes[2].set_title('South')
    g.set_axis_labels('Survey Year','Number of classes')
    g.add_legend(title='Grade')
    new_labels = ['2', '5']
    for t, l in zip(g._legend.texts, new_labels): t.set_text(l)
    plt.show()
   
def plot_dist(df,var2plot,figtitle,xlim,ylim,xlabel):
    for i in range(len(var2plot)):
        g = sns.FacetGrid(df,hue='area',col='grade',col_order=[1,0],margin_titles=True)
        g.map(sns.distplot,var2plot[i]).fig.set_size_inches(10,6)
        axes = g.axes.flatten()
        axes[0].set_title('Grade 2')
        axes[1].set_title('Grade 5')
        g.set_axis_labels(xlabel,'Frequency')
        g.add_legend(title='Area')
        plt.suptitle(figtitle[i])
        plt.xlim(xlim)
        plt.ylim(ylim)
    plt.show()
    
    return

def plot_score_dist(df):
    var2plot = ['answers_math_pct','answers_ital_pct']
    figtitle = ['Test scores in math','Test scores in Italian language']
    xlim = [0,100]
    ylim = [0,0.06]
    xlabel = 'Pct correct answers'
    plot_dist(df, var2plot, figtitle, xlim, ylim, xlabel)
    
    return
    
def plot_demo_dist(df,xlim,ylim):
    var2plot = ['immigrants_broad','dad_midedu','mom_employed']
    figtitle = ['Immigrant students','Father HS graduate','Mother employed']
    #xlim = [0.1,1]
    #ylim = [0,6]
    xlabel = 'Pct student'
    plot_dist(df, var2plot, figtitle, xlim, ylim, xlabel)
    
    return

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
    intervals = [[-12,-2],[2,12]]
    for i in range(1,3):
        for j, interval in enumerate(intervals):
            grouped_data['temp'] = np.nan
            cond_ma = ((grouped_data['north_center']== i) & (grouped_data['dev']>=interval[0]) & (grouped_data['dev']<=interval[1]) )
            grouped_data['temp'] = np.where(cond_ma,grouped_data[variable],np.nan)
            grouped_data['MA'] = np.where(cond_ma,grouped_data['temp'].rolling(window=3,min_periods=1,center=True).mean(),grouped_data['MA'])
            
    return grouped_data['MA']


def create_fig4(df):
    figtitle = 'FIGURE 4. CLASS SIZE AND ENROLLMENT, CENTERED AT MAIMONIDES\' CUTOFFS'
    yticks = [-5,-3,-1,1,3,5]
    RDgraph(df, 'clsize_snv', figtitle, yticks)
    
    return

def create_fig5a(df):
    figtitle = 'FIGURE 5a. MATH TEST SCORES AND ENROLLMENT, CENTERED AT MAIMONIDES\' CUTOFFS'
    yticks = [-0.16,-0.12,-0.08,-0.04,0,0.04,0.08,0.12,0.16]
    RDgraph(df,'answers_math_std', figtitle, yticks)
    
    return

def create_fig5b(df):
    figtitle = 'FIGURE 5b. LANGUAGE TEST SCORES AND ENROLLMENT, CENTERED AT MAIMONIDES\' CUTOFFS'
    yticks = [-0.16,-0.12,-0.08,-0.04,0,0.04,0.08,0.12,0.16]
    RDgraph(df,'answers_ital_std', figtitle, yticks)
    
    return

def create_fig6a(df):
    figtitle = 'FIGURE 6a. MATH SCORES MANIPULATION AND ENROLLMENT, CENTERED AT MAIMONIDES\' CUTOFFS'
    yticks = [-0.06,-0.04,-0.02,0,0.02,0.04,0.06]
    RDgraph(df,'our_CHEAT_math', figtitle, yticks)
    
    return

def create_fig6b(df):
    figtitle = 'FIGURE 6b. LANGUAGE SCORE MANIPULATION AND ENROLLMENT, CENTERED AT MAIMONIDES\' CUTOFFS'
    yticks = [-0.06,-0.04,-0.02,0,0.02,0.04,0.06]
    RDgraph(df,'our_CHEAT_ital', figtitle, yticks)
    
    return

def RDgraph(df,outcome,figtitle,yticks):
    df['dev'] = cutoffs_center(df)
    subset = pd.DataFrame()
    subset = df[['female', 'm_female', 'immigrants_broad','m_origin', 'dad_lowedu',
            'dad_midedu', 'dad_highedu', 'mom_unemp', 'mom_housew', 'mom_employed', 'm_mom_edu', 
            'survey', 'region', 'north_center', 'grade','dev','d',outcome]]
    subset = subset[subset[outcome].notna()]
    formula = outcome + '~ female + m_female + immigrants_broad + m_origin + dad_lowedu + dad_midedu + dad_highedu + mom_unemp + mom_housew + mom_employed + m_mom_edu + C(survey) + C(region) + C(grade) + C(d)'
    subset['res'] = smf.ols(formula,data=subset).fit().resid
    subset = subset.groupby(['grade','north_center','dev'],as_index=False)['res'].mean()
    subset['MA'] = onesided_MAs(subset,'res')
    Rows = 2
    Cols = 2
    Tot = Rows*Cols
    Position = range(1,Tot + 1)
    fig = plt.figure(1)
    areas = [1,2,1,2]
    grades = [1,1,0,0]
    ttls = ['North/Center (Grade 2)','South (Grade 2)','North/Center (Grade 5)','South (Grade 5)']
    xlab = ['','','Enrollment','Enrollment']
    ylab = ['Raw and Smoothed Residuals','','Raw and Smoothed Residuals','']
    for i in range(Tot):
        g = fig.add_subplot(Rows,Cols,Position[i])
        g.axvline(color='r')
        g.plot('dev','MA',data=subset[(subset['north_center']==areas[i]) & (subset['grade']==grades[i]) & (subset['dev'] < -2)],linestyle='-',marker=None,color='b' )
        g.plot('dev','MA',data=subset[(subset['north_center']==areas[i]) & (subset['grade']==grades[i]) & (subset['dev'] > 2 )],linestyle='-',marker=None,color='b' )
        g.plot('dev','res',data=subset[(subset['north_center']==areas[i]) & (subset['grade']==grades[i])],linestyle='none',marker='o',color='c' )
        g.set_title(ttls[i])
        g.set_ylabel(ylab[i])
        g.set_xlabel(xlab[i])
        plt.yticks(yticks)
    fig.set_size_inches(9,9)
    fig.text(0, 0, figtitle)
    fig.tight_layout() 
    plt.show()
    
    return



def plot_monitor_demodata(df):
    subject = ['dad_midedu','mom_employed','immigrants_broad']
    subject_ylabel = ['Pct students with father HS graduate',
                      'Pct students with mother employed','Pct immigrant students']
    monitor = ['o_math','o_math','o_math']
    monitor_legend = ['Monitor at institution','Monitor at institution',
                      'Monitor at institution']
    ttls = ['Effect of monitors at the institution on demographic data in Grade 2',
            'Effect of monitors at the institution on demographic data in Grade 5']
    grades_order = [1,0]
    for i in range(2):
        Tot = 3
        Rows = 1
        Cols = 3
        Position = range(1,Tot + 1)
        fig = plt.figure(1)
        for k in range(Tot):
            g = fig.add_subplot(Rows,Cols,Position[k])
            sns.violinplot(x='north_center', y=subject[k], hue=monitor[k],split=True, 
                           cut=0, inner='quart',palette={0: 'y', 1: 'b'},
                           data=df.loc[df.grade==grades_order[i]])
            g.legend(title= monitor_legend[k],loc='upper center')
            sns.despine(left=True)
            g.set_xticklabels(['North and Center','South'])
            g.set_ylabel(subject_ylabel[k])
            g.set_xlabel('')
        fig.set_size_inches(14,8)
        fig.suptitle(ttls[i])
        plt.show()
        
    return

def plot_monitor_testscores(df):
    subject = ['answers_math_pct','answers_math_pct','answers_ital_pct',
               'answers_ital_pct']
    subject_ylabel = ['Pct correct answers in math','Pct correct answers in math',
                      'Pct correct answers in Italian','Pct correct answers in Italian']
    monitor = ['sampled_math','o_math','sampled_ital','o_math']
    monitor_legend = ['Class monitored in math','Monitor at institution',
                      'Class monitored in Italian','Monitor at institution']
    figtitle = ['Effect of monitors in class and at the institution on Grade 2 test scores',
            'Effect of monitors in class and at the institution on Grade 5 test scores']
    grades_order = [1,0]
    for i in range(2):
        Tot = 4
        Rows = 2
        Cols = 2
        Position = range(1,Tot + 1)
        fig = plt.figure(1)
        for k in range(Tot):
            g = fig.add_subplot(Rows,Cols,Position[k])
            sns.violinplot(x='north_center', y=subject[k], hue=monitor[k],split=True, 
                           inner='quart',palette={0: 'y', 1: 'b'},cut=0,
                           data=df.loc[df.grade==grades_order[i]])
            g.legend(title= monitor_legend[k],loc='upper center')
            sns.despine(left=True)
            g.set_xticklabels(['North and Center','South'])
            g.set_ylabel(subject_ylabel[k])
            g.set_xlabel('')
        fig.set_size_inches(14,8)
        fig.suptitle(figtitle[i])
        plt.show()
        
    return