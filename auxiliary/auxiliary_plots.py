# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 23:35:07 2020

@author: FedericoAlexander
"""


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set(style="white", color_codes=True)

from auxiliary.auxiliary_tables import *

plt.style.use('seaborn')

def prepare_data_fig2and3(df):
    grouped = df[['grade','enrol_sch_snv','d','clsize_snv','clsize_hat']]
    grouped = grouped[grouped['enrol_sch_snv']<=150]
    grouped = grouped.groupby(['grade','enrol_sch_snv','d'],as_index=False)[['clsize_snv','clsize_hat']].mean()
    
    return grouped

def create_fig2(df):
    grouped = prepare_data_fig2and3(df)
    fig, (ax1,ax2) = plt.subplots(2,1,sharex='col', sharey=True,figsize=(8,8))
    plt.ylim(10,30)
    plt.xticks([0,25,50,75,100,125,150])
    ax1.plot('enrol_sch_snv','clsize_snv',data=grouped[(grouped['grade']==1) & (grouped['d']=='All remaining grades/years')],linestyle='none',marker='o',alpha=0.6)
    ax1.plot('enrol_sch_snv','clsize_hat',data=grouped[(grouped['grade']==1) & (grouped['d']=='All remaining grades/years')],linestyle='-',marker='',color='firebrick')
    ax1.set_title('Panel A. Grade 2')
    ax1.title.set_position([0.1, 1])
    ax1.set(ylabel='Class size')
    ax2.plot('enrol_sch_snv','clsize_snv',data=grouped[(grouped['grade']==0) & (grouped['d']=='All remaining grades/years')],linestyle='none',marker='o',alpha=0.6)
    ax2.plot('enrol_sch_snv','clsize_hat',data=grouped[(grouped['grade']==0) & (grouped['d']=='All remaining grades/years')],linestyle='-',marker='',color='firebrick')
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
    plt.plot('enrol_sch_snv','clsize_snv',data=grouped[(grouped['grade']==1) & (grouped['d']!='All remaining grades/years')],linestyle='none',marker='o',alpha=0.6)
    plt.plot('enrol_sch_snv','clsize_hat',data=grouped[(grouped['grade']==1) & (grouped['d']!='All remaining grades/years')],linestyle='-',marker='',color='firebrick')
    l = fig.legend(bbox_to_anchor=[-0.05,0,1,0.41])
    l.get_texts()[0].set_text('Actual classes')
    l.get_texts()[1].set_text('Maimonides\' Rule')
    fig.text(0.5, 0.0, 'FIGURE 3. CLASS SIZE BY ENROLLMENT IN POST-REFORM YEARS', ha='center', va='center')
    fig.tight_layout()
    
    return