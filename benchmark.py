#%%
import json
import os
import numpy as np
import scipy as sp
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from argparse import ArgumentParser

from mpl_toolkits.axes_grid.inset_locator import inset_axes
import csv

THREADS = {'host':16, 'mic':240, 'titanic':24} 
BOUNDS = {'host':14, 'mic':160, 'titanic':20} 
dataToElaborate = {'th':[], 'ff':[]}
mapping = {'th': 'C++ thread', 'ff':'fastflow'}

error_position=21
rep=20
threadsnum_position=23

DIR = './results/'

def import_data(filename):
    with open(filename, 'rb') as csvfile:
        spamreader = csv.reader(csvfile, delimiter = '\t')
        data=[]
        spamreader.next()
        for row in spamreader:
            data.append(row)
        return data

parser = ArgumentParser()
parser.add_argument("-s", "--saveFiles", default=False, required=False, type=bool)

args = parser.parse_args()

for fol in ['titanic', 'mic', 'host']:
    x_ideal = np.linspace(1, THREADS[fol])
    
    if fol == 'titanic':
        label = [5000, 10000, 15000, 25000]
    else:
        label = [5000, 10000, 15000, 25000, 30000]

    for i in label:
        if (fol == 'titanic'):
            ylim =[-0.18, 10.7]
            if i == 5000:
                ylim_sub =[-0.2, 0.5]
            elif i==10000:
                ylim_sub = [-0.2, 1.6]  
            else:
                ylim_sub =[-0.2, 4]                            
        else:
            if (i == 30000):
                ylim = [0, 6]
            else: ylim = [-0.3, 3.8]
            if i==5000:
                ylim_sub =[-0.1, 0.26]
            elif i==10000:
                ylim_sub = [-0.1, 0.9]  
            else:
                ylim_sub =[-0.12, 1.8]
                

        y = {'th': [], 'ff':[]}
        fig = plt.figure()
        save = args.saveFiles
        if save:    
            name = DIR + "graphs/" + fol + "/benchmark_graph_" + fol + '_' + str(i) + ".SVG"  
        else:
            name = ""
            

        sns.color_palette("Set1", n_colors=8, desat=.5)
        lw = 1.5
        markersize = 6

        plt.rc('text', usetex=True)
        params = {'legend.fontsize': 'large','figure.figsize': (18, 12), 'axes.labelsize': 'large', 'axes.titlesize':'large', 'xtick.labelsize':'large','ytick.labelsize':'large'}
            
        plt.rcParams.update(params)
        sns.set_style(style='white')
        
        ax_tc = fig.add_subplot(2, 2, 1)
        ax_scal = fig.add_subplot(2, 2, 2)
        ax_sp = fig.add_subplot(2, 2, 3)
        ax_eff = fig.add_subplot(2, 2, 4)
        if (i<=15000): 
            subplot_tc = inset_axes(ax_tc, width="80%", height="40%", loc=2)
            subplot_tc.set_ylim(ylim_sub)
            subplot_tc.set_xlim([0, BOUNDS[fol]]) #-0.22
            subplot_tc.yaxis.grid(True)
            subplot_tc.xaxis.grid(True)
            subplot_tc.yaxis.tick_right()

        
        if (fol == 'titanic'):
            cudaData = []
            cudafilename = fol + "/" +"cuda" + "_" + fol + "_" + str(i) +".csv"
            cudaData.append(import_data(DIR + cudafilename))

            tc_cuda = []
            for val in cudaData:
                p=[]
                for e in val:    
                    p = e[:rep]
                    p.sort()
                    p = p[:rep-1]
                    avg = reduce(lambda a, b: float(a) + float(b), p) / float(len(p))
                    tc_cuda.append(avg)

        for sel in ['ff', 'th']:   
            dataToElaborate[sel] = []
            data_seq = []
            
            seqfilename = fol + "/" +"seq" + "_" + fol + "_" + str(i) +".csv"
            data_seq.append(import_data(DIR + seqfilename))
            
            tc_seq = []
            for val in data_seq:
                p=[]
                for e in val:    
                    p = e[:rep]
                    p.sort()
                    p = p[:rep-1]
                    avg = reduce(lambda a, b: float(a) + float(b), p) / float(len(p))
                    tc_seq.append(avg)

            filename = fol + "/" + sel + "_" + fol + "_" + str(i) + ".csv"
            dataToElaborate[sel].append(import_data(DIR + filename))

            for val in dataToElaborate[sel]:
                x = []
                y_ideal = []
                p = []
                for e in val:
                    x.append(float(e[threadsnum_position]))
                    p = e[:rep]
                    p.sort()
                    p = p[:rep-1]
                    avg = reduce(lambda a, b: float(a) + float(b), p) / float(len(p))
                    y[sel].append(avg)
                    y_ideal.append(tc_seq[0] / float(e[threadsnum_position]))
                
            if (i<=15000): 
                subplot_tc.plot(x, y[sel], marker = '*',linewidth=lw, alpha=0.8, markersize=markersize)
            
            ax_tc.plot(x, y[sel], label = str(i) + ' ' + mapping[sel], linewidth = lw, marker = '*')
            ax_tc.set_xlabel(r'$\textit{Threads}$')
            ax_tc.set_ylabel(r'$\textit{Seconds}$')
            ax_tc.yaxis.grid(True)
            ax_tc.xaxis.grid(True)
            ax_tc.set_ylim(ylim)
            ax_tc.set_title('Completion time')
            
            #Scalability
            y_scal = [y[sel][0] / v for v in y[sel]]
            ax_scal.plot(x, y_scal, label= str(i) + ' ' + mapping[sel], linewidth=lw,alpha=0.8, marker= 'd', markersize=markersize)
            ax_scal.set_xlabel(r'$\textit{Threads}$')
            ax_scal.set_ylabel(r'$\textit{Scalability}$')
            ax_scal.yaxis.grid(True)
            ax_scal.xaxis.grid(True)
            ax_scal.set_title('Scalability')
            #Cuda scalability
            # if (fol == 'titanic'):
            #     y_scal_cuda = tc_cuda[0]/tc_cuda[1]
            #     x_scal_cuda = 16
            #     ax_scal.plot(x_scal_cuda, y_scal_cuda, label = 'cuda', marker= 'd', markersize=markersize, color='purple')
                
            #Speedup
            y_sp=[tc_seq[0]/v for v in y[sel]]
            ax_sp.plot(x, y_sp, label= str(i) + ' ' + mapping[sel],linewidth=lw,alpha=0.8, marker='o', markersize=markersize)
            ax_sp.set_xlabel(r'$\textit{Threads}$')
            ax_sp.set_ylabel(r'$\textit{Speedup}$')
            ax_sp.yaxis.grid(True)
            ax_sp.xaxis.grid(True)
            ax_sp.set_title('Speedup')
            #Cuda speedup
            if (fol == 'titanic'):
                y_sp_cuda = tc_seq[0]/tc_cuda[1]
                x_sp_cuda = 16
                ax_sp.plot(x_sp_cuda, y_sp_cuda, label = 'cuda', marker= 'o', markersize=markersize, color='purple')
            

            y2 = [tc_seq[0]/v for v in y[sel]]
            y_eff=[]
            for n, sp in zip(x,y2): 
                y_eff.append(sp/n)
            x_new = np.linspace(0, THREADS[fol], 100)
            y_new = [1 for val in x_new]
            ax_eff.plot(x, y_eff, label= str(i) + ' ' + mapping[sel], linewidth=lw, alpha=0.8, marker='*', markersize=markersize)
            ax_eff.set_xlabel(r'$\textit{Threads}$')
            ax_eff.set_ylabel(r'$\textit{Efficiency}$')
            ax_eff.yaxis.grid(True)
            ax_eff.xaxis.grid(True)
            ax_eff.set_ylim([0, 1.1])
            ax_eff.set_title('Efficiency')
            #Cuda efficiency
            if (fol == 'titanic'):
                y_eff_cuda = (tc_seq[0]/(16*tc_cuda[1]))
                x_eff_cuda = 16
                ax_eff.plot(x_eff_cuda, y_eff_cuda, label = 'cuda', marker= '*', markersize=markersize, color='purple')
            

        ax_scal.plot(x_ideal, x_ideal, label='ideal', linewidth=1.75, alpha=0.6, color='black', linestyle='--')
        ax_sp.plot(x_ideal, x_ideal, label='ideal', linewidth=1.75, alpha=0.6,color='black', linestyle='--')
        
        # tc Cuda purple line
        if (fol == 'titanic'):
            y_cuda = tc_cuda[1] #[tc_cuda[0] for val in x]
            x_cuda = 16 
            ax_tc.plot(x_cuda, y_cuda, label = 'cuda', color='purple', marker= '*', markersize=markersize ) #linewidth = 1.0, color='purple', linestyle='-.'
        
        yseq = [tc_seq[0] for val in x]   
        ax_tc.plot(x, yseq, label = 'sequential', linewidth = 1.0, color='green', linestyle='-.')     

        ax_tc.plot(x, y_ideal, label = 'ideal', linewidth=1.75, alpha=0.6, color='black', linestyle='--')
        
        if (i<=15000):
            subplot_tc.plot(x, y_ideal, label = 'ideal', linewidth=1.75, alpha=0.6,  color='black', linestyle='--')
            subplot_tc.plot(x, yseq, label = 'sequential', linewidth = 1.0, color='green', linestyle='-.')
            if (fol == 'titanic'):
                subplot_tc.plot(x_cuda, y_cuda, label = 'cuda', color='purple', marker= '*', markersize=markersize)   
        
        x_new = np.linspace(0, THREADS[fol], 100)
        y_new = [1 for val in x_new]
        ax_eff.plot(x_new, y_new, label='ideal', linewidth=1.75, alpha=0.6, color='black', linestyle='--')
        plt.subplots_adjust(wspace = 0.1, hspace = 0.35) #per mic
        # plt.tight_layout()
        ax_tc.legend(loc='lower center', bbox_to_anchor=(0.545, 0.436), bbox_transform=plt.gcf().transFigure)
    
        if name:
            fig.savefig(name)

