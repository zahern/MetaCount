import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.interpolate import griddata
import matplotlib.cm as cm
from matplotlib.ticker import ScalarFormatter,AutoMinorLocator
import matplotlib as mpl
plt.style.use('plt_style.txt')



os.chdir(os.path.expanduser('Z:/pap4aga'))

paran_data = "set_data.csv"
param_data_df = pd.read_csv(paran_data)
problem_number = 3
folders = param_data_df[param_data_df['problem_number'] ==problem_number]
folders = folders.index.values
folders = [i for i in range(0,6000)]
spread_checker = 100000
all_best = list()
all_best_hs = list()
all_best_sa = list()
for i in folders:
   
    file = str(i) +"/log.csv"
    try:
        
        df = pd.read_csv(file, on_bad_lines='skip')
        # Only keep the rows where the pvalue_exceed column is 0
        df = df[df['incumbent_pval_exceed'] == 0]
        df = df.reset_index(drop =True)
        print('good', i)
        x = df['incumbent_bic']
        
        
      

# Calculate weighted average
       

        
        
        if param_data_df['algorithm'][i] == 'de':
            param_1 = param_data_df['_hms'][i]
            param_2 = param_data_df['crossover'][i]
            val =min(x)
            all_best.append([param_1, param_2, val])
         
        if param_data_df['algorithm'][i] == 'hs':
            param_1 = param_data_df['_hms'][i]
            param_2 = param_data_df['_hmcr'][i]
            param_3 = param_data_df['_par'][i]
            
            val = min(x)
            all_best_hs.append([param_1, param_2, param_3, val])        
            
         
        if param_data_df['algorithm'][i] == 'sa':
            param_1 = param_data_df['crossover'][i]
            param_2 = param_data_df['temp_scale'][i]
            param_3 = param_data_df['steps'][i]
            
            val = min(x)
            all_best_sa.append([param_1, param_2, param_3, val])        
            
        
            
        if min(x) < spread_checker:
            spread_checker = min(x)
            print('best_folder is', i, 'with', spread_checker)
            store_2 = file
       
    except:
        print(i)

searborn_plots = True

def seaborn_plot(data):
    data = np.array(data)
    if np.shape(data)[1] ==3:
        x = data[:, 0]
        y  = data[:, 1]
        z = data[:, 2]
        x_unique = np.unique(x)
        y_unique = np.unique(y)
        xx, yy = np.meshgrid(x_unique, y_unique)
        zz = z.reshape(len(y_unique), len(x_unique))
        plt.imshow(zz, cmap='hot', interpolation='nearest')
        plt.colorbar()

        plt.xticks(range(len(x_unique)), x_unique)
        plt.yticks(range(len(y_unique)), y_unique)

        plt.xlabel('X')
        plt.ylabel('Y')

        plt.show()
        print(1)
        
    else:
        x = data[:, 0]
        y = data[:, 1]
        c = data[:, 2]
        z = data[:, 3]
        x_unique = np.unique(x)
        y_unique = np.unique(y)
        c_unique = np.unique(c)
        xx, yy, cc = np.meshgrid(x_unique, y_unique, c_unique, indexing='ij')
        
        points = np.column_stack((x, y, c))
        values = z
        interp_values = griddata(points, values, (xx, yy, cc), method='linear')
        max_index = np.unravel_index(np.argmax(interp_values), interp_values.shape)
        plt.imshow(interp_values[:, :, max_index[2]], cmap='hot', interpolation='nearest')
        plt.colorbar()

        plt.xticks(range(len(x_unique)), x_unique)
        plt.yticks(range(len(y_unique)), y_unique)

        plt.xlabel('X')
        plt.ylabel('Y')

        plt.title(f'Maximum Interpolated Z: {interp_values[max_index]:.2f}, X: {x_unique[max_index[1]]}, Y: {y_unique[max_index[0]]}, C: {c_unique[max_index[2]]}')

        plt.show()
        
        plt.imshow(interp_values[:, :, max_index[2]], cmap='hot', interpolation='nearest')
        plt.colorbar()

        plt.xticks(range(len(x_unique)), x_unique)
        plt.yticks(range(len(y_unique)), y_unique)

        plt.xlabel('X')
        print(1)

        

import seaborn as sns
def seaborn_sa(data, ax):
    # create a pivot table to aggregate the best solution for each combination of hyperparameters
    table = pd.pivot_table(data, values='best_solution', index=['temperature_steps'], columns=['temperature_decreasing_rate','percentage_of_crossover'], aggfunc='mean')
    table = table.fillna(method='ffill')
    # create the heatmap using seaborn
    sns.heatmap(table, cmap='RdBu_r', annot=False, fmt=".0f", linewidths=.5, ax=ax)

    # set the axis labels and title
    ax.set_xlabel("Temperature Decreasing Rate and Percentage of Crossover")
    ax.set_ylabel("Temperature Steps")
    ax.set_title("Simulated Annealing Hyperparameters")
    #ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

def seaborn_plot_hs(data, ax):
    # create a pivot table to aggregate the best solution for each combination of hyperparameters
    table = pd.pivot_table(data, values='best_solution', index=['_hms'], columns=['_hmcr', '_par'], aggfunc='mean')
    table = table.fillna(method='ffill')
    # create the heatmap using seaborn
    sns.heatmap(table, cmap='RdBu_r', annot=False, fmt=".0f", linewidths=.5, ax=ax)

    # set the axis labels and title
    ax.set_xlabel(" Harmony Memory Consideration Rate and Pitch Adjustment Rate")
    ax.set_ylabel("Harmony Memory Size")
    ax.set_title("Harmony Search Hyperparameters")
    
def seaborn_plot_de(data, ax):
    # Pivot the data to create a heatmap
    df_pivot = data.pivot_table(index='percentage_of_crossover', columns='population_size', values='best_solution', aggfunc='mean')
   # df_pivot = data.pivot_table('percentage_of_crossover', 'population_size', 'best_solution', aggfunc='mean')
    df_pivot = df_pivot.fillna(method='ffill')
    # Create the heatmap using Seaborn
    sns.heatmap(df_pivot, cmap='RdBu_r', annot=False, fmt='.0f', ax=ax)

    # Set the axis labels
    ax.set_ylabel('Population Size')
    ax.set_xlabel('Percentage of Crossover')
    ax.set_title("Differential Evolution Hyperparameters")
    #plt.show()
    #print(1)
            

if searborn_plots:

    #seaborn_plot(all_best)
    #seaborn_plot(all_best_hs)
    headings_sa = [ 'percentage_of_crossover', 'temperature_decreasing_rate', 'temperature_steps','best_solution' ]
    all_best_sa = pd.DataFrame(all_best_sa, columns=headings_sa)
    #seaborn_sa(all_best_sa)
    headings_hs = [ '_hms', '_hmcr', '_par', 'best_solution']
    all_best_hs = pd.DataFrame(all_best_hs, columns=headings_hs)
    headings_de = ['percentage_of_crossover', 'population_size', 'best_solution']
    all_best_de = pd.DataFrame(all_best, columns=headings_de)
    #seaborn_plot_de(all_best_de)

    #seaborn_plot_hs(all_best_hs)
    #seaborn_plot(all_best_sa)


    #plot all the same
    # create subplots with 1 row and 3 columns
    fig, axs = plt.subplots(3, 1, figsize=(5, 15))

    # create pivot tables and heatmaps for each dataset
    seaborn_sa(all_best_sa, axs[0])
    seaborn_plot_hs(all_best_hs, axs[1])
    seaborn_plot_de(all_best_de, axs[2])

    # Get the minimum and maximum values across all three plots
    vmin = min([ax.get_children()[0].get_clim()[0] for ax in axs])
    vmax = max([ax.get_children()[0].get_clim()[1] for ax in axs])

    # Set the same color scale for all three plots
    for ax in axs:
        ax.get_children()[0].set_clim(vmin=vmin, vmax=vmax)

    # adjust spacing between subplots
    plt.tight_layout()

    #tikzplotlib.save("output.tex")
    plt.savefig(f"pleasenew{problem_number}.svg")
    # show the plot
    plt.show()

   # plt.savefig(f"please{problem_number}.svg")
    print(1)



# Assuming data1 and data2 are your data sets





def split_lists(list1, list2, list3):
    sublists = {}
    for i in range(len(list1)):
        category = list3[i]
        if category in sublists:
            sublists[category][0].append(list1[i])
            sublists[category][1].append(list2[i])
        else:
            sublists[category] = ([list1[i]], [list2[i]])
    return sublists



