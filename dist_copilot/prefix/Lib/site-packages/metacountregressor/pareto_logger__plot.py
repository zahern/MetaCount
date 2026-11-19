import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.interpolate import griddata
import matplotlib.cm as cm
from matplotlib.ticker import ScalarFormatter,AutoMinorLocator
from matplotlib.ticker import FormatStrFormatter
from matplotlib.animation import FuncAnimation
import matplotlib as mpl
import tikzplotlib
from sklearn.preprocessing import StandardScaler, MinMaxScaler
searborn_plots = 1
if not searborn_plots:
    plt.style.use('plt_style.txt')
# Generate some example data for two objectives
np.random.seed(1234)
y1 = np.random.randint(10, 100, 1000)
y2 = np.random.randint(10, 100, 1000)


def trace_plot(y1, y2):
    # Initialize the best solution with the first point
    best_solution = [y1[0], y2[0]]
    best_solution_idx = 0

    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True)

    

    # Initialize the Pareto frontier with the first solution
    pareto_x = [y1[0]]
    pareto_y = [y2[0]]
    pareto_frontier = [0]

    # Plot the initial Pareto frontier and best solution
    ax1.scatter(0, y1[0], marker='o', color='k', label='Initial best solution')
    ax2.scatter(0, y2[0], marker='o', color='k', label='Initial best solution')

    ax1.legend()
    ax1.set_ylabel('Objective 1')

    ax2.legend()
    ax2.set_ylabel('Objective 2')

    # Iterate over the remaining solutions and update the Pareto frontier and best solution
    for i in range(1, len(y1)):
        is_pareto = True
        for j in range(len(pareto_x)):
            if y1[i] >= pareto_x[j] and y2[i] >= pareto_y[j] and (y1[i] > pareto_x[j] or y2[i] > pareto_y[j]):
                is_pareto = False
                break
        if is_pareto:
            pareto_x.append(y1[i])
            pareto_y.append(y2[i])
            pareto_frontier.append(i)
            ax1.plot(pareto_frontier, pareto_x, color='r')
            ax2.plot(pareto_frontier, pareto_y, color='r')
            if y1[i] + y2[i] < best_solution[0] + best_solution[1]:
                best_solution = [y1[i], y2[i]]
                best_solution_idx = i
        else:
            ax1.plot(pareto_frontier, pareto_x, color='b')
            ax2.plot(pareto_frontier, pareto_y, color='b')

    # Plot the final best solution
    

    # Add labels and title
    plt.xlabel('Iteration')
    plt.suptitle('Traceplot')

    # Show the plot
    plt.show()





#os.chdir(os.path.expanduser('Z:/paper_1_bic_again'))
#os.chdir(os.path.expanduser('Z:/paper2_trial'))

#os.chdir(os.path.expanduser('Z:/again_poisson'))
#os.chdir(os.path.expanduser('Z:/synthg'))

#os.chdir(os.path.expanduser('Z:/paper_printer2'))
os.chdir(os.path.expanduser('Z:/alptest3'))
os.chdir(os.path.expanduser('Z:/new_paper3_fixed'))
os.chdir(os.path.expanduser('Z:/pap3la'))

def batches(Xs, Ys, iteration, batch_n):
    cunt = batch_n
    for n in (0, len(iteration)/batch_n):
        Xn =Xs[0:cunt]
        Yn = Ys[0:cunt]
       
        it = iteration[0:cunt]
        
        cunt +=batch_n
        new_pareto2(Xn, Yn,it)

    
    
def new_pareto2(Xs, Ys, iteration, maxX=False, maxY=False):
    '''Pareto frontier selection process'''
    sorted_list = sorted([[Xs[i], Ys[i], iteration[i]] for i in range(len(Xs))], reverse=maxY)
    pareto_front = [sorted_list[0]]
    for pair in sorted_list[1:]:
        if maxY:
            if pair[1] >= pareto_front[-1][1]:
                pareto_front.append(pair)
        else:
            if pair[1] <= pareto_front[-1][1]:
                pareto_front.append(pair)

    '''Plotting process'''
    pf_X = [pair[0] for pair in pareto_front]
    pf_Y = [pair[1] for pair in pareto_front]
    pf_I = [pair[2] for pair in pareto_front]
    print(pf_X)
    print(pf_Y)
    print(pf_I)
    plt.scatter(pf_X, pf_Y)
    plt.plot(pf_X, pf_Y, color='red')
 


def new_pareto(Xs, Ys, result, iteration, maxX=False, maxY=False):
    '''Pareto frontier selection process'''
    sorted_list = sorted([[Xs[i], Ys[i], iteration[i]] for i in range(len(Xs))], reverse=maxY)
    pareto_front = [sorted_list[0]]
    for pair in sorted_list[1:]:
        if maxY:
            if pair[1] >= pareto_front[-1][1]:
                pareto_front.append(pair)
        else:
            if pair[1] <= pareto_front[-1][1]:
                pareto_front.append(pair)

    '''Plotting process'''
    marker_size = 4
    pf_X = [pair[0] for pair in pareto_front]
    pf_Y = [pair[1] for pair in pareto_front]
    pf_I = [pair[2] for pair in pareto_front]
    print(pf_X)
    print(pf_Y)
    print(pf_I)
    plt.scatter(pf_X, pf_Y)
    fig, ax = plt.subplots()
    plt.plot(pf_X, pf_Y, color='red')
    xlim = ax.get_xlim()

    ylim = ax.get_ylim()
    ax.set_xlim(xlim[0], xlim[1])
    ax.set_ylim(ylim[0], ylim[1])

    # Define a color map and get a list of colors based on the number of keys
    colors = cm.viridis(np.linspace(0, 1, len(result.keys())))
    colors = ['blue','green', 'red', 'purple', 's', 'D', 'o']
    keys = [key for key in result.keys()]
    a = 0
    
    for key, (x, y) in result.items():
        if "_p" in key:
            marker_shape = "s"  # square marker
        elif "_nb" in key:
            marker_shape = "D"  # diamond marker
        elif "_gp" in key:
            marker_shape = "o"  # circle marker
        else:
            marker_shape = "o"  # default to circle marker
        
        if '_rp_c_' in key:
           color = colors[0]
        elif '_rp_' in key:
            color = colors[1]
         
        elif '_c_' in key:
            color = colors[2]
        else:
            color = colors[3]
            
        #plt.scatter(x, y, c= color, label = '%s ' % keys[a], marker=marker_shape)
        plt.scatter(x, y, c= color, marker=marker_shape)
        a += 1
    # add labels for the color and tick legends
    #plt.legend(label='', handles=[])
    old = 1
    if old:
        color_labels = ['Pos', 'NB', 'GP']
        tick_labels =  [ 'Random Correlated and Non Correlated', 'Random Parameters', 'Random Correlated Parameters','Fixed Effects', 'Poisson', 'Negative Binomial', 'Generalize Poisson']

    # create the color legend with labels
        color_legend = plt.legend(colors, color_labels, title='Colors', loc='upper left')

    # add the tick legend with labels

        for i, tick_label in enumerate(tick_labels):
            if i >= 4:
                plt.plot([], [], color = 'black', marker = colors[i], label = tick_label)
            else:
                
                plt.plot([], [], color=colors[i], label=tick_label)    
    
   
        
    plt.xlabel("BIC")
    plt.ylabel("MSE")
    plt.legend()
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
          ncol=3, fancybox=True, shadow=True)
    
    plt.show()
    print(1)
    
    

def new_scatter(Xs, Ys, result, iteration, maxX=False, maxY=False):
    '''Pareto frontier selection process'''
    sorted_list = sorted([[Xs[i], Ys[i], iteration[i]] for i in range(len(Xs))], reverse=maxY)
    pareto_front = [sorted_list[0]]
    for pair in sorted_list[1:]:
        if maxY:
            if pair[1] >= pareto_front[-1][1]:
                pareto_front.append(pair)
        else:
            if pair[1] <= pareto_front[-1][1]:
                pareto_front.append(pair)

    '''Plotting process'''
    pf_X = [pair[0] for pair in pareto_front]
    pf_Y = [pair[1] for pair in pareto_front]
    pf_I = [pair[2] for pair in pareto_front]
    print(pf_X)
    print(pf_Y)
    print(pf_I)
    plt.scatter(pf_X, pf_Y)
    fig, ax = plt.subplots()
    plt.plot(pf_X, pf_Y, color='red')
    

    # Define a color map and get a list of colors based on the number of keys
    colors = cm.viridis(np.linspace(0, 1, len(result.keys())))
    colors = ['blue','green', 'red', 'purple', 's', 'D', 'o']
    keys = [key for key in result.keys()]
    a = 0
    
    for key, (x, y) in result.items():
        if "_p" in key:
            marker_shape = "s"  # square marker
        elif "_nb" in key:
            marker_shape = "D"  # diamond marker
        elif "_gp" in key:
            marker_shape = "o"  # circle marker
        else:
            marker_shape = "o"  # default to circle marker
        
        if '_rp_c_' in key:
           color = colors[0]
        elif '_rp_' in key:
            color = colors[1]
         
        elif '_c_' in key:
            color = colors[2]
        else:
            color = colors[3]
            
        #plt.scatter(x, y, c= color, label = '%s ' % keys[a], marker=marker_shape)
        plt.scatter(x, y, c= color, marker=marker_shape)
        a += 1
    # add labels for the color and tick legends
    #plt.legend(label='', handles=[])
    old = 1
    if old:
        color_labels = ['Pos', 'NB']
        tick_labels =  [ 'Random Correlated and Non Correlated', 'Random Parameters', 'Random Correlated Parameters','Fixed Effects', 'Poisson', 'Negative Binomial']

    # create the color legend with labels
        color_legend = plt.legend(colors, color_labels, title='Colors', loc='upper left')

    # add the tick legend with labels

        for i, tick_label in enumerate(tick_labels):
            if i >= 4:
                plt.plot([], [], color = 'black', marker = colors[i], label = tick_label)
            else:
                
                plt.plot([], [], color=colors[i], label=tick_label)    
    
   
        
    plt.xlabel("BIC")
    plt.ylabel("MSE")
    plt.legend()

    plt.show()
    print(1)
    



def get_marker(key):
    if '_p' in key:
        return '^'
    elif '_nb' in key:
        return 's'
    elif '_gp' in key:
        return 'o'
    else:
        return 'o'
    


def plot_pareto_frontier(Xs, Ys, maxX=False, maxY=False):
    '''Pareto frontier selection process'''
    sorted_list = sorted([[Xs[i], Ys[i]] for i in range(len(Xs))], reverse=maxY)
    pareto_front = [sorted_list[0]]
    for pair in sorted_list[1:]:
        if maxY:
            if pair[1] >= pareto_front[-1][1]:
                pareto_front.append(pair)
        else:
            if pair[1] <= pareto_front[-1][1]:
                pareto_front.append(pair)
    
    '''Plotting process'''
    #plt.scatter(Xs,Ys)
    pf_X = [pair[0] for pair in pareto_front]
    pf_Y = [pair[1] for pair in pareto_front]
    plt.scatter(pf_X, pf_Y)
    fig, ax = plt.subplots()
    plt.plot(pf_X, pf_Y, color = 'red')
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    plt.scatter(Xs, Ys, color = 'green')
    plt.scatter(pf_X, pf_Y)
    ax.set_xlim(xlim[0], xlim[1])
    ax.set_ylim(ylim[0], ylim[1])
    plt.xlabel("BIC")
    plt.ylabel("MAE")
    plt.show()
    print(1)
    
def pareto_frontier(X, Y, maxX=False, maxY=False):
    """
    Given two sequences of x and y values, return the indices of the points on the Pareto frontier.
    """
    
    sorted_list = sorted([[X[i], Y[i]] for i in range(len(X))], reverse=maxY)
    pareto_front = [sorted_list[0]]
    for pair in sorted_list[1:]:
        if maxY:
            if pair[1] >= pareto_front[-1][1]:
                pareto_front.append(pair)
        else:
            if pair[1] <= pareto_front[-1][1]:
                pareto_front.append(pair)
    
    

    pareto_points_X = [point[0] for point in pareto_front]
    pareto_points_Y = [point[1] for point in pareto_front]
    return pareto_points_X, pareto_points_Y

def spread(X, Y, true_X, true_Y):
    """
    Calculate the spread of the Pareto frontier using the generational distance measure
    """
    pareto_X, pareto_Y = pareto_frontier(X, Y)
    gd = []
    for i, j in zip(pareto_X, pareto_Y):
        d = float("inf")
        for a, b in zip(true_X, true_Y):
            d = min(d, np.sqrt((i - a)**2 + (j - b)**2))
        gd.append(d)
    spread = sum(gd) / len(gd)
    return spread    
    
# plot pareto frontier
##plot_pareto_frontier(y,x)

#x\, y, = pareto_frontier(x, y)
true_x = list()
true_y = list()
folders = [1, 2, 3, 4] #etc
folders = list(range(0, 253, 4))
folders = list(range(1, 333, 2))

folders = list(range(1, 250, 2))
folders = list(range(1, 499, 3))

all_best = list()
all_best_hs = list()
all_best_sa = list()

spread_checker = 100000000

paran_data = "set_data.csv"
param_data_df = pd.read_csv(paran_data)
#param_data_df = param_data_df.iloc[:567]
#a = 9
#param_data_df = param_data_df.iloc[100:300]
problem_number = 8
folders = param_data_df[param_data_df['problem_number'] ==problem_number]
folders = folders.index.values

lowest_bic = 12000000
df_master = []
for i in folders:
    
        file = str(i) +"/log.csv"
        try:
            columns = ['incumbent_bic', 'incumbent_MAE', 'incumbent_HQIC', 'incumbent_RMSE', 'incumbent_aic', 'incumbent_CAIC', 'incumbent_simple', 'incumbent_RMSE_TEST', 'incumbent_MSE_TEST']
            df = pd.read_csv(file, usecols = columns)
            x = df['incumbent_bic']
            y = df['incumbent_MSE_TEST']
            if problem_number ==3:
                df = df[df['incumbent_bic'] <= 1000] 
                df = df[df['incumbent_bic'] >= 400] 
                df = df[df['incumbent_MSE_TEST'] <= 10]  
            df_master.append(df)
           
            
            
        except Exception as e:
            print(e)    
df = pd.concat(df_master)
#df = df_master
x = df['incumbent_bic']
y = df['incumbent_MSE_TEST']
scaler = MinMaxScaler()
scaler2 = MinMaxScaler()
# Fit the scaler to the first dataset and transform it
x = np.array(x).reshape(-1, 1)
y = np.array(y).reshape(-1, 1)
scaler.fit_transform(x)
scaler2.fit_transform(y)

skip = 1


def calculate_hypervolume(pareto_front, ref_point):
    hypervolume = 0.0
    pareto_front.sort(reverse = True)
    for i in range(len(pareto_front) - 1):
        delta_x = abs(pareto_front[i+1][0] - pareto_front[i][0])
        height = abs(ref_point[1] - pareto_front[i][1])
        hypervolume += delta_x * height
    return hypervolume

print(true_x)
hv_max = 0
hv_list = []


if skip:
    
    for i in folders:
    
        file = str(i) +"/log.csv"
        
        
        
        
        
        
        
        try:
            columns = ['incumbent_bic', 'incumbent_MAE', 'incumbent_HQIC', 'incumbent_RMSE', 'incumbent_aic', 'incumbent_CAIC', 'incumbent_simple', 'incumbent_RMSE_TEST', 'incumbent_MSE_TEST']
            df = pd.read_csv(file, usecols = columns)
            df_master.append(df)
            # Only keep the rows where the pvalue_exceed column is 0
            #df = df[df['incumbent_pval_exceed'] == 0]
            
            
            #if problem_number ==7:
            #    df = df[df['incumbent_bic'] >= 3600]
            
            if problem_number ==3:
                df = df[df['incumbent_bic'] <= 1000] 
                df = df[df['incumbent_bic'] >= 400] 
                df = df[df['incumbent_MSE_TEST'] <= 10]  
            if problem_number ==8:
                df = df[df['incumbent_bic'] <= 18000]
                
            if problem_number ==9:
                df = df[df['incumbent_bic'] <= 30000]    
                df = df[df['incumbent_MSE_TEST'] <= 10000]    
                    
            df =df[df['incumbent_MSE_TEST'] <= 10000000]
            
            df = df.reset_index(drop =True)
            print('good', i)
            x = df['incumbent_bic']
            y = df['incumbent_MSE_TEST']
            x, y = pareto_frontier(x, y)
        
            x = scaler.transform(np.array(x).reshape(-1,1))
            y = scaler2.transform(np.array(y).reshape(-1,1))
            
            pareto_front1 = [list(point) for point in zip(x, y)]
            hv = calculate_hypervolume(pareto_front1, [1,1])[0]
            hv_list.append(hv)
            print("Hypervolume: ", hv)
            
            
            if min(x) <lowest_bic:
                lowest_bic = min(x)
                print('best folder', i, 'at', lowest_bic)
                a = i
            else:
                print('best folder', a, 'at', lowest_bic)
                    
            w1 = .5
            w2 = 0.5

    # Calculate weighted average
            weighted_avg = [(w1 * x[i]) + (w2 * y[i]) for i in range(len(x))]

            weighted_avg = hv
            
            if param_data_df['algorithm'][i] == 'de':
                param_1 = param_data_df['_hms'][i]
                param_2 = param_data_df['crossover'][i]
                #val = int(min(weighted_avg))
                val = hv
                all_best.append([param_1, param_2, val])
            if param_data_df['algorithm'][i] == 'hs':
                param_1 = param_data_df['_hms'][i]
                param_2 = param_data_df['_hmcr'][i]
                param_3 = param_data_df['_par'][i]
                
               # val = min(x)
               # val = int(min(weighted_avg))
                val = hv
                all_best_hs.append([param_1, param_2, param_3, val])
            if param_data_df['algorithm'][i] == 'sa':
                param_1 = param_data_df['crossover'][i].round(2)
                param_2 = param_data_df['temp_scale'][i].round(2)
                param_3 = param_data_df['steps'][i]
                
               # val = min(x)
                #val =int(min(weighted_avg))
                val = hv
                all_best_sa.append([param_1, param_2, param_3, val])        
                
            if min(x) < spread_checker:
                spread_checker = min(x)
                print('best_folder is', i)
                store_2 = file
            x, y = pareto_frontier(x, y)
            #plot_pareto_frontier(x,y)
            new_x = x+ true_x
            new_y = y + true_y
            true_x, true_y = pareto_frontier(new_x, new_y)
        except:
            print(i)
        
#df_master = pd.concat(df_master, ignore_index=True)
#df_master.to_csv('Z:/master.csv', index=False)
#plot_pareto_frontier(true_x,true_y)


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

    # create the heatmap using seaborn
    sns.heatmap(table, cmap='RdBu', annot=False, fmt=".0f", linewidths=.5, ax=ax)

    # set the axis labels and title
    ax.set_xlabel("Temperature Decreasing Rate and Percentage of Crossover")
    ax.set_ylabel("Temperature Steps")
    ax.set_title("Simulated Annealing Hyperparameter Tuning")
    #ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

def seaborn_plot_hs(data, ax):
    # create a pivot table to aggregate the best solution for each combination of hyperparameters
    table = pd.pivot_table(data, values='best_solution', index=['_hms'], columns=['_hmcr', '_par'], aggfunc='mean')

    # create the heatmap using seaborn
    sns.heatmap(table, cmap='RdBu', annot=False, fmt=".0f", linewidths=.5, ax=ax)

    # set the axis labels and title
    ax.set_xlabel(" Harmony Memory Consideration Rate and Pitch Adjustment Rate")
    ax.set_ylabel("Harmony Memory Size")
    ax.set_title("Harmony Search Hyperparameter Tuning")
    
def seaborn_plot_de(data, ax):
    # Pivot the data to create a heatmap
    df_pivot = data.pivot_table(index='percentage_of_crossover', columns='population_size', values='best_solution', aggfunc='mean')
   # df_pivot = data.pivot_table('percentage_of_crossover', 'population_size', 'best_solution', aggfunc='mean')

    # Create the heatmap using Seaborn
    sns.heatmap(df_pivot, cmap='RdBu', annot=False, fmt='.0f', ax=ax)

    # Set the axis labels
    ax.set_ylabel('Population Size')
    ax.set_xlabel('Percentage of Crossover')
    ax.set_title("Differential Evolution Hyperparameter Tuning")
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
    plt.savefig(f"please{problem_number}.svg")
    # show the plot
    plt.show()

   # plt.savefig(f"please{problem_number}.svg")
    print(1)



# Assuming data1 and data2 are your data sets


def create_bar_plots(data):
    # Generate x-coordinates for the bars
    x_values = range(len(data))

    # Create a bar plot for each element in the array
    for i, value in enumerate(data):
        plt.bar(x_values[i], value)

    # Find the index of the maximum value
    max_index = data.index(max(data))

    # Add callout point for the maximum value
    plt.annotate(text=str(data[max_index]), xy=(x_values[max_index], data[max_index]), xytext=(x_values[max_index], data[max_index] + 1),
                 ha='center', va='bottom', fontsize=8, arrowprops=dict(arrowstyle="->"))

    # Modify x-axis grid segments
    plt.xticks(x_values)

    # Add labels and title
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.title('Bar Plots for Array Elements')

    # Show the plot
    plt.tight_layout()
    plt.show()








for i in folders:
    try:
        file = str(i) +"/log.csv"
        df = pd.read_csv(file)
        #df = df[df['incumbent_pval_exceed'] == 0]
        if problem_number ==4:
            df = df[df['incumbent_bic'] <= 2000]
            
        if problem_number ==3:
            df = df[df['incumbent_bic'] <= 1000] 
            df = df[df['incumbent_bic'] >= 300] 
            df = df[df['incumbent_MSE_TEST'] <= 10]  
            
        if problem_number ==9:
            df = df[df['incumbent_MSE_TEST'] <=1000000]
            df = df[df['incumbent_bic'] <=4500000]
        
            df =df[df['incumbent_MAE'] <= 1000000]
        df = df.reset_index(drop =True)
        x = df['incumbent_bic']
        y = df['incumbent_MSE_TEST']
        
        x, y = pareto_frontier(x, y)
        
        x = scaler.transform(np.array(x).reshape(-1,1))
        y = scaler2.transform(np.array(y).reshape(-1,1))
        
        pareto_front1 = [list(point) for point in zip(x, y)]
        hv = calculate_hypervolume(pareto_front1, [1,1])
        hv_list.append(hv)
        print("Hypervolume: ", hv)

   # print('the spread is ', spread(x,y, true_x, true_y))

    #spread_new = min(x) +min(y)
        if hv > hv_max:
            hv_max = hv
            store = file


    except:
        print('wh')   
    
create_bar_plots(hv_list)
#store = '137/log.csv'
#store = store_2
print('now print the best', store)
#store = '126/log.csv'

df = pd.read_csv(store)
if problem_number ==4:
    df = df[df['incumbent_bic'] <= 2000]
    

    
df =df[df['incumbent_MSE_TEST'] <= -1+10*10**5]
#df = df[df['incumbent_pval_exceed'] == 0]
df = df.reset_index(drop =True)

#df = pd.read_csv(store_2)
x = df['incumbent_bic']
y = df['incumbent_MSE_TEST']
iter = df['iteration']
level = df['incumbent_simple']


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



def boxplot(y, name = None):
    # Create a figure and axis object
    fig, (ax, ax1) = plt.subplots(nrows=2, ncols=1, sharex=False)
    y = y[y <10000]
    # Create the box and whisker plot
    y2 = y[y>np.median(y)]
    y1 = y[y<np.median(y)]
    #ax.boxplot(y)
    bin_edges = [0, 0.1, .5, 1]
    bin_edges =  np.linspace(np.min(y1), np.max(y1), num = 10)
    bin_edges2 = np.linspace(np.min(y2), np.max(y2), num = 10)
    ax.hist(y1, bins=bin_edges)
    ax1.hist(y2, bins=bin_edges2)
    #ax.hist(x=[y],bins=20,edgecolor='black')
    # Add a title and labels to the plot
   
    ax.set_ylabel('Solution Spread Lower Median')
    ax.set_xlabel(name)
    ax1.set_ylabel('Solution Spread Upper Median')
    ax1.set_xlabel(name)

    # Show the plot
    plt.show()
    
def boxplotx(y, name = None):
    # Create a figure and axis object
    fig, ax = plt.subplots()

    # Create the box and whisker plot
   
    bin_edges =  np.linspace(np.min(y), np.max(y), num = 10)
    ax.hist(y, bins=bin_edges)
    #ax.hist(x=[y],bins=20,edgecolor='black')
    # Add a title and labels to the plot
   
    ax.set_ylabel('Solution Spread')
    ax.set_xlabel('BIC')

    # Show the plot
    plt.show()
    



xMax = np.median(x)
yMax = np.median(y)

result = split_lists(x, y, level)
keys = [key for key in result.keys()]

#trace_plot(x,y)
#print(1)
new_pareto(x, y, result, iter)
def just_a_scatter(Xs, Ys, result, iteration, maxX=False, maxY=False, xMax = None, yMax = None):
    '''Pareto frontier selection process'''
    sorted_list = sorted([[Xs[i], Ys[i], iteration[i]] for i in range(len(Xs))], reverse=maxY)
    pareto_front = [sorted_list[0]]
    for pair in sorted_list[1:]:
        if maxY:
            if pair[1] >= pareto_front[-1][1]:
                pareto_front.append(pair)
        else:
            if pair[1] <= pareto_front[-1][1]:
                pareto_front.append(pair)

    '''Plotting process'''
    marker_size = 4
    pf_X = [pair[0] for pair in pareto_front]
    pf_Y = [pair[1] for pair in pareto_front]
    pf_I = [pair[2] for pair in pareto_front]
    print(pf_X)
    print(pf_Y)
    print(pf_I)
    if xMax is None:
        xMax = np.max(pf_X)*2
    if yMax is None:
        yMax = np.max(pf_Y) *2    
    #plt.scatter(pf_X, pf_Y)
    fig, ax = plt.subplots()
   # plt.plot(pf_X, pf_Y, color='red')
   # xlim = ax.get_xlim()

   # ylim = ax.get_ylim()
   
    ax.set_xlim(left = np.min(pf_X)*.99, right =  xMax)
    ax.set_ylim(bottom = np.min(pf_Y)*.95, top = yMax)
    for key, (x, y) in result.items():
       
            
        #plt.scatter(x, y, c= color, label = '%s ' % keys[a], marker=marker_shape)
        plt.scatter(x, y, c = 'black', alpha = 0.5)

    # Define a color map and get a list of colors based on the number of keys
   
    
   
        
    plt.xlabel("BIC")
    plt.ylabel("MSE")
    plt.legend()
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
          ncol=3, fancybox=True, shadow=True)
    
    plt.show()
    print(1)
#boxplot(y, 'MSE')
#boxplot(x, 'BIC')
#new_scatter(x,y,result, iter)
#x, y = pareto_frontier(x, y)
just_a_scatter(x, y, result, iter, False, False, xMax, yMax)
for i  in folders:
    if i > -1:
        
        store = str(i) +'/log.csv'
        
        df = pd.read_csv(store)
        #df = df[df['incumbent_pval_exceed'] == 0]
        #if problem_number ==9:
        #    df = df[df['incumbent_MAE'] <=10000]
        #    df = df[df['incumbent_bic'] <=2500000]
        df = df.reset_index(drop =True)
        #df = pd.read_csv(store_2)
        x = df['incumbent_bic']
        y = df['incumbent_RMSE_TEST']
        iter = df['iteration']
        level = df['incumbent_simple']
        result = split_lists(x, y, level)
        keys = [key for key in result.keys()]

#trace_plot(x,y)
#print(1)
        try:
            new_pareto(x, y, result, iter)
            print(1)
        except Exception as e:
            print(1)
        
print('happy')

    
