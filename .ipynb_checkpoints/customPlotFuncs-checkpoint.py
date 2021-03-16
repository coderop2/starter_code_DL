import pandas as pd
import matplotlib.cm as cm
import matplotlib
import numpy as np
import matplotlib.pyplot as plt

def plot_clustered_stacked(dfall, labels=None, title="multiple stacked bar plot",  H="/", **kwargs):
    """Given a list of dataframes, with identical columns and index, create a clustered stacked bar plot. 
labels is a list of the names of the dataframe, used for the legend
title is a string for the title of the plot
H is the hatch used for identification of the different dataframe"""

    n_df = len(dfall)
    n_col = len(dfall[0].columns) 
    n_ind = len(dfall[0].index)
#     figure = plt.figure(figsize=(10,10))
#     axe = plt.subplot(111, figure = matplotlib.figure.Figure(figsize=(10,10)))
    _,axe = plt.subplots(nrows=1, figsize=(15, 13))
#     axe.set_figheight(15)
#     axe.set_figwidth(15)
    for df in dfall : # for each data frame
        axe = df.plot(kind="bar",
                      linewidth=0,
                      stacked=True,
                      ax=axe,
                      legend=False,
                      grid=False,
                      **kwargs)  # make bar plots

    h,l = axe.get_legend_handles_labels() # get the handles we want to modify
#     print(h,l)
    for i in range(0, n_df * n_col, n_col): # len(h) = n_col * n_df
        for j, pa in enumerate(h[i:i+n_col]):
            for rect in pa.patches: # for each index
                rect.set_x(rect.get_x() + 1 / float(n_df + 1) * i / float(n_col))
                rect.set_hatch(H * int(i / n_col)) #edited part     
                rect.set_width(1 / float(n_df + 1))

    axe.set_xticks((np.arange(0, 2 * n_ind, 2) + 1 / float(n_df + 1)) / 2.)
    names = [ '\n'.join(wrap(l, 20)) for l in df.index ]
    axe.set_xticklabels(names, rotation = 0)
    axe.set_title(title)
    
    # Add invisible data to add another legend
    n=[]        
    for i in range(n_df):
        n.append(axe.bar(0, 0, color="gray", hatch=H * i * 2))

    l1 = axe.legend(h[:n_col], l[:n_col], loc=[1.01, 0.5])
    if labels is not None:
        l2 = plt.legend(n, labels, loc=[1.01, 0.1]) 
    axe.add_artist(l1)
    axe.figure.savefig('stackedGraph.png')
    return axe

# create fake dataframes
df1 = x.copy()
df2 = x2.copy()
df3 = x3.copy()

# Then, just call :
plot_clustered_stacked([df1, df2, df3],['26-35' ,'36-45', '17-25'])

def groupedPie(groupedData):
    names = list(grouped.index)
    names = [ '\n'.join(wrap(l, 60)) for l in names ]
    size = list(grouped[grouped.columns[0]].values)

    # Create a circle at the center of the plot
    fig = plt.figure(figsize=(10,10))
    my_circle = plt.Circle( (0,0), 0.4, color='white')

    # Custom wedges
    plt.pie(size[:6], labels=names[:6], wedgeprops = { 'linewidth' : 7, 'edgecolor' : 'white' })
    p = plt.gcf()
    p.gca().add_artist(my_circle)
    plt.savefig("piechart.png", bbox_inches='tight')
    plt.show()
    
grouped = df.groupby("Group Column").agg({"Aggregate Column":["count"]})
groupedPie(grouped)