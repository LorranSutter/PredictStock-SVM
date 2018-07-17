import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

dirRes = 'db/results/C_gamma/{0}/BestWorsts/'
tickers = ('ABBV', 'AMZN', 'GOOGL', 'TSLA', 'ZTS')
days = (3,5,7,10)

bestWorsts = pd.DataFrame(columns = ['ticker','type','value','cluster_number'])

for ticker in tickers:
    for f1 in os.listdir(dirRes.format(ticker)):
        with open(dirRes.format(ticker) + f1, 'r') as f2:
            day = int(f1.split('_')[1])
            for line in f2:
                line = json.loads(line)

                if 'best' in line.keys():
                    line = {'day' : day, 'type' : 'best', 'value' : line['best'], 'cluster_number' : line['cluster_number']}
                else:
                    line = {'day' : day, 'type' : 'worst', 'value' : line['worst'], 'cluster_number' : line['cluster_number']}
                
                line['ticker'] = ticker

                bestWorsts = bestWorsts.append(pd.DataFrame([line.values()], columns = line.keys()), ignore_index = True)


ind = np.arange(5)
width = 0.37
colors = ["gray", "deepskyblue", "orange", "limegreen", "orchid"]

df_type = bestWorsts.loc[bestWorsts['type'] == 'worst']
sub = (221,222,223,224)

def autolabel(rects, colors):
    for k, rect in enumerate(rects):
        height = rect.get_height()
        height_str = '{0:.2f}'.format(height)
        ax.text(rect.get_x() + rect.get_width()/2., 1.02*height, height_str, \
                color = colors[k], fontweight = 'bold', \
                ha='center', va='bottom')

fig = plt.figure()

for k in range(len(days)):
    df_day = df_type.loc[df_type['day'] == days[k]]
    two_clusters   = df_day.loc[df_day['cluster_number'] == '2']
    three_clusters = df_day.loc[df_day['cluster_number'] == '3']

    ax = plt.subplot(sub[k])
    rects1 = ax.bar(ind, two_clusters['value'].values * 100, width, edgecolor = 'white', color = colors, hatch = "//")
    rects2 = ax.bar(ind + width, three_clusters['value'].values * 100, width, color = colors)

    ax.set_title("{} Dias de Predição".format(days[k]))

    ax.set_ylabel("Acurácia (%)")
    ax.set_xlabel("Tickers")

    ax.set_xticks(ind + width/2)
    ax.set_xticklabels(tickers)

    ax.set_yticks(np.arange(0,110,10))

    autolabel(rects1, colors)
    autolabel(rects2, colors)

leg = fig.legend((rects1[0], rects2[0]), \
            ['2 clusters', '3 clusters'], \
            loc = 'lower center',  \
            ncol = 2, fancybox = True, shadow = True, prop = {'size' : 13})

leg.legendHandles[0].set_color('black')
leg.legendHandles[0].set(edgecolor='white', hatch = '///')
leg.legendHandles[1].set_color('black')

plt.show()


def plotBestAccuracy():
    ABBV = [79.45205, 77.88018, 80.00000, 87.73585]
    AMZN = [81.55556, 78.06236, 81.13839, 84.77044]
    GOOGL = [78.52349, 76.93603, 78.71622, 73.68421]
    TSLA = [82.72727, 83.84146, 81.90184, 77.08978]
    ZTS = [83.71041, 80.36530, 86.63594, 83.64486]

    ind = np.arange(4)
    width = 0.15

    fig, ax = plt.subplots()

    rectsABBV = ax.bar(ind, ABBV, width, color='gray')
    rectsAMZN = ax.bar(ind + width, AMZN, width, color='deepskyblue')
    rectsGOOGL = ax.bar(ind + 2*width, GOOGL, width, color='orange')
    rectsTSLA = ax.bar(ind + 3*width, TSLA, width, color='limegreen')
    rectsZTS = ax.bar(ind + 4*width, ZTS, width, color='orchid')

    ax.set_title("Melhores Resultados das Predições", y = 1.05)

    ax.set_ylabel("Acurácia")
    ax.set_xlabel("Dias de predição")

    ax.set_xticks(ind + width*2)
    ax.set_xticklabels(('3','5','7','10'))

    ax.legend((rectsABBV[0], rectsAMZN[0], rectsGOOGL[0], rectsTSLA[0], rectsZTS[0]), \
            tickers, \
            loc='upper center', bbox_to_anchor=(0.5, 1.05),
            ncol=5, fancybox=True, shadow=True)

    plt.show()
