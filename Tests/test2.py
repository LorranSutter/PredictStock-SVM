import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.decomposition import FastICA

ticker = 'TSLA2016'
df = pd.read_csv('db' + '/{0}.csv'.format(ticker), parse_dates = True)

df.set_index('Date', inplace = True)

if 'Volume' in df.columns:
    df = df[df['Volume'] != 0]

df['ids'] = np.linspace(0,1,len(df))

df['Close'] = df['Close'].apply(lambda X: (X - min(df['Close']))/(max(df['Close']) - min(df['Close'])))

# X[:,1] = (X[:,1] - min(X[:,1]) )/( max(X[:,1]) - min(X[:,1]))
X = np.array(list(zip(df['ids'],df['Close']))) * np.random.rand(len(df), 2)

ica = FastICA()
S_ica = ica.fit(X)

res = S_ica.transform(X)

plt.scatter(X[:,0], X[:,1])
plt.scatter(res[:,0], res[:,1])

plt.show()