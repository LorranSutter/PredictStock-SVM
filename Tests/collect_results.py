import os
import json
import matplotlib.pyplot as plt

import itertools as it

from collections import Counter

db_dir = 'db/results/C_gamma/'
tickers = ['ABBV','AMZN','GOOGL','TSLA','ZTS']
days = [3,5,7,10]

def missingValues(ticker, files):
    p = [0,0,0,0,0] # 3,5,7,10,none
    for f1 in files:
        with open(db_dir + ticker + '/' + f1, 'r') as f:
            for line in f:
                if 'predict_nxt_day' not in line:
                    p[-1] += 1
                    break
                else:
                    line = json.loads(line)
                    if int(line['predict_nxt_day']) == 3:
                        p[0] += 1
                        break
                    elif int(line['predict_nxt_day']) == 5:
                        p[1] += 1
                        break
                    elif int(line['predict_nxt_day']) == 7:
                        p[2] += 1
                        break
                    elif int(line['predict_nxt_day']) == 10:
                        p[3] += 1
                        break
    return p

def includePredNxtDay(ticker, files, day):
    for f1 in files:
        with open(db_dir + ticker + '/' + f1, 'r') as f:
            records = []
            include = False
            for line in f:
                if 'predict_nxt_day' not in line:
                    include = True
                    line = json.loads(line)
                    line['predict_nxt_day'] = day
                    line['cluster_number'] = '2'
                    records.append(line)
        if include:
            try:
                os.remove('{0}/{1}{2}/{3}'.format(os.getcwd(),db_dir,ticker,f1))
                print(f1 + ' removed')
                with open(db_dir + ticker + '/' + f1, 'a') as f:
                    for rec in records:
                        json.dump(rec, f)
                        f.write('\n')
            except Exception as e:
                print(e)

def worseBetterValues(ticker, files, day):
    d = dict()
    for f1 in files:
        with open(db_dir + ticker + '/' + f1, 'r') as f:
            for line in f:
                line = json.loads(line)
                if 'ERROR' in line.keys():
                    continue
                if int(line['predict_nxt_day']) != day:
                    break

                vals = line['preds_comp']
                best = vals[-4]
                worst = vals[-4]

                if f1 not in d.keys():
                    d[f1] = {'C_best' : line['C'],
                             'gamma_best' : line['gamma'],
                             'best' : best ,
                             'C_worst' : line['C'],
                             'gamma_worst' : line['gamma'],
                             'worst' : worst}
                elif best > d[f1]['best']:
                    d[f1]['C_best'] = line['C']
                    d[f1]['gamma_best'] = line['gamma']
                    d[f1]['best'] = best
                elif worst < d[f1]['worst']:
                    d[f1]['C_worst'] = line['C']
                    d[f1]['gamma_worst'] = line['gamma']
                    d[f1]['worst'] = worst
    return d

def printAllBestWorstValues():
    for ticker in tickers:
        files = os.listdir(db_dir + ticker)
        files = [f for f in files if 'json' in f]

        print(ticker)
        for day in [3,5,7,10]:
            d = worseBetterValues(ticker, files, day)

            bests = sorted([k['best'] for k in d.values()], reverse = True)
            worsts = sorted([k['worst'] for k in d.values()])

            print("     {0} {1:.5f} {2:.5f} {3:.5f} {4:.5f}".
                        format(day, worsts[0]*100, worsts[3]*100, bests[0]*100, bests[3]*100))

def writeAllBestWorst():
    for ticker in tickers:
        files = os.listdir(db_dir + ticker)
        files = [f for f in files if 'json' in f]

        print(ticker)
        for day in [3,5,7,10]:
            d = worseBetterValues(ticker, files, day)

            bests = sorted(d.values(), key = lambda x: x['best'], reverse = True)
            worsts = sorted(d.values(), key = lambda x: x['worst'])

            first_best   = {'C_best' : bests[0]['C_best'], 'gamma_best' : bests[0]['gamma_best'], 'best' : bests[0]['best']}
            second_best  = {'C_best' : bests[3]['C_best'], 'gamma_best' : bests[3]['gamma_best'], 'best' : bests[3]['best']}
            first_worst  = {'C_worst' : worsts[0]['C_worst'], 'gamma_worst' : worsts[0]['gamma_worst'], 'worst' : worsts[0]['worst']}
            second_worst = {'C_worst' : worsts[3]['C_worst'], 'gamma_worst' : worsts[3]['gamma_worst'], 'worst' : worsts[3]['worst']}

            with open(db_dir + ticker + '/BestWorsts/bestWorsts_{}_days.json'.format(day), 'w') as f:
                json.dump(first_best,f)
                f.write('\n')
                json.dump(second_best,f)
                f.write('\n')
                json.dump(first_worst,f)
                f.write('\n')
                json.dump(second_worst,f)
                f.write('\n')

def funcname(ticker, day):
    files = os.listdir(db_dir + ticker)
    files = [f for f in files if 'json' in f and ticker in f]
    params = [[0,0] for k in range(72)]
    for f1 in files:
        with open(db_dir + ticker + '/' + f1, 'r') as f:
            for k, line in enumerate(f):
                line = json.loads(line)
                if 'ERROR' in line.keys():
                    continue
                if int(line['predict_nxt_day']) != day:
                    break

                try:
                    vals = line['preds_comp']
                    best = vals[-4]
                    worst = vals[-4]

                    params[k][0] += best
                    params[k][1] += 1
                except:
                    print(f1)

    return params

def getAllDayValues(ticker, day):
    files = os.listdir(db_dir + ticker)
    files = [f for f in files if 'json' in f and ticker in f]
    res = []

    for f1 in files:
        with open(db_dir + ticker + '/' + f1, 'r') as f:
            for line in f:
                line = json.loads(line)
                if 'ERROR' in line.keys():
                    continue
                try:
                    if int(line['predict_nxt_day']) != day:
                        break
                except:
                    print(line)

                vals = line['preds_comp']

                res.append(vals[-4])
    return res

id_t = 1

res = {d : [] for d in days}
for k, ticker in enumerate(tickers):
    for day in days:
        res[day].append(getAllDayValues(ticker, day))

# fig = plt.figure()
# ax = fig.add_subplot(111)

# res1 = [np.random.choice(r, size = 25, replace = False) for r in res]

box_props = {'color' : 'b'}
median_props = {'color' : 'r'}
# ax.set_xticks(range(len(tickers)), tickers)
for k,day in enumerate(days):
    plt.figure(k)
    plt.boxplot(res[day], sym = 'r.', boxprops = box_props, medianprops = median_props)
    plt.xticks(range(1,len(tickers)+1), tickers)
    plt.xlabel("Tickers")
    plt.ylabel("Acurácia")
    # plt.title("{} dias de predição".format(day))

# plt.figure(2)
# plt.boxplot(res, sym = 'r.', boxprops = box_props, medianprops = median_props)
# plt.xticks(range(1,len(tickers)+1), tickers)
# plt.xlabel("Tickers")
# plt.ylabel("Acurácia")

plt.show()

# params = funcname('ABBV',3)

# params2 = [p[0]/p[1] for p in params if p[1] != 0]

# C_range = [2e-5*100**k for k in range(9)] # Max 2e11
# gamma_range = [2e-15*100**k for k in range(8)] # Max 2e-1

# C_range = ["{0:.0e}".format(k) for k in C_range]
# gamma_range = ["{0:.0e}".format(k) for k in gamma_range]

# res = []
# for k in it.product(C_range, gamma_range):
#     res.append(k[0] + '_' + k[1])

# plt.xticks(list(range(72)), res, rotation = 45, rotation_mode = "anchor")
# plt.plot(range(72),params2)
# plt.show()

# files = os.listdir(db_dir + tickers[0])
# files = [f for f in files if '_test' in f]
# for f in files:
#     os.system('rm {0}{1}/{2}'.format(db_dir,tickers[0],f))
#     print(f + ' removed!')
