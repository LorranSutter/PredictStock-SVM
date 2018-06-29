import os
import json

db_dir = 'db/results/C_gamma/'
tickers = ['ABBV','AMZN','GOOGL','TSLA','ZTS']

# GOOGL exclude 10 of 3's

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
                

def worseBetterValues(ticker, files):
    # d = {'file' : [], 'best' : [], 'worst' : []}
    # d = {'file' : [], 'best' : -1}
    d = dict()
    for f1 in files:
        with open(db_dir + ticker + '/' + f1, 'r') as f:
            for line in f:
                line = json.loads(line)
                if 'ERROR' in line.keys():
                    continue
                vals = line['preds_comp']
                best = max(vals)
                worst = min(vals)
                if f1 not in d.keys():
                    d[f1] = [best,worst]
                elif best > d[f1][0]:
                    d[f1][0] = best
                elif worst < d[f1][1]:
                    d[f1][1] = worst
    return d

id_t = 4

files = os.listdir(db_dir + tickers[id_t])
files = [f for f in files if 'json' in f]

# includePredNxtDay(tickers[id_t], files, 7)
# p = missingValues(tickers[id_t], files)
d = worseBetterValues(tickers[id_t], files)

# files = os.listdir(db_dir + tickers[0])
# files = [f for f in files if '_test' in f]
# for f in files:
#     os.system('rm {0}{1}/{2}'.format(db_dir,tickers[0],f))
#     print(f + ' removed!')