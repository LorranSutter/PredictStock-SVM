import pandas as pd


def SMA(df, n):
    '''
    Simple Moving Average
    '''
    df['SMA_' + str(n)] = pd.Series.rolling(df['Close'], n).mean()


def EMA(df, n):
    '''
    Exponential Moving Average
    '''
    df['EMA_' + str(n)] = pd.Series.ewm(df['Close'],
                                        span=n, min_periods=n - 1).mean()


def MOM(df, n):
    '''
    Momentum
    '''
    df['MOM_' + str(n)] = pd.Series(df['Close'].diff(n))


def ROC(df, n):
    '''
    Rate of Change
    '''
    M = df['Close'].diff(n - 1)
    N = df['Close'].shift(n - 1)

    df['ROC_' + str(n)] = pd.Series(M / N)


def ATR(df, n):
    '''
    Average True Range
    '''
    TR_l = [0]
    for k in range(len(df.index)-1):
        TR = max(df['High'][k + 1], df['Close'][k]) - \
            min(df['Low'][k + 1], df['Close'][k])
        TR_l.append(TR)

    TR_s = pd.Series(TR_l)

    df['ATR_' + str(n)] = pd.Series.ewm(TR_s, span=n,
                                        min_periods=n).mean().values


def BBANDS(df, n, multiplier=2, middle=False):
    '''
    Bollinger Bands
    '''
    ma = pd.Series.rolling(df['Close'], n).mean()
    msd = pd.Series.rolling(df['Close'], n).std()

    b1 = 4 * msd / ma
    b2 = (df['Close'] - ma + multiplier * msd) / (4 * msd)

    df['BBANDSup_' + str(n)] = b1

    if middle:
        df['BBANDSmiddle_' + str(n)] = ma

    df['BBANDSdown_' + str(n)] = b2


def PPSR(df):
    '''
    Pivot Points, Supports and Resistances
    '''
    pp = pd.Series((df['High'] + df['Low'] + df['Close']) / 3)

    s1 = pd.Series(2 * pp - df['High'])
    s2 = pd.Series(pp - df['High'] + df['Low'])
    s3 = pd.Series(df['Low'] - 2 * (df['High'] - pp))

    r1 = pd.Series(2 * pp - df['Low'])
    r2 = pd.Series(pp + df['High'] - df['Low'])
    r3 = pd.Series(df['High'] + 2 * (pp - df['Low']))

    df['PP'] = pp
    df['S1'] = s1
    df['S2'] = s2
    df['S3'] = s3
    df['R1'] = r1
    df['R2'] = r2
    df['R3'] = r3


def PPSRFIBO(df):
    '''
    Pivot Points, Supports and Resistances Fibonacci
    '''
    pp = pd.Series((df['High'] + df['Low'] + df['Close']) / 3)

    s1_fibo = pd.Series(pp - 0.382 * (df['High'] - df['Low']))
    s2_fibo = pd.Series(pp - 0.618 * (df['High'] - df['Low']))
    s3_fibo = pd.Series(pp - 1 * (df['High'] - df['Low']))

    r1_fibo = pd.Series(pp + 0.382 * (df['High'] - df['Low']))
    r2_fibo = pd.Series(pp + 0.618 * (df['High'] - df['Low']))
    r3_fibo = pd.Series(pp + 1 * (df['High'] - df['Low']))

    df['PP'] = pp
    df['R1fibo'] = r1_fibo
    df['S1fibo'] = s1_fibo
    df['R2fibo'] = r2_fibo
    df['S2fibo'] = s2_fibo
    df['R3fibo'] = r3_fibo
    df['S3fibo'] = s3_fibo


def STOK(df):
    '''
    Stochastic oscillator %K
    '''
    df['STOK'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'])


def STO(df, n):
    '''
    Stochastic oscillator %D
    '''
    SOk = pd.Series((df['Close'] - df['Low']) / (df['High'] - df['Low']))

    df['STO_' + str(n)] = pd.Series.ewm(SOk, span=n, min_periods=n - 1).mean()


def TRIX(df, n):
    '''
    Trix
    '''
    ex1 = pd.Series.ewm(df['Close'], span=n, min_periods=n - 1).mean()
    ex2 = pd.Series.ewm(ex1, span=n, min_periods=n - 1).mean()
    ex3 = pd.Series.ewm(ex2, span=n, min_periods=n - 1).mean()

    trix = [0]
    for k in range(len(df.index) - 1):
        roc = (ex3[k + 1] - ex3[k]) / ex3[k]
        trix.append(roc)

    df['TRIX_' + str(n)] = trix


def ADX(df, n, n_ADX):
    '''
    Average Directional Movement Index
    '''
    UpI = []
    DoI = []
    for k in range(len(df.index) - 1):
        UpMove = df['High'][k + 1] - df['High'][k]
        DoMove = df['Low'][k] - df['Low'][k + 1]

        if UpMove > DoMove and UpMove > 0:
            UpD = UpMove
        else:
            UpD = 0

        UpI.append(UpD)

        if DoMove > UpMove and DoMove > 0:
            DoD = DoMove
        else:
            DoD = 0

        DoI.append(DoD)

    TR_l = [0]
    for k in range(len(df.index) - 1):
        TR = max(df['High'][k + 1], df['Close'][k]) - \
            min(df['Low'][k + 1], df['Close'][k])
        TR_l.append(TR)

    TR_s = pd.Series(TR_l)
    atr = pd.Series.ewm(TR_s, span=n, min_periods=n).mean()
    UpI = pd.Series(UpI)
    DoI = pd.Series(DoI)
    PosDI = pd.Series.ewm(UpI, span=n, min_periods=n - 1).mean() / atr
    NegDI = pd.Series.ewm(DoI, span=n, min_periods=n - 1).mean() / atr

    df['ADX_' + str(n) + '_' + str(n_ADX)] = pd.Series.ewm(abs(PosDI - NegDI) /
                                                           (PosDI + NegDI), span=n_ADX, min_periods=n_ADX - 1).mean().values


def MACD(df, n_fast=12, n_slow=26):
    '''
    MACD, MACD Signal and MACD difference
    '''
    emaFast = pd.Series.ewm(df['Close'], span=n_fast,
                            min_periods=n_slow - 1).mean()
    emaSlow = pd.Series.ewm(df['Close'], span=n_slow,
                            min_periods=n_slow - 1).mean()
    macd = pd.Series(emaFast - emaSlow)
    macdSign = pd.Series.ewm(macd, span=9, min_periods=8).mean()
    macdDiff = pd.Series(macd - macdSign)

    df['MACD_' + str(n_fast) + '_' + str(n_slow)] = macd
    df['MACDsignal_' + str(n_fast) + '_' + str(n_slow)] = macdSign
    df['MACDdiff_' + str(n_fast) + '_' + str(n_slow)] = macdDiff


def MASS(df):
    '''
    Mass Index
    '''
    Range = df['High'] - df['Low']
    ex1 = pd.Series.ewm(Range, span=9, min_periods=8).mean()
    ex2 = pd.Series.ewm(ex1, span=9, min_periods=8).mean()
    Mass = ex1 / ex2
    Mass = pd.Series.rolling(Mass, 25).sum()

    df['MASS'] = Mass


def VORTEX(df, n):
    '''
    Vortex Indicator: http://www.vortexindicator.com/VFX_VORTEX.PDF
    '''
    tr = [0]
    for k in range(len(df.index) - 1):
        Range = max(df['High'][k + 1], df['Close'][k]) - \
            min(df['Low'][k + 1], df['Close'][k])
        tr.append(Range)

    vm = [0]
    for k in range(len(df.index) - 1):
        Range = abs(df['High'][k + 1] - df['Low'][k]) - \
            abs(df['Low'][k + 1] - df['High'][k])
        vm.append(Range)

    vm = pd.Series(vm)
    tr = pd.Series(tr)
    vi = pd.Series.rolling(vm, n).sum() / pd.Series.rolling(tr, n).sum()

    df['VORTEX_' + str(n)] = vi.values


def KST(df, r1, r2, r3, r4, n1, n2, n3, n4, sigLen=9):
    '''
    KST Oscillator
    '''
    M = df['Close'].diff(r1 - 1)
    N = df['Close'].shift(r1 - 1)
    roc1 = M / N

    M = df['Close'].diff(r2 - 1)
    N = df['Close'].shift(r2 - 1)
    roc2 = M / N

    M = df['Close'].diff(r3 - 1)
    N = df['Close'].shift(r3 - 1)
    roc3 = M / N

    M = df['Close'].diff(r4 - 1)
    N = df['Close'].shift(r4 - 1)
    roc4 = M / N

    kst = pd.Series.rolling(roc1, n1).sum() + \
        pd.Series.rolling(roc2, n2).sum() * 2 + \
        pd.Series.rolling(roc3, n3).sum() * 3 + \
        pd.Series.rolling(roc4, n4).sum() * 4

    params = '_'.join(map(str, [r1, r2, r3, r4, n1, n2, n3, n4]))
    df['KST_' + params] = kst
    # df['KST_' + params + '_SMA_' + str(sigLen)] = pd.Series.rolling(df['Close'], sigLen).mean()


def RSI(df, n):
    '''
    Relative Strength Index
    '''
    UpI = [0]
    DoI = [0]
    for k in range(len(df.index)-1):
        UpMove = df['High'][k + 1] - df['High'][k]
        DoMove = df['Low'][k] - df['Low'][k + 1]
        if UpMove > DoMove and UpMove > 0:
            UpD = UpMove
        else:
            UpD = 0
        UpI.append(UpD)
        if DoMove > UpMove and DoMove > 0:
            DoD = DoMove
        else:
            DoD = 0
        DoI.append(DoD)

    UpI = pd.Series(UpI)
    DoI = pd.Series(DoI)
    PosDI = pd.Series.ewm(UpI, span=n, min_periods=n - 1).mean()
    NegDI = pd.Series.ewm(DoI, span=n, min_periods=n - 1).mean()

    df['RSI_' + str(n)] = pd.Series(PosDI / (PosDI + NegDI)).values


def TSI(df, r, s):
    '''
    True Strength Index
    '''
    M = pd.Series(df['Close'].diff(1))
    aM = abs(M)
    ema1 = pd.Series.ewm(M, span=r, min_periods=r - 1).mean()
    aEMA1 = pd.Series.ewm(aM, span=r, min_periods=r - 1).mean()
    ema2 = pd.Series.ewm(ema1, span=s, min_periods=s - 1).mean()
    aEMA2 = pd.Series.ewm(aEMA1, span=s, min_periods=s - 1).mean()

    df['TSI_' + str(r) + '_' + str(s)] = pd.Series(ema2 / aEMA2)


def ACCDIST(df, n):
    '''
    Accumulation/Distribution
    '''
    ad = (2 * df['Close'] - df['High'] - df['Low']) / \
        (df['High'] - df['Low']) * df['Volume']
    roc = ad.diff(n - 1) / ad.shift(n - 1)

    df['ACCDIST_' + str(n)] = roc


def CHAIKIN(df):
    '''
    Chaikin Oscillator
    '''
    ad = (2 * df['Close'] - df['High'] - df['Low']) / \
        (df['High'] - df['Low']) * df['Volume']
    Chaikin = pd.Series.ewm(ad, span=3, min_periods=2).mean(
    ) - pd.Series.ewm(ad, span=10, min_periods=9).mean()

    df['CHAIKIN'] = Chaikin


def MFI(df, n):
    '''
    Money Flow Index and Ratio
    '''
    pp = (df['High'] + df['Low'] + df['Close']) / 3
    PosMF = [0]
    for k in range(len(df.index) - 1):
        if pp[k + 1] > pp[k]:
            PosMF.append(pp[k + 1] * df['Volume'][k + 1])
        else:
            PosMF.append(0)

    PosMF = pd.Series(PosMF)
    TotMF = pp * df['Volume']

    # .values was used beacause in a nonsense way PosMF/TotMF was
    # generating a double size dataFrame and the first half had datas as index
    # ! We got an RuntimeWarning because division buy zero, but it still works
    mfr = pd.Series(PosMF.values / TotMF.values)

    df['MFI_' + str(n)] = pd.Series.rolling(mfr, n).mean().values


def OBV(df, n):
    '''
    On-balance Volume
    '''
    obv = [0]
    for k in range(len(df.index) - 1):
        if df['Close'][k + 1] - df['Close'][k] > 0:
            obv.append(df['Volume'][k + 1])
        if df['Close'][k + 1] - df['Close'][k] == 0:
            obv.append(0)
        if df['Close'][k + 1] - df['Close'][k] < 0:
            obv.append(-df['Volume'][k + 1])

    obv = pd.Series(obv)
    df['OBV_' + str(n)] = pd.Series.rolling(obv, n).mean().values


def FORCE(df, n):
    '''
    Force Index
    '''
    df['FORCE_' + str(n)] = pd.Series(df['Close'].diff(n)
                                      * df['Volume'].diff(n))
    return pd.Series(df['Close'].diff(n) * df['Volume'].diff(n)).values


def EOM(df, n):
    '''
    Ease of Movement
    '''
    EoM = (df['High'].diff(1) + df['Low'].diff(1)) * \
        (df['High'] - df['Low']) / (2 * df['Volume'])

    df['EOM_' + str(n)] = pd.Series.rolling(EoM, n).mean()


def CCI(df, n):
    '''
    Commodity Channel Index
    '''
    pp = (df['High'] + df['Low'] + df['Close']) / 3

    df['CCI_' + str(n)] = pd.Series((pp - pd.Series.rolling(pp,
                                                            n).mean()) / pd.Series.rolling(pp, n).std())


def COPP(df, n):
    '''
    Coppock Curve
    '''
    M = df['Close'].diff(int(n * 11 / 10) - 1)
    N = df['Close'].shift(int(n * 11 / 10) - 1)
    roc1 = M / N
    M = df['Close'].diff(int(n * 14 / 10) - 1)
    N = df['Close'].shift(int(n * 14 / 10) - 1)
    roc2 = M / N

    df['COPP_' + str(n)] = pd.Series.ewm(roc1 + roc2,
                                         span=n, min_periods=n).mean()


def KELCH(df, n):
    '''
    Keltner Channel
    '''
    kelChM = pd.Series.rolling(
        (df['High'] + df['Low'] + df['Close']) / 3, n).mean().values
    kelChU = pd.Series.rolling(
        (4 * df['High'] - 2 * df['Low'] + df['Close']) / 3, n).mean().values
    kelChD = pd.Series.rolling(
        (-2 * df['High'] + 4 * df['Low'] + df['Close']) / 3, n).mean().values

    df['KELCHmiddle_' + str(n)] = kelChM
    df['KELCHup_' + str(n)] = kelChU
    df['KELCHdown_' + str(n)] = kelChD


def ULTOSC(df):
    '''
    Ultimate Oscillator
    '''
    TR_l = [0]
    BP_l = [0]
    for k in range(len(df.index) - 1):
        TR = max(df['High'][k + 1], df['Close'][k]) - \
            min(df['Low'][k + 1], df['Close'][k])
        TR_l.append(TR)
        BP = df['Close'][k + 1] - min(df['Low'][k + 1], df['Close'][k])
        BP_l.append(BP)

    TR_l = pd.Series(TR_l)
    BP_l = pd.Series(BP_l)
    UltO = pd.Series((4 * pd.Series.rolling(BP_l,  7).sum() / pd.Series.rolling(TR_l,  7).sum()) +
                     (2 * pd.Series.rolling(BP_l, 14).sum() / pd.Series.rolling(TR_l, 14).sum()) +
                     (pd.Series.rolling(BP_l, 28).sum() / pd.Series.rolling(TR_l, 28).sum()))

    df['ULTOSC'] = UltO.values


def DONCH(df, n):
    '''
    Donchian Channel
    '''
    DC_l = [0 for k in range(n-1)]
    for k in range(len(df.index) - n + 1):
        DC = max(df['High'].iloc[k:k + n]) - min(df['Low'].iloc[k:k + n])
        DC_l.append(DC)

    DonCh = pd.Series(DC_l)
    DonCh = DonCh.shift(n - 1)

    df['DONCH_' + str(n)] = DonCh.values
