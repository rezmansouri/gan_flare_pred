import numpy as np

EPS = 1e-9

def conf_mat(yhat, y):
    # 0 = positive, 1 = negative
    yhat = np.array(yhat)
    y = np.array(y)
    tp = ((y == 0) & (yhat == 0)).sum()
    fn = ((y == 0) & (yhat == 1)).sum()
    fp = ((y == 1) & (yhat == 0)).sum()
    tn = ((y == 1) & (yhat == 1)).sum()
    return tp, fp, tn, fn

def tss(TP, FP, TN, FN):
    '''empty'''
    tp_rate = TP / float(TP + FN) if TP > 0 else 0  
    fp_rate = FP / float(FP + TN) if FP > 0 else 0
    return tp_rate - fp_rate


def hss(TP, FP, TN, FN):
    N = TN + FP
    P = TP + FN
    return (2 * (TP * TN - FN * FP)) / float((P * (FN + TN) + (TP + FP) * N))