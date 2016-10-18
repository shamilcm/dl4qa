import logging
from collections import defaultdict


def calc_accuracy(probs, labels):
    """
    Written by Tapas. 
    """
    correct_1 = 0
    total_1=0
    correct_0=0
    total_0=0
    cut_off=0.5
    for i in range(0, len(probs)):
        if labels[i]==1:
            total_1 +=1
        if probs[i] >= cut_off and labels[i] == 1:
            correct_1 += 1
        if  labels[i]==0:
            total_0 +=1
        if probs[i] < cut_off and labels[i] == 0:
            correct_0 += 1
    return float(correct_1) / total_1, float(correct_0) / total_0
    #print "Avg. Accuracy: ", (float(correct_1) / total_1 + float(correct_0) / total_0)/2

   # print "Finished......"

def print_accuracy(probs, labels):
    accuracy1, accuracy0 = calc_accuracy(probs, labels)
    print "Accuracy 1: ", accuracy1
    print "Accuracy 0: ", accuracy0
    print "Avg. Accuracy: ", (accuracy0 + accuracy1)/2

def get_preds(ref_fname, probs):
    preds = defaultdict(list)
    with open(ref_fname) as f:
        for line,prob in zip(f,probs):
            line = line.strip()
            ids = line.split()
            qid = int(ids[0])
            aid = int(ids[2])
            label = int(ids[3])
            preds[ids[0]].append((aid, label,prob))
    return preds

def calc_mean_avg_prec(preds):
    """
    skip all questions w/o correct answers
    and all questions w/ only correct answers
    """
    mean_avg_prec, relQ = 0.0, 0.0
    for pred in preds.values():
        cnt = 0
        for tri in pred: cnt += tri[1]
        if cnt == 0 or cnt == len(pred): continue
        sorted_pred = sorted(pred, key=lambda res: res[1])
        sorted_pred = sorted(sorted_pred, key=lambda res: res[2], reverse=True)
        avg_prec, rel = 0.0, 0.0
        for i, tri in enumerate(sorted_pred):
            if tri[1] == 1:
                rel += 1.0
                avg_prec += rel / (i + 1)
        avg_prec /= rel
        mean_avg_prec += avg_prec
        relQ += 1.0
    mean_avg_prec /= relQ
    return mean_avg_prec

def calc_mean_reciprocal_rank(preds):
    """
    skip all questions w/o correct answers
    and all questions w/ only correct answers
    """
    mean_reciprocal_rank, relQ = 0.0, 0.0
    for pred in preds.values():
        cnt = 0
        for tri in pred: cnt += tri[1]
        if cnt == 0 or cnt == len(pred): continue
        sorted_pred = sorted(pred, key=lambda res: res[1])
        sorted_pred = sorted(sorted_pred, key=lambda res: res[2], reverse=True)
        reciprocal_rank, rel = 0.0, 0.0
        for i, tri in enumerate(sorted_pred):
            if tri[1] == 1:
                rel += 1.0
                reciprocal_rank += rel / (i + 1)
                break
        relQ += 1.0
        mean_reciprocal_rank += reciprocal_rank
    mean_reciprocal_rank /= relQ
    return mean_reciprocal_rank