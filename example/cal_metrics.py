
import numpy

def dice(result, reference):
    result = numpy.atleast_1d(result.astype(numpy.bool))
    reference = numpy.atleast_1d(reference.astype(numpy.bool))
    intersection = numpy.count_nonzero(result & reference)
    size_i1 = numpy.count_nonzero(result)
    size_i2 = numpy.count_nonzero(reference)
    
    try:
        dc = 2. * intersection / float(size_i1 + size_i2)
    except ZeroDivisionError:
        dc = 1.0
    
    return dc


def sensitivity(result, reference):
    result = numpy.atleast_1d(result.astype(numpy.bool))
    reference = numpy.atleast_1d(reference.astype(numpy.bool))
        
    tp = numpy.count_nonzero(result & reference)
    fn = numpy.count_nonzero(~result & reference)

    try:
        recall = tp / float(tp + fn)
    except ZeroDivisionError:
        recall = 1.0
    
    return recall


def specificity(result, reference):
    result = numpy.atleast_1d(result.astype(numpy.bool))
    reference = numpy.atleast_1d(reference.astype(numpy.bool))
    tn = numpy.count_nonzero(~result & ~reference)
    fp = numpy.count_nonzero(result & ~reference)

    try:
        specificity = tn / float(tn + fp)
    except ZeroDivisionError:
        specificity = 1.0
    
    return specificity


def Get_Metrics(pred, label):
    metric_dict = {}
    pred_WT, label_WT = (pred!=0), (label!=0)
    pred_TC, label_TC = ((pred==1)+(pred==4)), ((label==1)+(label==4))
    pred_ET, label_ET = (pred==4), (label==4)

    metric_dict['WT_Dice'] = dice(pred_WT, label_WT)
    metric_dict['WT_Sensitivity'] = sensitivity(pred_WT, label_WT)
    metric_dict['WT_Specificity'] = specificity(pred_WT, label_WT)

    metric_dict['TC_Dice'] = dice(pred_TC, label_TC)
    metric_dict['TC_Sensitivity'] = sensitivity(pred_TC, label_TC)
    metric_dict['TC_Specificity'] = specificity(pred_TC, label_TC)

    metric_dict['ET_Dice'] = dice(pred_ET, label_ET)
    metric_dict['ET_Sensitivity'] = sensitivity(pred_ET, label_ET)
    metric_dict['ET_Specificity'] = specificity(pred_ET, label_ET)
    return metric_dict

