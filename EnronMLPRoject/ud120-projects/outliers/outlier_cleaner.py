#!/usr/bin/python

import math

def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """
    #print "in outlier cleaner"
    cleaned_data = []
    cleaned_data_temp = []
    #print predictions-net_worths
    cleaned_data_temp = tuple(zip(list(ages), list(net_worths), list((predictions-net_worths)**2)))
    cleaned_data = sorted(cleaned_data_temp, key=lambda x:x[-1])
    length = int(math.ceil(len(cleaned_data)*0.95))
    #print length
    cleaned_data = cleaned_data[:length]
    #print math.floor(len(cleaned_data))
    #print len(cleaned_data)
    #length to be kept
    #length = floor(len(cleaned_data) * 0.9)
    #print "the length of the data"
    #print length
    #print cleaned_data
    ### your code goes here

    return cleaned_data

