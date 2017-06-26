#!/usr/bin/python

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import pickle
import numpy as np
from feature_format import featureFormat
count = 0
enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))
for key, value in enron_data.items():
#	print key
#	print enron_data[key]["total_payments"]
	if enron_data[key]["email_address"] != "NaN":
		count = count+1
#	if key == "FASTOW ANDREW":
#		print "Found james"
#		print enron_data[key]["total_payments"]
#print enron_data['James Prentice']["stock"]

features = list(enron_data.keys())
features = list(enron_data[features[1]].keys())
features.remove("poi")
features.insert(0, "poi")
features = ['poi', 'salary', 'to_messages', 'deferral_payments', 'total_payments', 'exercised_stock_options', 'bonus', 'restricted_stock', 'shared_receipt_with_poi', 'restricted_stock_deferred', 'total_stock_value', 'expenses', 'loan_advances', 'from_messages', 'other', 'from_this_person_to_poi', 'director_fees', 'deferred_income', 'long_term_incentive', 'from_poi_to_this_person']
print features
data_array = featureFormat(enron_data, features,sort_keys = True, remove_NaN = True)
print data_array

print count
	#print(len(enron_data[key]))
