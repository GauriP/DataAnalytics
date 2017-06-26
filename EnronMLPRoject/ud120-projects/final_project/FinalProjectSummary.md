##### 1. Summarize for us the goal of this project and how machine learning is useful in trying to accomplish it. As part of your answer, give some background on the dataset and how it can be used to answer the project question. Were there any outliers in the data when you got it, and how did you handle those?  [relevant rubric items: “data exploration”, “outlier investigation”]

Enron was an American company started in 1985 . It mainly dealt in Energy and services sector. Enron went bankrupt in 2001, Affecting thousands of people and many countries. Upon investigation it was revealed that the financial condition for the company was sustained by systematic corruption and cooking the accounting books. 

This scandal was architected by a few employees of the company. Going forward they will be referred to as Persons of interest or POI. In this project we will try and use Machine learning knowledge to figure out the POIs just based on their financial data and email meta data. This can help us get an understanding of the usefulness of Machine learning algorithms in finding causes of fraud in future financial scandals. 
In simplistic term ML uses algorithms to find patterns in the data so we can figure out who were responsible for the Enron scandal by looking at their financial data and email meta data. 

Upon inspecting the data visually I found out there was one outlier in the data provided. The data was presented in a dictionary format with the name of the employee being the key and the financial and email information being the value for the dictionary. There was one key which was the sum total of all the data for various employees and it was the obvious outlier. I removed this outlier since we do not need information for all the employees combined, we need information for individual persons. 



##### 2. What features did you end up using in your POI identifier, and what selection process did you use to pick them? Did you have to do any scaling? Why or why not? As part of the assignment, you should attempt to engineer your own feature that does not come ready-made in the dataset -- explain what feature you tried to make, and the rationale behind it. (You do not necessarily have to use it in the final analysis, only engineer and test it.) In your feature selection step, if you used an algorithm like a decision tree, please also give the feature importances of the features that you use, and if you used an automated feature selection function like SelectKBest, please report the feature scores and reasons for your choice of parameter values.  [relevant rubric items: “create new features”, “intelligently select features”, “properly scale features”]

These are the features I used in the POI identifier : ['poi','salary', 'deferral_payments',   'bonus','deferred_income',   'exercised_stock_options', 'other', 'from_poi_to_this_person', 'from_this_person_to_poi', 'shared_receipt_with_poi']. I have also used , 'total_stock_value' & 'total_payments' as a part of creating new feature. I took the ratio of total stock value to total payments made to the employees. I think this would give a good understanding of how much stock the employees earned as compared to the total amount they earned. Stocks are usually looked on as short term income if needed to liquidate fast. Hence my assumption would be the POIs would have a larger stock portfolio in their payments. On comparing the Recall value calculated by using the above mentioned ratio and by not using the ratio I get a better result with using the ratio of the newly generated feature. Hence, I will be leaving the new feature intact in the final algorithm. I ran the min max scaler on the data set to understand of scaling the data make a huge impact on the output. It does, but in a negative way. The recall and precision drop dramatically upon scaling. Also based on the data provided, I would not want to scale the information since we want to know if there were major discrepancies in the amount of money received by the POIs as opposed to rest of the employees.

I used FeatureUnion from the SKlearn library. Feature union creates a chain where in the data is worked on by the consecutive algorithm in a list. I run the data through SelectKbest and PCA algorithms before feeding it to the ML algorithm. Select K best gives the K best variables having the most effect on the output. PCA creates principal components out of the selected variables and proceeds with the best components to the next segment that is the Machine learning algorithm.  



##### 3. What algorithm did you end up using? What other one(s) did you try? How did model performance differ between algorithms?  [relevant rubric item: “pick an algorithm”]
I am using K nearest neighbors after trying out multiple algorithms. The ones that i tried were: GNB, Random Forest, SVC. The KNN algorithm gave the best Recall value out of the many I tried. While the performance was not hugely varied I did not want to implicate someone who was innocent, and just increase the Recall value. KNN gave the best result with a balance of precision and recall. with Recall being 34.4% and precision being 59.87%. 

#####  4. What does it mean to tune the parameters of an algorithm, and what can happen if you don’t do this well?  How did you tune the parameters of your particular algorithm? What parameters did you tune? (Some algorithms do not have parameters that you need to tune -- if this is the case for the one you picked, identify and briefly explain how you would have done it for the model that was not your final choice or a different model that does utilize parameter tuning, e.g. a decision tree classifier).  [relevant rubric items: “discuss parameter tuning”, “tune the algorithm”]

Tuning the parameters gives the best possible result using that particular algorithm with the given data. Without tuning the parameters we might get stuck with the default parameters which won't always be otpimal for the dataset in question. 

For the KNN algorithm  I tuned the following parameters by using the ranges shown below for GridsearchCV. 
{'knn_n_neighbors': range(2,len(features_list)), ## Helps select the best possible values of the number of K neighbours which gives best output value.
"feature_sel_skb_k" : range(2,(len(features_list)-1)), ## This is for the Select k best algorithm where best k variables are selcted. 
'feature_sel_pca_n_components':range(2,(len(features_list)-1))} ## This is for Principal Component Analysis Algorithm where the best components are selected. 


##### 5. What is validation, and what’s a classic mistake you can make if you do it wrong? How did you validate your analysis?  [relevant rubric items: “discuss validation”, “validation strategy”]

Once we train our model, we need to test it on testing data. This is called validation of the ML algorithm. The classic mistake made if we don't use model validation is overfitting. Hypothetically, we can gain really good results using the ML algorithm but, this outcome might not be same out new data. to avoid such unpleasant surprises it makes sense to use a separate testing and training dataset and run the model on testing data. 

I am using stratifiedshufflesplit and grdisearchCV for cross validation. Stratifiedshufflesplit creates a stratified groups of data and each chunk will then be validated using the gridsearchCV function using the combinations of the various tuned parameters. 


##### 6. Give at least 2 evaluation metrics and your average performance for each of them.  Explain an interpretation of your metrics that says something human-understandable about your algorithm’s performance. [relevant rubric item: “usage of evaluation metrics”]

The evaluation metrics I am using are:
Recall: The value I get for recall is 34.4%
Precision:The value I get for precision is 59.87%
F1 score: The value of F1 score is 0.37 . 

The value of recall is 34.4% meaning, 34.4% of the time the algorithm predicted correctly. while 65.6% of time it did not catch the POI. 
The value of precision is 59.87% means those many predictions were made correctly. rest of the times the predictions were wrong implication a non POI. 
F1 is a weighted average of the two. 
While it is important to catch POIs, we do not want to implicate a non POI at the expense of being right. Increasing the recall should not be the only criterion as it can be misguided. For e.g. if we implicate everyone, which will include the actual perpetrators we will get a 100% Recall, but at the same time we will be implicating many innocents. 