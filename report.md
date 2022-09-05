# Report 

## Overview of the Analysis

The purpose of the analysis was to correctly identify whether or not credit loans were labeled as healthy or high-risk. 

Input data used for the analysis is located in the `Resources` folder in a csv filled with both the feature set (X) and the labels (y). The feature set includes basic information about the loan (loan_size, interest_rate, borrower_income, debt_to_income, num_of_accounts, derogatory_marks and total_debt). The labels are a set of 1's and 0's in the loan_status column of the csv which indicate whether or not the loan was labeled as healthy or high-risk, respectively. 

After loading in the data to the model and splitting it into a training and testing set, a Logistic Regression model was used to model the data, with and without oversampling. In this assignment, directions were not given to apply standardscaler() to the dataset prior to modeling, so that's taken as a potential future next step to achieve even more desireable results from the model. 

## Results

* Machine Learning Model 1 (no oversampling method applied):
  * The model was made purely with the `LogisticRegression()` function from sklearn with `random_state = 1`. 
  * Summary of model performance statistics:  
    * Balanced accuracy score: 0.95 
    * Precision: 0.85 (label 1's) - 1.0 (label 0's)
    * Recall: 0.91 (label 1's) - 0.99 (label 0's)

* Machine Learning Model 2 (oversampling method applied):
  * After applying imblearn's oversampling function to the dataset, the model was then made with the `LogisticRegression()` function from sklearn with `random_state = 1`. 
  * Summary of model performance statistics: 
    * Balanced accuracy score: 0.99
    * Precision: 0.84 (label 1's) - 1.0 (label 0's)
    * Recall: 0.99 (label 1's) - 0.99 (label 0's) 

## Summary

Although both models fit really well on a statistical basis (as accuracy scores are between 0.95 and 0.99, etc), Model 2 (where oversampling was included prior to model fitting) has the best performance metrics.

Choosing whether or not this model should be used in the future depends on the end goal of the project. In my eyes, I think that more work should be done on the model as I think that the high-risk loan cases could be predicted with greater precision, but at the same time Model 2 could be used going forward until a better model is discovered as long as the lower precision value for the high-risk loans is acceptable for the model. 