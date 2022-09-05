# Time Series Modeling with Fbprophet

This Jupyter Notebook uses Logistic Regression methods & random oversampling to predict the creditworthiness of loan borrowers. 

## Technologies

This program is written in Python (3.7.13) and developed using Jupyter Notebooks on a Windows computer. Additional libraries that are used in this application are pandas (1.3.5), sklearn (1.02), numpy (1.21.5), nand imblearn (0.7.0) (see parenthesis for versions used in program development).

## Installation Guide

Downloading the code & associated files using `git clone` from the repository is sufficient to download the Jupyter Notebook, ensure that the associated libaries (see Technologies section) are installed on your machine as well. If there are any issues with the library functions please refer to the versions used for app development (see Technnologies section for this information as well).  Please note that this is a Jupyter notebook. 

## Usage

This notebook is referencing data (lending_data.csv) stored in the Resources folder in the repository. This data is imported into the notebook simply by using `pd.read_csv`.

## Code Examples - Logistic Regression Model

If your dataset is in a dataframe, you can easily get a logistic regression model up and running in no time. In the code examples below and in the notebook, the variable `X` is a Pandas dataframe that only contains (numerical) feature data, and the variable `y` is a Pandas series that contains the true y values (the data we're trying to get our model to predict). 

Before fitting the model, we specify the datasets to be used for the training and testing data by using sklearn's `train_test_split` function, like in the following code: 

`python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
`

X_train and y_train are then used to train the model: 

`python
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(random_state = 1)
model.fit(X_train, y_train)
`

Making predictions with the model can easily be done as well:

`python
y_pred = model.predict(X_test)
`

## Code Examples - Evaluating Model Performance

To evaluate model performance, a variety of metrics can be used. Accuracy scores, confusion matrices, and classification reports are used in this notebook. 

Importing these functions into the notebook is done with a few lines of code: 

`python
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import confusion_matrix
from imblearn.metrics import classification_report_imbalanced
`
Call the functions and specify the data that is being predicted with your model (`y_test`) and the true values (`y_test`).

`python
print(balanced_accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report_imbalanced(y_test, y_pred))
`

## Code Examples - Random Oversampling

The credit loan classification problem yields a dataset that's imbalanced, as there are far more loans considered "healthy" than "risky" (at least in this dataset!). 

In order to get a better fit on the data that's under-represented in the dataset, you can use a strategy called "oversampling," which you can implement in a few lines of code. The function used in the notebook, `RandomOverSampler()` from imblearn, makes it such that "the majority class does not take over the other classes during the training process." You can read more about it in the [documentation](https://imbalanced-learn.org/stable/over_sampling.html#random-over-sampler).

After calling `train_test_split()`, call `RandomOverSampler()`, and then feed the results of that (which are `X_resampled` and `y_resampled in this example`) into your `model.fit()` function instead of the original `X_train` and `y_train` variables. 

`python
from imblearn.over_sampling import RandomOverSampler
random_oversampler = RandomOverSampler(random_state=1)
X_resampled, y_resampled = random_oversampler.fit_resample(X_train, y_train)
`
## Model Performance

Check out `report.md` for a summary of model results!

## References

For more information about the libraries, check out the documentation!

[Sklearn Documentation](https://scikit-learn.org/stable/index.html)

[Pandas Documentation](https://pandas.pydata.org/docs/)

[Numpy Documentation](https://numpy.org/doc/)

[Imblearn Documentation](https://imbalanced-learn.org/stable/)

## Contributors

Project contributors are the Rice FinTech bootcamp program team (instructor Eric Cadena) who developed the tasks for this project along with myself (Paula K) who's written the code in the workbook.
