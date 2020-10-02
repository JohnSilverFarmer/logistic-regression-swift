# Logistic Regression

A tiny implementation of logistic regression using `Swift`. Matrix operations are made fast using `Accelerate`. Can be build by creating a new command line tool in `Xcode` or using the Swift Package Manager. Datasets for testing:

1. `mvn.csv` contains linearly separable data from two Gaussians. 
2. `hours.csv` contains a simple dataset taken from [Wiki](https://en.wikipedia.org/wiki/Logistic_regression#Probability_of_passing_an_exam_versus_hours_of_study) to compare learned coefficients to.
