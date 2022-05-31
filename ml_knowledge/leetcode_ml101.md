# Machine Learning 101

* Problem Conversion between Classification and Regression
  * if we reformulate the problem as predicting the price range of real estate instead of a single price tag, then we could expect to obtain a more robust model.
  * we can also transform it from classification to regression. Instead of giving a binary value as the output, we could define a model to give a probability value [0, 100%] on whether the photo shows a cat. It helps to compare models.
  * In this scenario, one often applies one of the machine learning models called ==Logistic Regression==, which gives continuous probability values as output, but it is served to solve the classification problem.

* **Data determines the ==upper bound== of performance that the model can achieve**.



## Workflow (summary from EE559)

> Data-Centric Workflow: The workflow to build a machine learning model is centralized around the data.

1. Preprocessing
   1. Feature Analysis
   2. Imputation of Missing Values
   3. Format Conversion (e.g., encoding categorical features)
   4. Data Cleaning
   5. Normalization
2. Feature Engineering
   1. Create/Remove Features
3. Feature Dimensionality Adjustment
4. Hyper-parameter Tuning / Cross Validation
5. Training & Testing



## Bias and Variance

> Another perspective to look at the phenomenon of **underfitting** and **overfitting**.

* **Bias is a learner’s tendency to consistently learn the same wrong thing. Variance is the tendency to learn random things unrelated to the real signal**
* Concept -- Main prediction: Expectation.
* *Related to EE503.*
* For example,  $MSE = (bias)^2 + variance$.
