* **Lab Assignment 3: Scikit-learn: Data Preprocessing and Model Performance Evaluation**
* **Name : Madi Priyanshu**
* **ID: 202618046**


* **Objective :  Train and fit models on different normalization and standardization pipelines and determine overfit and confusion matrix among different trained logistic and decision tree models and determining which model is best, which is overfitting and which is most suitable pipeline** 

* **Final Observations** : 

- Decision Tree + MinMaxScaler produced the best overall test performance, achieving 86.21% testing accuracy and an F1-score of 0.8155.

- Decision Tree models showed strong overfitting, with training accuracy of 99.60% compared with approximately 86.20% testing accuracy, giving a train-test gap of about 13.4 percentage points.

- Logistic Regression showed much less overfitting. The StandardScaler version had a training accuracy of 81.91% and testing accuracy of 81.48%, resulting in only a 0.43 percentage-point difference.

- StandardScaler performed slightly better than MinMaxScaler for Logistic Regression. Its F1-score was 0.7259, compared with 0.7184 for MinMaxScaler.

- Scaling had virtually no effect on the Decision Tree. Both scaling methods produced an F1-score of 0.8155, confirming that Decision Trees are largely insensitive to feature scaling.