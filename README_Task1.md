# Credit Scoring Model (German Credit Data)

CodeAlpha Machine Learning Internship, Task 1.

## Objective
Predict whether a loan applicant is a good or bad credit risk using past financial data.

## Dataset
German Credit Data, loaded from OpenML.
* 1,000 loan applicants and 20 features (loan amount, duration, savings, employment, housing, age and others)
* No missing values
* Target: 1 = bad credit risk (300 applicants, 30%), 0 = good credit risk (700 applicants, 70%)

## Method
1. Loaded the data and checked for missing values.
2. Created four new features from the financial history:
   * `credit_per_month`: loan amount divided by duration (monthly repayment burden)
   * `log_credit_amount`: log of the loan amount, to reduce the effect of very large loans
   * `long_loan`: 1 if the duration is longer than 24 months
   * `age_group`: four age bands
3. Split into 80% training and 20% testing (stratified, random_state=42).
4. Scaled numeric features and one hot encoded categorical features.
5. Trained three models: Logistic Regression, Decision Tree (max depth 5) and Random Forest (300 trees). All use balanced class weights because bad borrowers are only 30% of the data.
6. Evaluated with Accuracy, Precision, Recall, F1 Score and ROC AUC. Confirmed the ranking with 5 fold cross validation.

## Results (test set of 200 applicants, 60 of them bad credit)

| Model | Accuracy | Precision | Recall | F1 Score | ROC AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.7550 | 0.5647 | 0.8000 | 0.6621 | 0.8088 |
| Decision Tree | 0.6000 | 0.3913 | 0.6000 | 0.4737 | 0.6342 |
| Random Forest | 0.7600 | 0.6875 | 0.3667 | 0.4783 | 0.8043 |

5 fold cross validation, mean ROC AUC:

| Model | Mean AUC | Std |
|---|---|---|
| Logistic Regression | 0.785 | 0.018 |
| Decision Tree | 0.677 | 0.041 |
| Random Forest | 0.797 | 0.021 |

## Conclusion
Logistic Regression is the final model.

* Accuracy is misleading here. A model that calls every applicant good would already score 70%. Random Forest has the highest accuracy (76%) but catches only 22 of the 60 bad borrowers (recall 0.367).
* Logistic Regression catches 48 of the 60 bad borrowers (recall 0.80). The cost is 37 good applicants wrongly flagged as bad (precision 0.565).
* In credit scoring, approving a borrower who defaults usually costs a bank more than rejecting a good one. That is why Recall is the main metric here.
* In cross validation, Logistic Regression (0.785) and Random Forest (0.797) are very close, and the gap is smaller than the spread between folds. Their ranking ability is about the same. Logistic Regression was chosen because of its higher recall at the default threshold and because it is simpler to explain.
* Random Forest could probably reach a higher recall if its decision threshold were lowered. This was not tested for that model.
* The Decision Tree performed worst on every measure.

## Threshold test (Logistic Regression)
A lower threshold flags more applicants as bad.

| Threshold | Precision | Recall | F1 Score | Bad borrowers caught (of 60) | Good applicants wrongly flagged (of 140) |
|---|---|---|---|---|---|
| 0.5 | 0.5647 | 0.8000 | 0.6621 | 48 | 37 |
| 0.4 | 0.4762 | 0.8333 | 0.6061 | 50 | 55 |
| 0.3 | 0.4262 | 0.8667 | 0.5714 | 52 | 70 |

Going from 0.5 to 0.3 catches 4 more bad borrowers but wrongly flags 33 more good applicants. The gain is small compared to the cost, so the default threshold of 0.5 was kept. A lower threshold only makes sense if one missed default costs the bank more than about 8 rejected good customers.

## Limitations
* The dataset is small (1,000 rows) and the test set has only 200 applicants, so small differences between models are not strong evidence.
* The data is old and comes from one source. This is a learning project and not a real lending tool.
* Features such as age, personal status and foreign worker status can lead to unfair decisions. A real credit model would need a fairness review before use.

## How to run
1. Open the notebook file in Google Colab. An internet connection is needed to download the data.
2. Click Runtime, then Run all.

## Tools
Python, Pandas, NumPy, Scikit-learn, Matplotlib
