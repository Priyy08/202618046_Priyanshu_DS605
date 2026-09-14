# NYC Airbnb Price Prediction: End-to-End Machine Learning System Report

## 1. Executive Summary

This project establishes an end-to-end machine learning pipeline to predict nightly listing prices for Airbnb properties across New York City using the **New York City Airbnb Open Data (2019)** dataset. 

Beginning with exploratory data analysis and rigorous preprocessing across **48,895 raw records**, multiple regression architectures (**Linear Regression**, **Gradient Boosting**, **Random Forest**, and **Histogram-based Gradient Boosting**) were systematically evaluated. While standard tree ensembles such as Random Forest achieved strong raw fit, they exhibited pronounced overfitting on the training set (Train $R^2 \approx 0.88$ vs. Test $R^2 \approx 0.65$, exceeding the $0.15$ overfitting tolerance threshold). 

In contrast, **Histogram-based Gradient Boosting (`HistGradientBoostingRegressor`)** emerged as the superior architecture. A dedicated, optimized pipeline was constructed with hyperparameter tuning via 3-fold cross-validation. The final tuned HistGradient Boosting model achieved an **out-of-sample Test $R^2$ of 0.6518 (log-space)**, a **Mean Absolute Error (MAE) of $45.38**, a **Root Mean Squared Error (RMSE) of $88.03**, and maintained an $R^2$ generalization gap of only **0.0346** (indicating robust generalization with no significant overfitting). The complete pipeline was serialized and validated for single- and batch-inference serving.

---

## 2. Exploratory Data Analysis & Data Preprocessing

### 2.1. Initial Dataset Hygiene & Anomaly Detection
- **Original Dimensions**: 48,895 listings and 16 features.
- **Duplicate Records**: 0 duplicate rows; 0 duplicate listing IDs.
- **Missing Value Assessment**:
  - `last_review`: 10,052 missing values (20.56%)
  - `reviews_per_month`: 10,052 missing values (20.56%)
  - `host_name`: 21 missing values (0.04%)
  - `name`: 16 missing values (0.03%)
  - All critical identifiers, spatial coordinates, room types, and price attributes had 0 missing values.
- **Invalid Values**: 11 records had `price <= 0` (dropped from dataset). No negative `minimum_nights` or invalid availability records were present. Dimensions after zero-price cleaning: **48,884 listings**.

### 2.2. Distributional Characteristics & Target Transformation
- **Target Distribution (`price`)**: Severely right-skewed with a long tail:
  - Minimum: \$10.00 | 25th Percentile: \$69.00 | Median: \$106.00 | Mean: \$152.72
  - 75th Percentile: \$175.00 | 95th Percentile: \$355.00 | 99th Percentile: \$799.00
  - 99.5th Percentile: \$1,000.00 | Maximum: \$10,000.00
- **Outlier Filtering**: Listings above the 99.5th percentile (\$1,000) were removed (239 extreme outlier records removed), producing a modeling dataset of **48,645 listings**.
- **Logarithmic Target Transformation**: Given the heavy price skewness, the target was transformed using $\log(1 + \text{price})$ (`np.log1p`), successfully stabilizing variance and converting the distribution into a near-normal bell curve.
- **Predictor Capping**: `minimum_nights` possessed extreme outliers up to 1,250 nights; capped at its 99th percentile of **40 nights** (`minimum_nights_capped`).

### 2.3. Key Domain Insights
- **Geographic Impact**:
  - **Manhattan** command the highest prices (Mean: \$196.88, Median: \$150.00; 21,660 listings).
  - **Brooklyn** follows (Mean: \$124.44, Median: \$90.00; 20,095 listings).
  - **Staten Island** (Mean: \$114.81, Median: \$75.00; 373 listings).
  - **Queens** (Mean: \$99.52, Median: \$75.00; 5,666 listings).
  - **Bronx** offers the lowest entry price (Mean: \$87.58, Median: \$65.00; 1,090 listings).
- **Listing Typology**:
  - **Entire home/apt**: Highest rate (Mean: \$211.81, Median: \$160.00; 25,407 listings).
  - **Private room**: Moderate rate (Mean: \$89.81, Median: \$70.00; 22,319 listings).
  - **Shared room**: Budget tier (Mean: \$70.25, Median: \$45.00; 1,158 listings).

### 2.4. Feature Engineering
Six engineered features were added to enhance spatial and temporal signals:
1. `last_review_year`: Extracted year component from review timestamp.
2. `last_review_month`: Extracted seasonal month component.
3. `reviews_per_month`: Missing values imputed with `0.0` (signifying listings with 0 review activity).
4. `has_review`: Binary flag ($1$ if `number_of_reviews > 0`, else $0$).
5. `has_last_review`: Binary indicator for timestamp existence.
6. `distance_from_nyc_center`: Euclidean distance from central NYC coordinate ($40.7128^\circ \text{N}, -74.0060^\circ \text{W}$):
   $$\text{distance} = \sqrt{(\text{latitude} - 40.7128)^2 + (\text{longitude} - (-74.0060))^2}$$

### 2.5. Data Splitting & Preprocessing Architecture
- **Train/Test Partition**: 80/20 train/test split with `random_state=42`:
  - **Training set**: 38,916 samples (15 features)
  - **Testing set**: 9,729 samples (15 features)
- **Scikit-Learn `ColumnTransformer`**:
  - **Numerical Features (12 features)**: `latitude`, `longitude`, `minimum_nights_capped`, `number_of_reviews`, `reviews_per_month`, `calculated_host_listings_count`, `availability_365`, `last_review_year`, `last_review_month`, `has_review`, `has_last_review`, `distance_from_nyc_center`. Processed with `SimpleImputer(strategy="median")` followed by `StandardScaler()`.
  - **Categorical Features (3 features)**: `neighbourhood_group`, `neighbourhood`, `room_type`. Processed with `SimpleImputer(strategy="most_frequent")` followed by `OneHotEncoder(handle_unknown="ignore", sparse_output=False)`.

---

## 3. Initial Model Comparison & Benchmark

Four candidate models wrapped in end-to-end preprocessing pipelines were benchmarked under identical splits on the log-transformed price target:

| Model Architecture | Train MAE | Test MAE | Train RMSE | Test RMSE | Train $R^2$ | Test $R^2$ | $R^2$ Gap (Overfit) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **HistGradient Boosting** | 0.2896 | 0.2984 | 0.3883 | **0.3985** | 0.6543 | **0.6436** | **0.0107** |
| **Random Forest** | **0.1099** | **0.2949** | **0.1507** | 0.3986 | **0.9480** | 0.6434 | *0.3045 (Severe)* |
| **Gradient Boosting** | 0.3071 | 0.3104 | 0.4116 | 0.4137 | 0.6117 | 0.6158 | -0.0041 |
| **Linear Regression** | 0.3165 | 0.3185 | 0.4230 | 0.4244 | 0.5897 | 0.5958 | -0.0060 |

### Observations from Benchmark:
1. **HistGradient Boosting** achieved the highest test score ($R^2 = 0.6436$) and lowest test error ($\text{RMSE} = 0.3985$) with virtually zero overfit gap ($0.0107$).
2. **Linear Regression** underperformed ($R^2 \approx 0.596$), indicating non-linear relationships between geolocation, neighborhood categories, and listing prices.
3. **Baseline Random Forest** fit the training data excessively ($R^2 = 0.9480$), but degraded by over $0.30$ on the test set ($R^2 = 0.6434$).
4. Converting baseline HistGradient predictions back to dollars ($\text{expm1}$) yielded:
   - **Baseline Test MAE**: **\$45.96**
   - **Baseline Test RMSE**: **\$88.93**
   - **Baseline Dollar $R^2$**: **0.4332**

---

## 4. Hyperparameter Tuning & Overfitting Analysis: Random Forest vs. HistGradient

Following baseline evaluations, hyperparameter tuning was conducted to inspect whether Random Forest could be regularized, followed by a dedicated standalone pipeline for HistGradient Boosting.

### 4.1. Random Forest Tuning & Overfitting Diagnosis
- **Search Strategy**: `RandomizedSearchCV` across 20 iterations, 3-fold CV (60 total fits).
- **Best Hyperparameters**:
  - `n_estimators`: 200
  - `max_depth`: 20
  - `max_features`: 0.7
  - `min_samples_leaf`: 2
  - `min_samples_split`: 2
- **Best CV RMSE**: 0.4042
- **Tuned RF Metrics (Log-Space)**:
  - Train $R^2$: **0.8789** | Test $R^2$: **0.6521**
  - Train RMSE: **0.2298** | Test RMSE: **0.3937**
  - **$R^2$ Gap**: **0.2268**
- **Automated Diagnostic**: 
  > `⚠️ Possible overfitting detected. The training R² is considerably higher than the testing R² (Gap = 0.2268 > 0.1500).`
- **Tuned RF Metrics (Dollar-Scale)**:
  - MAE: **\$45.02** | RMSE: **\$87.28** | $R^2$: **0.4541**

### 4.2. Dedicated HistGradient Boosting Pipeline & Tuning
Because HistGradient was identified as the intrinsically superior architecture, an independent pipeline was developed and subjected to randomized cross-validation:
- **Search Space**:
  - `max_iter`: `[100, 200, 300, 500]`
  - `learning_rate`: `[0.01, 0.03, 0.05, 0.10]`
  - `max_leaf_nodes`: `[15, 31, 63]`
  - `max_depth`: `[None, 5, 10, 15]`
  - `min_samples_leaf`: `[10, 20, 30, 50]`
  - `l2_regularization`: `[0, 0.1, 1, 10]`
- **Search Execution**: 25 candidates $\times$ 3 folds = 75 total fits.
- **Optimal Hyperparameters Selected**:
  ```python
  {
      'model__max_iter': 200,
      'model__learning_rate': 0.05,
      'model__max_leaf_nodes': 63,
      'model__max_depth': 15,
      'model__min_samples_leaf': 20,
      'model__l2_regularization': 10
  }