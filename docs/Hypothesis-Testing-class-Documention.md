#  `hypothesis_testing.py` Documentation

## Module Overview

This module provides a comprehensive framework for conducting **A/B testing** and **hypothesis testing** on insurance-related datasets. It enables statistical analysis of risk factors—such as claim frequency, claim severity, and profit margin—across customer segments defined by features like **Province**, **Postal Code**, and **Gender**.

It includes:

* Data loading and preprocessing
* Balanced group validation
* Statistical testing (Chi-square, t-tests)
* Risk reporting with actionable insights

---

## Class: `InsuranceRiskAnalyzer`

### Purpose:

Encapsulates the logic for analyzing insurance data and performing hypothesis testing between customer groups, aiming to detect statistically significant differences in insurance risk metrics.

### Constructor

```python
InsuranceRiskAnalyzer(file_path: str = "../data/data.csv", date_col: str = "TransactionMonth")
```

#### Parameters:

* `file_path` *(str)*: Path to the CSV file containing the insurance dataset.
* `date_col` *(str)*: Name of the date column in the dataset (converted to datetime).

---

## Methods

### 1. `_load_and_preprocess_data() -> pd.DataFrame`

Loads and prepares the dataset by:

* Parsing dates
* Creating derived columns: `has_claim`, `margin`, `claim_frequency`
* Cleaning categorical variables

### 2. `_check_group_balance(group_a, group_b, balance_cols) -> bool`

Validates whether the control and test groups are statistically balanced on specified covariates using:

* Chi-square test for categorical
* t-test for numerical

### 3. `run_ab_test(feature_col, group_a_value, group_b_value=None, balance_cols=None) -> Dict`

Performs hypothesis testing between two groups on:

* Claim Frequency (Chi-square)
* Claim Severity (t-test on claims only)
* Margin (t-test)

#### Parameters:

* `feature_col`: Column name for grouping.
* `group_a_value`: Value defining group A.
* `group_b_value`: Value defining group B (optional).
* `balance_cols`: Columns to verify statistical balance.

#### Returns:

A dictionary with test results including statistics, p-values, group means/rates, and conclusions.

---

## Built-in Test Routines

### 4. `test_province_risk() -> Dict`

Runs A/B tests for each **Province** vs the rest. Uses `'Gender'` and `'VehicleType'` as balance checks.

### 5. `test_zip_code_risk(top_n: int = 5) -> Dict`

Compares the top `n` highest and lowest risk zip codes based on **claim frequency**. Ensures samples meet minimum policy threshold.

### 6. `test_gender_risk() -> Dict`

Compares insurance risk between `'Male'` and `'Female'` genders, using `'Province'` and `'VehicleType'` for balance checking.

---

## Report Generation

### 7. `generate_test_report(test_results: Dict) -> str`

Converts the results of a hypothesis test into a human-readable report that includes:

* Sample sizes
* Test statistics
* Significance decisions
* Recommendations for insurance strategy

---

## Master Analysis

### 8. `run_full_analysis() -> Dict`

Executes all major tests:

* Province-level
* Zip code-level
* Gender-based risk comparison

Handles errors gracefully and compiles:

* Raw results
* Text reports per segment

---

## Metrics Analyzed

| Metric            | Test Used          | Description                              |
| ----------------- | ------------------ | ---------------------------------------- |
| `claim_frequency` | Chi-square         | % of customers who filed a claim         |
| `claim_severity`  | Independent t-test | Avg. amount claimed (only for claimants) |
| `margin`          | Independent t-test | Profit per customer = Premium - Claims   |

---

## ⚠️ Configuration Parameters

These thresholds are set in the constructor:

| Attribute               | Value | Purpose                                  |
| ----------------------- | ----- | ---------------------------------------- |
| `alpha`                 | 0.05  | Significance level for statistical tests |
| `min_sample_size`       | 30    | Warn if sample size is smaller           |
| `min_policies_for_test` | 10    | Skip test if fewer policies available    |

---

## 🧪 Example Usage

```python
analyzer = InsuranceRiskAnalyzer(file_path="insurance_data.csv")

# A/B test for claim severity and margin between two vehicle types
ab_results = analyzer.run_ab_test(
    feature_col="VehicleType",
    group_a_value="SUV",
    group_b_value="Sedan",
    balance_cols=["Gender", "Province"]
)

print(analyzer.generate_test_report(ab_results))

# Run full analysis across key demographics and regions
full_output = analyzer.run_full_analysis()

# Print one of the reports
print(full_output['reports']['gender_risk'])
```

---

## 🛠 Dependencies

* `pandas`, `numpy` – data processing
* `scipy.stats` – statistical tests (`ttest_ind`, `chi2_contingency`)
* `datetime` – date parsing
* `warnings` – for runtime alerts

---

## Strengths

* **Modular**: Easily extendable to other grouping features.
* **Balanced Group Verification**: Reduces bias.
* **Practical Metrics**: Relevant to insurance profitability.
* **Robust Reporting**: Actionable insights and safety warnings.