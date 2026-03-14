# End-to-End Insurance Risk Analytics & Predictive Modeling

## Project Title
**Predictive Analytics and Statistical Modeling for Insurance Risk Assessment and Profitability**

## Overview
This repository presents a comprehensive, end-to-end framework for analyzing insurance portfolio data, quantifying risk, evaluating profitability, and building predictive models. The project bridges the gap between raw insurance claim data and actionable business intelligence by employing rigorous statistical methodologies, exploratory data analysis (EDA), and machine learning regression models. This repository follows best practices in software engineering, version control, and data science workflows.

## Objectives
The primary objectives of this project are:
1. **Data Quality and Preparation:** To programmatically load, clean, and impute missing values in large-scale insurance datasets.
2. **Exploratory Data Analysis (EDA):** To uncover temporal, geographical, and demographic trends in insurance premiums and claims.
3. **Statistical Risk Profiling:** To conduct hypothesis testing (A/B testing) to determine if significant risk differences exist across various customer segments (e.g., provinces, gender, vehicle types).
4. **Predictive Modeling:** To develop and evaluate machine learning models capable of predicting future claims and profitability metrics based on historical data.

## System Architecture
The system is built as a modular Python application consisting of data ingestion, statistical testing scripts, and Jupyter notebooks for interactive analysis and modeling. The project uses standard data science libraries and integrates with version control systems to ensure reproducibility.

**Core Technologies and Frameworks:**
* **Programming Language:** Python 3
* **Data Manipulation & Analysis:** Pandas, NumPy
* **Statistical Analysis:** SciPy (`scipy.stats`)
* **Machine Learning:** Scikit-Learn (`sklearn`), XGBoost
* **Data Visualization:** Matplotlib, Seaborn
* **Development Environment:** Jupyter Notebook, JupyterLab
* **Version Control:** Git, DVC (Data Version Control)

## Project Structure
The repository is structured to separate source code, notebooks, data, and documentation cleanly:

```text
├── data/               # Raw and processed datasets (ignored by git)
├── docs/               # Project documentation and module details
│   └── Hypothesis-Testing-class-Documention.md
├── models/             # Saved machine learning models
├── notebooks/          # Jupyter notebooks for EDA and Model Training
│   ├── task-1.ipynb    # Data Loading, Cleaning, and EDA
│   ├── task-3.ipynb    # Statistical Hypothesis Testing
│   └── task-4.ipynb    # Predictive Modeling
├── scripts/            # Core Python modules
│   ├── eda_stats.py          # Data quality, imputation, and EDA utilities
│   └── hypothesis_testing.py # A/B testing and statistical analysis class
├── test/               # Unit testing directory
├── .dvc/               # DVC configuration
├── .gitignore          # Git ignore rules
├── requirements.txt    # Python dependency list
└── README.md           # Project Overview
```

## Methodology
The analytical workflow consists of four main phases:

1. **Data Preprocessing & Quality Assessment:** 
   Custom loaders parse pipe-delimited text files. Missing values are imputed using mode (categorical) and median/mean strategies (numerical). Outliers are detected via Z-scores.
   
2. **Exploratory Data Analysis (EDA, `task-1.ipynb`):**
   Univariate and bivariate analyses define the distribution of policies. Key metrics such as **Loss Ratio** are analyzed using heatmaps, bar plots, and bubble charts. Discoveries include an overall portfolio loss ratio of 1.0477, with distinct geographical variations (e.g., Gauteng at 1.22 vs. Northern Cape at 0.28). Data quality steps included outlier detection (e.g., Z-score anomalies in `mmcode`) and handling high missingness in fields like `CrossBorder`.
   
3. **Statistical Hypothesis Testing (`task-3.ipynb`):**
   An object-oriented approach (`InsuranceRiskAnalyzer`) iteratively tests risk factors across subsets of the data (e.g., Province A vs. All Other Provinces, Men vs. Women). It utilizes `_check_group_balance` to ensure control and test groups are statistically balanced on covariates. This proves risk segmentations mathematically across built-in tests like `test_province_risk`, `test_zip_code_risk`, and `test_gender_risk`.
   
4. **Predictive Modeling (`tas-4-part.ipynb`):**
   Tree-based ensemble models and linear regression algorithms are trained on the processed features to predict `CalculatedPremiumPerTerm`. Crucially, leaky features that directly or indirectly contain target information (such as `TotalPremium`, `PremiumPerTerm`, and `ExcessSelected`) were identified and dropped to ensure robust real-world generalization. The models are evaluated using standard regression metrics.

## Algorithms and Mathematical Formulas
The project utilizes rigorous statistical equations and machine learning algorithms to assess risk.

### 1. Loss Ratio and Profit Margin
The core business metrics evaluated are the **Loss Ratio** and **Margin**:
$$ \text{Loss Ratio} = \frac{\sum \text{Total Claims}}{\sum \text{Total Premium}} $$
$$ \text{Margin} = \text{Total Premium} - \text{Total Claims} $$

### 2. Chi-Square Test for Independence (Claim Frequency)
To evaluate categorical risk differences (e.g., does claim frequency differ significantly between high-risk and low-risk zip codes?), the Pearson's Chi-squared test is used:
$$ \chi^2 = \sum \frac{(O_i - E_i)^2}{E_i} $$
*Where $O_i$ is the observed claim frequency and $E_i$ is the expected claim frequency.*

### 3. Independent Two-Sample T-Test (Claim Severity and Margin)
To compare continuous risk metrics (claim severity and profit margin) between two groups (e.g., Male vs. Female), the independent t-test (Welch's t-test for unequal variances) is applied:
$$ t = \frac{\bar{X}_1 - \bar{X}_2}{\sqrt{\frac{s_1^2}{n_1} + \frac{s_2^2}{n_2}}} $$
*Where $\bar{X}$ is the sample mean, $s^2$ is the sample variance, and $n$ is the sample size of groups 1 and 2.*

### 4. Machine Learning Algorithms
The predictive modeling pipeline evaluates several regression algorithms to approximate the function $f(X) \approx y$ (where $y$ is the predicted premium `CalculatedPremiumPerTerm`):
- **Multiple Linear Regression:** Establishes a baseline linear relationship.
- **Random Forest Regressor:** An ensemble learning method using bagging to reduce variance.
- **Gradient Boosting & XGBoost:** Implementations of gradient boosted decision trees that sequentially minimize the loss function.

## Implementation Details
The codebase is heavily modularized within the `scripts/` directory:
- **`eda_stats.py`**: Functions for data loading, type detection, automated mean/mode imputation, Z-score outlier detection, and generating complex Seaborn/Matplotlib visualizations.
- **`hypothesis_testing.py`**: A robust `InsuranceRiskAnalyzer` class (fully documented in `docs/Hypothesis-Testing-class-Documention.md`). It checks for control/test group balance using `_check_group_balance`, enforces minimum sample size validation (to prevent $p$-value hacking on small subsets), and dynamically routes metrics to either Chi-square or T-tests based on data types.

## Installation
To reproduce the environment locally:

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/End-to-End-Insurance-Risk-Analytics-Predictive-Modeling.git
   cd End-to-End-Insurance-Risk-Analytics-Predictive-Modeling
   ```
2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. Place the raw data files in the `data/` directory (ignored by git due to size).
2. Execute the interactive analysis by launching Jupyter Lab:
   ```bash
   jupyter lab
   ```
3. Navigate to `notebooks/` and run `task-1.ipynb` through `task-4.ipynb` sequentially.
4. Review the generated statistical reports in `insurance_risk_analysis_report.txt`.

## Example Results
* **A/B Testing Insights (`insurance_risk_analysis_report.txt`):** 
  * **Geography**: Gauteng showed a highly significant difference in **claim frequency** ($p < 0.001$) and **profit margin** ($p < 0.001$) compared to other provinces. Kwazulu-Natal exhibited significant differences in **claim severity**.
  * **Zip Codes**: High-risk versus low-risk zip codes demonstrated overwhelming statistical variance in both claim frequency and profit margin.
  * **Gender**: No statistically significant difference in claim frequency, severity, or margin was observed between Male and Female policyholders.
* **Modeling (`tas-4-part.ipynb`):** After rigorous feature engineering (datetime extraction, target encoding) and addressing data leakage, the **Random Forest Regressor** emerged as the best performing model based on RMSE and $R^2$ scores, edging out Gradient Boosting, XGBoost, and Linear Regression. The top 20 feature importances were extracted to provide interpretability to the ensemble.

## Applications
The methodologies presented in this repository have far-reaching applications in actuarial science and insurtech:
* **Dynamic Pricing:** Adjusting base premiums dynamically based on geographic and demographic risk calculations.
* **Targeted Marketing:** Allocating marketing spend toward demographic subsets historically proven to return higher profit margins.
* **Claim Triage:** Identifying high-risk policies early in the underwriting process.

## References and Learning Resources
To understand the underlying techniques utilized in this repository, refer to the following academic and technical resources:

1. **Statistical Hypothesis Testing (T-Tests & Chi-Square):**
   * [Penn State Eberly College of Science: Introduction to Hypothesis Testing](https://online.stat.psu.edu/stat200/lesson/6)
   * [SciPy Documentation: `scipy.stats`](https://docs.scipy.org/doc/scipy/reference/stats.html)
2. **Predictive Modeling & Ensemble Methods:**
   * [Scikit-Learn Algorithms Documentation](https://scikit-learn.org/stable/supervised_learning.html)
   * Chen, T., & Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System*. Proceedings of the 22nd ACM SIGKDD International Conference. [Read the Paper](https://arxiv.org/abs/1603.02754)
3. **Data Analysis and Manipulation:**
   * [Pandas Official Documentation](https://pandas.pydata.org/docs/)

## Limitations and Future Improvements
* **Class Imbalance:** Features such as `Gender` and `VehicleType` exhibit high class imbalance which can bias linear models. Future iterations will incorporate SMOTE or weighted loss functions.
* **Feature Engineering:** Implementing deep feature crosses (e.g., `Province * VehicleType`) could uncover hidden interaction effects to feed into the XGBoost algorithm.
* **Deep Learning Integration:** Applying Neural Networks for tabular data (e.g., TabNet) may extract deeper representations of policyholder risk.

## Contributing
Academic and professional contributions are welcome. Please ensure that all pull requests are accompanied by corresponding updates to the markdown documentation and Jupyter notebooks.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.
