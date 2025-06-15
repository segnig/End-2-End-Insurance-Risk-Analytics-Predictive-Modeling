# End-to-End Insurance Risk Analytics & Predictive Modeling

This repository documents a comprehensive workflow for analyzing insurance portfolio data, assessing risk and profitability, and laying the groundwork for predictive modeling. The project demonstrates best practices in data science, including data cleaning, exploratory data analysis (EDA), statistical thinking, and version control.

---

## Project Structure

- **scripts/**: Contains Python scripts for data loading, cleaning, EDA, and utility functions (e.g., `eda_stats.py`).
- **notebooks/**: Jupyter notebooks for step-by-step analysis, visualizations, and reporting.
- **data/**: (Ignored by git) Directory for raw and processed data files.
- **requirements.txt**: List of all Python dependencies required for the project.
- **.gitignore**: Specifies files and folders to be ignored by Git, including large data files and virtual environments.
- **README.md**: Project overview and instructions.

---

## Key Tasks & Methodology

### 1. Data Loading & Initial Assessment
- Developed custom functions to load large, pipe-delimited text files efficiently.
- Inspected and converted data types for all columns, ensuring correct handling of numerical, categorical, and boolean features.
- Assessed memory usage and data structure to optimize performance.

### 2. Data Quality Assessment
- Calculated missing value counts and percentages for all features.
- Visualized missing data patterns using bar plots.
- Interpreted the impact of missingness, especially for features with high proportions of missing values (e.g., `Citizenship`, `CustomValueEstimate`, `Bank`).
- Added `.gitignore` rules to prevent large files and sensitive data from being tracked.

### 3. Exploratory Data Analysis (EDA)
- Generated descriptive statistics for numerical features (mean, std, min, max, quartiles).
- Detected and interpreted class imbalances in categorical features such as `Gender`, `VehicleType`, and `Province`.
- Visualized distributions of both numerical and categorical variables using histograms, bar plots, and box plots.
- Detected outliers in numerical features using the Z-score method, quantified their prevalence, and discussed their potential impact.
- Analyzed temporal trends in claims and premiums to identify seasonality or shifts in risk.

### 4. Risk & Profitability Analysis
- Calculated the overall loss ratio (`TotalClaims / TotalPremium`) for the portfolio.
- Segmented loss ratio analysis by `Province`, `VehicleType`, and `Gender` to identify high-risk and high-profitability segments.
- Interpreted results to inform potential business actions, such as pricing adjustments or targeted risk management.

### 5. Version Control & Collaboration
- Initialized a GitHub repository and created a dedicated branch (`task-1`) for day-one analysis.
- Maintained a clean commit history with descriptive messages.
- Used `.gitignore` to exclude large data files and virtual environments, ensuring a lightweight and shareable repository.

---

## How to Use

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/End-to-End-Insurance-Risk-Analytics-Predictive-Modeling.git
   cd End-to-End-Insurance-Risk-Analytics-Predictive-Modeling
   ```

2. **Set up a virtual environment and install dependencies:**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # On Windows
   pip install -r requirements.txt
   ```

3. **Add your data files to the `data/` directory** (these files are not tracked by git).

4. **Run the Jupyter notebooks or Python scripts** in the `notebooks/` and `scripts/` folders to reproduce the analysis and visualizations.

---

## Key Insights

- **Data Quality:** Several features have high missingness and may require exclusion or special handling. Most features have low missingness and are suitable for analysis after basic cleaning.
- **Class Imbalance:** Features such as `Gender`, `VehicleType`, and `Province` show significant class imbalance, which should be considered in modeling and interpretation.
- **Risk Segmentation:** Loss ratios vary significantly by province, vehicle type, and gender, highlighting areas for targeted business strategies.
- **Outlier Detection:** Outliers are prevalent in several numerical features and can impact statistical analysis and modeling. Their treatment should be carefully considered.

---

## References & Further Reading

- [Pandas Documentation](https://pandas.pydata.org/)
- [Seaborn Documentation](https://seaborn.pydata.org/)
- [GitHub Docs](https://docs.github.com/)
- [Git LFS](https://git-lfs.github.com/)

---

*Project by Segni Girma, 2025*
