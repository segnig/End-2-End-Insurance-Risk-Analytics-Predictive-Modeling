# InsuranceRiskAnalyzer Class Documentation

## 1. Introduction
The `InsuranceRiskAnalyzer` class provides a statistical framework for analyzing insurance risk patterns across multiple dimensions including geography, demographics, and product types.

## 2. Class Definition
```python
class InsuranceRiskAnalyzer:
    def __init__(self, file_path: str = "../data/data.csv", date_col: str = "TransactionMonth"):
        """
        Initialize the risk analyzer with data source information.
        
        Parameters:
            file_path (str): Path to CSV file containing policy data
            date_col (str): Name of datetime column for temporal filtering
        """
```

## 3. Data Requirements

### 3.1 Required Columns
| Column Name      | Data Type | Description                          |
|------------------|-----------|--------------------------------------|
| PolicyID         | Any       | Unique policy identifier             |
| TotalPremium     | Numeric   | Total premium amount                 |
| TotalClaims      | Numeric   | Total claims amount                  |

### 3.2 Optional Columns
| Column Name      | Analysis Dimension |
|------------------|--------------------|
| Province         | Geographic         |
| PostalCode       | Geographic         |
| Gender           | Demographic        |
| VehicleType      | Product            |

## 4. Core Methods

### 4.1 Data Loading
```python
def _load_and_preprocess_data(self) -> pd.DataFrame:
    """
    Load and preprocess insurance data.
    Returns:
        pd.DataFrame: Processed policy data
    """
```

### 4.2 Sampling Methods
```python
def get_sample(self, method: str = 'auto', **kwargs) -> pd.DataFrame:
    """
    Get data sample using specified method.
    
    Parameters:
        method (str): 'random', 'stratified', 'systematic', 'cluster', or 'auto'
        **kwargs: Method-specific parameters
    
    Returns:
        pd.DataFrame: Sampled data
    """
```

### 4.3 Risk Analysis
```python
def run_risk_analysis(self, sampling_method: str = None, 
                     recent_months: int = None) -> Dict:
    """
    Execute complete risk analysis.
    
    Parameters:
        sampling_method (str): Sampling technique
        recent_months (int): Months of history to include
    
    Returns:
        Dict: Analysis results with structure:
        {
            'geographic_risk': {...},
            'demographic_risk': {...},
            'product_risk': {...},
            'metadata': {...}
        }
    """
```

### 4.4 Reporting
```python
def generate_report(self, results: Dict) -> str:
    """
    Generate formatted business report.
    
    Parameters:
        results (Dict): Analysis results from run_risk_analysis()
    
    Returns:
        str: Formatted report text
    """
```

## 5. Statistical Tests

### 5.1 Test Specifications
| Test Type        | Application               | Metric Tested     | Significance Level |
|------------------|---------------------------|-------------------|--------------------|
| Chi-square       | Claim frequency           | Categorical       | α = 0.05           |
| T-test           | Claim severity (2 groups) | Continuous        | α = 0.05           |
| ANOVA            | Claim severity (>2 groups)| Continuous        | α = 0.05           |

### 5.2 Minimum Sample Sizes
- Per group: ≥30 policies
- For zip code analysis: ≥5 valid postal codes

## 6. Usage Example

### 6.1 Basic Implementation
```python
# Initialize analyzer
analyzer = InsuranceRiskAnalyzer("policy_data.csv")

# Run analysis with automatic sampling on recent year
results = analyzer.run_risk_analysis(
    sampling_method='auto',
    recent_months=12
)

# Generate report
report = analyzer.generate_report(results)
```

### 6.2 Advanced Usage
```python
# Custom stratified sampling
sample = analyzer.get_sample(
    method='stratified',
    stratify_col='Province',
    sample_frac=0.2
)

# Targeted vehicle type analysis
vehicle_results = analyzer._test_vehicle_type_risk()
```

## 7. Output Interpretation

### 7.1 Report Sections
1. **Header**: Analysis metadata
2. **Geographic Findings**: Province and postal code results
3. **Demographic Findings**: Gender-based analysis
4. **Product Findings**: Vehicle type analysis
5. **Recommendations**: Actionable business insights

### 7.2 Key Metrics
- **Claim Frequency**: Proportion of policies with claims
- **Claim Severity**: Average claim amount when claims occur
- **Loss Ratio**: TotalClaims/TotalPremium

## 8. Error Handling

### 8.1 Common Exceptions
| Error Type               | Cause                          | Resolution                     |
|--------------------------|--------------------------------|--------------------------------|
| MissingColumnError       | Required column not in data    | Verify input data structure    |
| InsufficientDataError    | Not enough samples for test    | Increase sample size           |
| InvalidSamplingMethod    | Unsupported sampling type      | Use valid method name          |

## 9. Performance Notes

### 9.1 Computational Complexity
| Operation               | Time Complexity | Space Complexity |
|-------------------------|-----------------|------------------|
| Data loading            | O(n)            | O(n)             |
| Random sampling         | O(1)            | O(k)             |
| Stratified sampling     | O(n)            | O(k)             |
| Statistical tests       | O(n)            | O(1)             |

### 9.2 Recommended Hardware
- Minimum: 8GB RAM for datasets <1M records
- Recommended: 16GB+ RAM for datasets >1M records

## 10. Version History

| Version | Date       | Changes                     |
|---------|------------|-----------------------------|
| 1.0     | 2025-11-15 | Initial release             |


This documentation follows standard technical documentation practices with clear section organization, method specifications, and practical usage examples while avoiding unnecessary marketing language.
