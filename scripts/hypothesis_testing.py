import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency, ttest_ind, f_oneway
from typing import Dict, Union, List, Tuple
from datetime import datetime
import warnings

class InsuranceRiskAnalyzer:
    """
    Enhanced insurance risk analyzer with robust handling of sample size requirements.
    """
    
    def __init__(self, file_path: str = "../data/data.csv", date_col: str = "TransactionMonth"):
        self.file_path = file_path
        self.date_col = date_col
        self.df = self._load_and_preprocess_data()
        
        # Configuration
        self.alpha = 0.05  # Significance level
        self.min_sample_size = 30  # Minimum samples per group
        self.min_policies_for_test = 10  # Minimum to attempt test (with warning)
        
    def _load_and_preprocess_data(self) -> pd.DataFrame:
        """Load and preprocess the insurance data."""
        try:
            df = pd.read_csv(self.file_path, low_memory=False)
            
            if self.date_col in df.columns:
                df[self.date_col] = pd.to_datetime(df[self.date_col])
            
            # Calculate required metrics
            df['has_claim'] = df['TotalClaims'] > 0
            df['margin'] = df['TotalPremium'] - df['TotalClaims']
            df['claim_frequency'] = df['has_claim'].astype(int)
            
            # Clean categoricals
            for col in ['Province', 'PostalCode', 'Gender', 'VehicleType']:
                if col in df.columns:
                    if col == 'Gender':
                        df[col] = df[col].str.strip().str.title()
                    df[col] = df[col].astype('category')
            
            return df.dropna(subset=['TotalPremium', 'TotalClaims'])
        except Exception as e:
            warnings.warn(f"Error loading data: {str(e)}")
            return pd.DataFrame()

    def _check_group_balance(self, group_a: pd.DataFrame, group_b: pd.DataFrame, 
                           balance_cols: List[str]) -> bool:
        """
        Check if control and test groups are balanced on specified columns.
        Returns True if groups are balanced (no significant differences).
        """
        for col in balance_cols:
            if col not in group_a.columns or col not in group_b.columns:
                continue
                
            if pd.api.types.is_numeric_dtype(group_a[col]):
                # Use t-test for numeric
                _, p = ttest_ind(group_a[col].dropna(), group_b[col].dropna())
            else:
                # Use chi2 for categorical
                contingency = pd.crosstab(
                    pd.concat([group_a[col], group_b[col]]),
                    np.concatenate([
                        np.zeros(len(group_a)),
                        np.ones(len(group_b))
                    ])
                )
                if contingency.size > 1:
                    _, p, _, _ = chi2_contingency(contingency)
                else:
                    p = 1
            
            if p < self.alpha:
                warnings.warn(f"Groups are not balanced on {col} (p={p:.4f})")
                return False
        return True

    def run_ab_test(self, feature_col: str, group_a_value, group_b_value=None,
                   balance_cols: List[str] = None) -> Dict:
        """
        Perform A/B test comparing two groups on risk metrics.
        If group_b_value is None, compares group_a_value against all others.
        """
        if feature_col not in self.df.columns:
            return {'error': f"Feature column {feature_col} not found"}
            
        # Create groups
        group_a = self.df[self.df[feature_col] == group_a_value]
        
        if group_b_value is None:
            group_b = self.df[self.df[feature_col] != group_a_value]
            comparison = f"{group_a_value} vs Others"
        else:
            group_b = self.df[self.df[feature_col] == group_b_value]
            comparison = f"{group_a_value} vs {group_b_value}"
        
        # Check sample sizes
        sample_sizes = {'group_a': len(group_a), 'group_b': len(group_b)}
        if len(group_a) < self.min_policies_for_test or len(group_b) < self.min_policies_for_test:
            return {
                'error': f"Insufficient samples ({len(group_a)} vs {len(group_b)})",
                'comparison': comparison,
                'sample_sizes': sample_sizes
            }
        elif len(group_a) < self.min_sample_size or len(group_b) < self.min_sample_size:
            warnings.warn(f"Small sample sizes ({len(group_a)} vs {len(group_b)}) - results may be unreliable")
        
        # Check group balance if specified
        is_balanced = True
        if balance_cols:
            is_balanced = self._check_group_balance(group_a, group_b, balance_cols)
        
        # Perform tests
        results = {
            'feature': feature_col,
            'comparison': comparison,
            'sample_sizes': sample_sizes,
            'is_balanced': is_balanced,
            'tests': {}
        }
        
        # Claim Frequency (Chi-square)
        freq_contingency = pd.crosstab(
            pd.concat([group_a['has_claim'], group_b['has_claim']]),
            np.concatenate([np.zeros(len(group_a)), np.ones(len(group_b))])
        )
        if freq_contingency.size > 1:
            chi2, p_freq, _, _ = chi2_contingency(freq_contingency)
            results['tests']['claim_frequency'] = {
                'test': 'chi2',
                'statistic': chi2,
                'p_value': p_freq,
                'group_a_rate': group_a['has_claim'].mean(),
                'group_b_rate': group_b['has_claim'].mean(),
                'reject_null': p_freq < self.alpha
            }
        
        # Claim Severity (t-test) - only for policies with claims
        severity_a = group_a[group_a['has_claim']]['TotalClaims']
        severity_b = group_b[group_b['has_claim']]['TotalClaims']
        if len(severity_a) >= 2 and len(severity_b) >= 2:
            t_stat, p_sev = ttest_ind(severity_a, severity_b, equal_var=False)
            results['tests']['claim_severity'] = {
                'test': 't-test',
                'statistic': t_stat,
                'p_value': p_sev,
                'group_a_mean': severity_a.mean(),
                'group_b_mean': severity_b.mean(),
                'reject_null': p_sev < self.alpha
            }
        
        # Margin (t-test)
        if len(group_a) >= 2 and len(group_b) >= 2:
            t_stat, p_margin = ttest_ind(group_a['margin'], group_b['margin'], equal_var=False)
            results['tests']['margin'] = {
                'test': 't-test',
                'statistic': t_stat,
                'p_value': p_margin,
                'group_a_mean': group_a['margin'].mean(),
                'group_b_mean': group_b['margin'].mean(),
                'reject_null': p_margin < self.alpha
            }
        
        return results

    def test_province_risk(self) -> Dict:
        """Test risk differences across provinces."""
        if 'Province' not in self.df.columns:
            return {'error': 'Province data not available'}
            
        provinces = self.df['Province'].unique()
        if len(provinces) < 2:
            return {'error': 'Insufficient provinces for comparison'}
            
        # Compare each province to all others
        results = {}
        for province in provinces:
            province_test = self.run_ab_test(
                feature_col='Province',
                group_a_value=province,
                balance_cols=['Gender', 'VehicleType']
            )
            if 'error' not in province_test:
                results[province] = province_test
            elif province_test['sample_sizes']['group_a'] >= self.min_policies_for_test:
                # Include provinces that had some data but not enough for full test
                results[province] = province_test
        
        return results if results else {'error': 'No provinces met minimum sample requirements'}

    def test_zip_code_risk(self, top_n: int = 5) -> Dict:
        """Test risk differences between highest/lowest risk zip codes."""
        if 'PostalCode' not in self.df.columns:
            return {'error': 'PostalCode data not available'}
            
        # Identify high/low risk zip codes with sufficient policies
        zip_stats = self.df.groupby('PostalCode', observed=True).agg(
            claim_rate=('has_claim', 'mean'),
            policy_count=('PolicyID', 'count')
        ).query(f'policy_count >= {self.min_policies_for_test}')
        
        if len(zip_stats) < 2:
            return {'error': 'Insufficient zip codes for comparison'}
            
        high_risk = zip_stats.nlargest(top_n, 'claim_rate').index
        low_risk = zip_stats.nsmallest(top_n, 'claim_rate').index
        
        # Compare high vs low risk groups
        high_risk_df = self.df[self.df['PostalCode'].isin(high_risk)]
        low_risk_df = self.df[self.df['PostalCode'].isin(low_risk)]
        
        # Check balance
        is_balanced = self._check_group_balance(high_risk_df, low_risk_df, ['Gender', 'VehicleType'])
        
        # Run tests
        results = {
            'high_risk_zips': list(high_risk),
            'low_risk_zips': list(low_risk),
            'sample_sizes': {
                'high_risk': len(high_risk_df),
                'low_risk': len(low_risk_df)
            },
            'is_balanced': is_balanced,
            'tests': {}
        }
        
        # Claim Frequency
        freq_contingency = pd.crosstab(
            pd.concat([high_risk_df['has_claim'], low_risk_df['has_claim']]),
            np.concatenate([np.zeros(len(high_risk_df)), np.ones(len(low_risk_df))])
        )
        if freq_contingency.size > 1:
            chi2, p_freq, _, _ = chi2_contingency(freq_contingency)
            results['tests']['claim_frequency'] = {
                'test': 'chi2',
                'statistic': chi2,
                'p_value': p_freq,
                'high_risk_rate': high_risk_df['has_claim'].mean(),
                'low_risk_rate': low_risk_df['has_claim'].mean(),
                'reject_null': p_freq < self.alpha
            }
        
        # Claim Severity
        severity_high = high_risk_df[high_risk_df['has_claim']]['TotalClaims']
        severity_low = low_risk_df[low_risk_df['has_claim']]['TotalClaims']
        if len(severity_high) >= 2 and len(severity_low) >= 2:
            t_stat, p_sev = ttest_ind(severity_high, severity_low, equal_var=False)
            results['tests']['claim_severity'] = {
                'test': 't-test',
                'statistic': t_stat,
                'p_value': p_sev,
                'high_risk_mean': severity_high.mean(),
                'low_risk_mean': severity_low.mean(),
                'reject_null': p_sev < self.alpha
            }
        
        # Margin
        if len(high_risk_df) >= 2 and len(low_risk_df) >= 2:
            t_stat, p_margin = ttest_ind(high_risk_df['margin'], low_risk_df['margin'], equal_var=False)
            results['tests']['margin'] = {
                'test': 't-test',
                'statistic': t_stat,
                'p_value': p_margin,
                'high_risk_mean': high_risk_df['margin'].mean(),
                'low_risk_mean': low_risk_df['margin'].mean(),
                'reject_null': p_margin < self.alpha
            }
        
        return results

    def test_gender_risk(self) -> Dict:
        """Test risk differences between genders."""
        if 'Gender' not in self.df.columns:
            return {'error': 'Gender data not available'}
            
        valid_genders = ['Male', 'Female']
        gender_df = self.df[self.df['Gender'].isin(valid_genders)]
        
        if len(gender_df['Gender'].unique()) < 2:
            return {'error': 'Insufficient gender data for comparison'}
            
        return self.run_ab_test(
            feature_col='Gender',
            group_a_value='Male',
            group_b_value='Female',
            balance_cols=['VehicleType', 'Province']
        )

    def generate_test_report(self, test_results: Dict) -> str:
        """Generate a report from test results."""
        report = []
        
        if 'error' in test_results:
            if 'sample_sizes' in test_results:
                return (
                    f"Analysis not performed: {test_results['error']}\n"
                    f"Sample sizes: {test_results['sample_sizes']}"
                )
            return f"Analysis not performed: {test_results['error']}"
        
        # Header
        if 'comparison' in test_results:
            report.append(f"A/B Test Results: {test_results['comparison']}")
        else:
            report.append("Risk Comparison Results")
        
        report.append("=" * 50)
        
        # Sample information
        if 'sample_sizes' in test_results:
            report.append(
                "Sample Sizes:\n" +
                "\n".join([f"  - {k}: {v:,}" for k, v in test_results['sample_sizes'].items()])
            )
        
        if 'is_balanced' in test_results and not test_results['is_balanced']:
            report.append("\nWarning: Groups are not balanced - results may be biased")
        
        # Test results
        for metric, test in test_results.get('tests', {}).items():
            report.append(f"\n{metric.replace('_', ' ').title()}:")
            report.append(f"  - Test: {test['test']}")
            report.append(f"  - Statistic: {test['statistic']:.4f}")
            report.append(f"  - p-value: {test['p_value']:.4f}")
            
            # Format comparison values appropriately
            if 'rate' in test:
                report.append(
                    f"  - Group Values: {test.get('group_a_rate', 0):.2%} vs "
                    f"{test.get('group_b_rate', 0):.2%}"
                )
            else:
                report.append(
                    f"  - Group Values: {test.get('group_a_mean', 0):.2f} vs "
                    f"{test.get('group_b_mean', 0):.2f}"
                )
            
            report.append(
                f"  - Conclusion: {'REJECT' if test['reject_null'] else 'FAIL TO REJECT'} null hypothesis"
            )
        
        # Recommendations
        report.append("\nRecommendations:")
        any_significant = False
        for metric, test in test_results.get('tests', {}).items():
            if test['reject_null']:
                any_significant = True
                if metric == 'claim_frequency':
                    report.append(
                        f"- Significant difference in claim frequency detected. "
                        f"Consider adjusting underwriting criteria or pricing."
                    )
                elif metric == 'claim_severity':
                    report.append(
                        f"- Significant difference in claim severity detected. "
                        f"Review coverage terms or claims handling processes."
                    )
                elif metric == 'margin':
                    report.append(
                        f"- Significant difference in profitability detected. "
                        f"Evaluate pricing strategy by segment."
                    )
        
        if not any_significant:
            report.append("- No significant differences found. Current approach appears appropriate.")
        
        return "\n".join(report)

    def run_full_analysis(self) -> Dict:
        """Run all requested hypothesis tests with robust error handling."""
        results = {}
        reports = {}
        
        # Province risk (with error handling)
        try:
            province_results = self.test_province_risk()
            results['province_risk'] = province_results
            if isinstance(province_results, dict) and 'error' not in province_results:
                for province, province_test in province_results.items():
                    reports[f'province_{province}'] = self.generate_test_report(province_test)
            else:
                reports['province_risk'] = self.generate_test_report(province_results)
        except Exception as e:
            results['province_risk'] = {'error': str(e)}
            reports['province_risk'] = f"Province analysis failed: {str(e)}"
        
        # Zip code risk
        try:
            zip_results = self.test_zip_code_risk()
            results['zip_code_risk'] = zip_results
            reports['zip_code_risk'] = self.generate_test_report(zip_results)
        except Exception as e:
            results['zip_code_risk'] = {'error': str(e)}
            reports['zip_code_risk'] = f"Zip code analysis failed: {str(e)}"
        
        # Gender risk
        try:
            gender_results = self.test_gender_risk()
            results['gender_risk'] = gender_results
            reports['gender_risk'] = self.generate_test_report(gender_results)
        except Exception as e:
            results['gender_risk'] = {'error': str(e)}
            reports['gender_risk'] = f"Gender analysis failed: {str(e)}"
        
        return {
            'results': results,
            'reports': reports
        }