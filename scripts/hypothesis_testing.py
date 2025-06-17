import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency, ttest_ind, f_oneway
from typing import Dict, Union, List
from datetime import datetime
import warnings

class InsuranceRiskAnalyzer:
    """
    A comprehensive insurance risk analysis tool that performs hypothesis testing
    on key risk indicators across different dimensions (geography, demographics, etc.).
    """
    
    def __init__(self, file_path: str = "../data/data.csv", date_col: str = "TransactionMonth"):
        self.file_path = file_path
        self.date_col = date_col
        self.df = self._load_and_preprocess_data()
        
        # Configuration defaults
        self.sampling_defaults = {
            'random': {'sample_frac': 0.1},
            'stratified': {'stratify_col': 'Province', 'sample_frac': 0.1},
            'systematic': {'step': 100},
            'cluster': {'cluster_col': 'PostalCode', 'n_clusters': 20}
        }
        
        # Test configuration
        self.test_config = {
            'min_policies_per_group': 30,
            'top_n_zip_codes': 5,
            'recent_months_default': 12
        }

    def _load_and_preprocess_data(self) -> pd.DataFrame:
        """Load and preprocess the insurance data."""
        try:
            df = pd.read_csv(self.file_path, low_memory=False)
            
            if self.date_col in df.columns:
                df[self.date_col] = pd.to_datetime(df[self.date_col])
            
            self._calculate_derived_metrics(df)
            self._clean_categorical_variables(df)
            
            return df.dropna(subset=['TotalPremium', 'TotalClaims'])
        except Exception as e:
            warnings.warn(f"Error loading data: {str(e)}")
            return pd.DataFrame()

    def _calculate_derived_metrics(self, df: pd.DataFrame) -> None:
        """Calculate derived risk metrics."""
        if 'TotalPremium' in df.columns and 'TotalClaims' in df.columns:
            df['has_claim'] = df['TotalClaims'] > 0
            df['margin'] = df['TotalPremium'] - df['TotalClaims']
            df['loss_ratio'] = np.where(
                df['TotalPremium'] > 0,
                df['TotalClaims'] / df['TotalPremium'],
                np.nan
            )
            df['claim_frequency'] = df['has_claim'].astype(int)

    def _clean_categorical_variables(self, df: pd.DataFrame) -> None:
        """Clean and standardize categorical variables."""
        cat_cols = ['Province', 'PostalCode', 'Gender', 'VehicleType', 'CoverType']
        for col in cat_cols:
            if col in df.columns:
                if col == 'Gender':
                    df[col] = df[col].str.strip().str.title()
                df[col] = df[col].astype('category').cat.as_ordered()

    def get_sample(self, method: str = 'auto', **kwargs) -> pd.DataFrame:
        """Get a sample of the data using the specified method."""
        if method == 'auto':
            method = self._recommend_sampling_method()
            
        sampler = {
            'random': self._random_sample,
            'stratified': self._stratified_sample,
            'systematic': self._systematic_sample,
            'cluster': self._cluster_sample
        }.get(method.lower())
        
        if not sampler:
            raise ValueError(f"Unknown sampling method: {method}")
            
        return sampler(**kwargs)

    def _recommend_sampling_method(self) -> str:
        """Recommend the most appropriate sampling method."""
        if len(self.df) < 10000:
            return 'none'
        if 'PostalCode' in self.df.columns and len(self.df['PostalCode'].unique()) >= 20:
            return 'cluster'
        if 'Province' in self.df.columns and 2 <= len(self.df['Province'].unique()) <= 20:
            return 'stratified'
        if len(self.df) > 1000000:
            return 'systematic'
        return 'random'

    def _random_sample(self, sample_size: int = None, sample_frac: float = None, 
                      random_state: int = None) -> pd.DataFrame:
        """Simple random sampling."""
        params = self.sampling_defaults['random'].copy()
        if sample_size is not None or sample_frac is not None:
            params.update({'sample_size': sample_size, 'sample_frac': sample_frac})
        return self.df.sample(**params, random_state=random_state)

    def _stratified_sample(self, stratify_col: str = None, sample_size: int = None,
                          sample_frac: float = None, random_state: int = None) -> pd.DataFrame:
        """Stratified sampling by a specific column."""
        params = self.sampling_defaults['stratified'].copy()
        if stratify_col is not None:
            params['stratify_col'] = stratify_col
        if sample_size is not None or sample_frac is not None:
            params.update({'sample_size': sample_size, 'sample_frac': sample_frac})
            
        if params['stratify_col'] not in self.df.columns:
            raise ValueError(f"Stratification column {params['stratify_col']} not found")
            
        return self.df.groupby(params['stratify_col'], group_keys=False, observed=True).apply(
            lambda x: x.sample(frac=params['sample_frac'], random_state=random_state))

    def _systematic_sample(self, step: int = None, random_start: bool = True) -> pd.DataFrame:
        """Systematic sampling with fixed step interval."""
        params = self.sampling_defaults['systematic'].copy()
        if step is not None:
            params['step'] = step
            
        if random_start:
            start = np.random.randint(0, params['step'])
        else:
            start = 0
        indices = range(start, len(self.df), params['step'])
        return self.df.iloc[indices]

    def _cluster_sample(self, cluster_col: str = None, n_clusters: int = None,
                       random_state: int = None) -> pd.DataFrame:
        """Cluster sampling by selecting entire clusters."""
        params = self.sampling_defaults['cluster'].copy()
        if cluster_col is not None:
            params['cluster_col'] = cluster_col
        if n_clusters is not None:
            params['n_clusters'] = n_clusters
            
        if params['cluster_col'] not in self.df.columns:
            raise ValueError(f"Cluster column {params['cluster_col']} not found")
            
        clusters = self.df[params['cluster_col']].unique()
        selected = np.random.RandomState(random_state).choice(
            clusters, size=min(params['n_clusters'], len(clusters)), replace=False)
        return self.df[self.df[params['cluster_col']].isin(selected)]

    def run_risk_analysis(self, sampling_method: str = None, recent_months: int = None) -> Dict:
        """Run complete risk analysis with optional sampling and time filtering."""
        original_df = self.df
        
        try:
            if recent_months is not None:
                self.df = self._filter_recent_data(recent_months)
                
            if sampling_method is not None and sampling_method.lower() != 'none':
                self.df = self.get_sample(method=sampling_method)
                
            results = {
                'geographic_risk': self._test_geographic_risk(),
                'demographic_risk': self._test_demographic_risk(),
                'product_risk': self._test_product_risk(),
                'metadata': self._get_metadata()
            }
            
            return results
            
        finally:
            self.df = original_df

    def _test_geographic_risk(self) -> Dict:
        """Test risk differences across geographic dimensions."""
        return {
            'by_province': self._test_province_risk(),
            'by_zip_code': self._test_zip_code_risk()
        }

    def _test_demographic_risk(self) -> Dict:
        """Test risk differences across demographic dimensions."""
        return {
            'by_gender': self._test_gender_risk()
        }

    def _test_product_risk(self) -> Dict:
        """Test risk differences across product dimensions."""
        return {
            'by_vehicle_type': self._test_vehicle_type_risk()
        }

    def _test_province_risk(self) -> Dict[str, float]:
        """Test risk differences across provinces."""
        if 'Province' not in self.df.columns:
            return {'error': 'Province data not available'}
            
        freq_table = pd.crosstab(self.df['Province'], self.df['has_claim'])
        freq_result = self._run_chi2_test(freq_table) if freq_table.size > 1 else {'p_value': np.nan}
        severity_result = self._run_anova_test('Province', 'TotalClaims')
        
        return {
            'claim_frequency': freq_result,
            'claim_severity': severity_result
        }

    def _test_zip_code_risk(self) -> Dict[str, float]:
        """Test risk differences between highest and lowest risk zip codes."""
        if 'PostalCode' not in self.df.columns:
            return {'error': 'PostalCode data not available'}
            
        min_policies = self.test_config['min_policies_per_group']
        top_n = self.test_config['top_n_zip_codes']
        
        zip_stats = self.df.groupby('PostalCode', observed=True).agg(
            claim_rate=('has_claim', 'mean'),
            policy_count=('PolicyID', 'count')
        ).query(f'policy_count >= {min_policies}')
        
        if len(zip_stats) < 2:
            return {'error': 'Insufficient zip code data for comparison'}
            
        high_risk = zip_stats.nlargest(top_n, 'claim_rate').index
        low_risk = zip_stats.nsmallest(top_n, 'claim_rate').index
        
        freq_result = self._run_ttest(
            self.df['PostalCode'].isin(high_risk),
            self.df['PostalCode'].isin(low_risk),
            self.df['has_claim']
        )
        
        claim_df = self.df[self.df['has_claim']]
        severity_result = self._run_ttest(
            claim_df['PostalCode'].isin(high_risk),
            claim_df['PostalCode'].isin(low_risk),
            claim_df['TotalClaims']
        )
        
        return {
            'claim_frequency': freq_result,
            'claim_severity': severity_result,
            'high_risk_zips': list(high_risk),
            'low_risk_zips': list(low_risk)
        }

    def _test_gender_risk(self) -> Dict[str, float]:
        """Test risk differences between genders."""
        if 'Gender' not in self.df.columns:
            return {'error': 'Gender data not available'}
            
        gender_df = self.df[self.df['Gender'].isin(['Male', 'Female'])]
        
        if len(gender_df['Gender'].unique()) < 2:
            return {'error': 'Insufficient gender data for comparison'}
            
        freq_table = pd.crosstab(gender_df['Gender'], gender_df['has_claim'])
        freq_result = self._run_chi2_test(freq_table)
        
        claim_df = gender_df[gender_df['has_claim']]
        severity_result = self._run_ttest(
            claim_df['Gender'] == 'Male',
            claim_df['Gender'] == 'Female',
            claim_df['TotalClaims']
        )
        
        return {
            'claim_frequency': freq_result,
            'claim_severity': severity_result
        }

    def _test_vehicle_type_risk(self) -> Dict[str, float]:
        """Test risk differences between vehicle types."""
        if 'VehicleType' not in self.df.columns:
            return {'error': 'VehicleType data not available'}
            
        min_policies = self.test_config['min_policies_per_group']
        
        vehicle_stats = self.df.groupby('VehicleType', observed=True).agg(
            policy_count=('PolicyID', 'count')
        ).query(f'policy_count >= {min_policies}')
        
        if len(vehicle_stats) < 2:
            return {'error': 'Insufficient vehicle type data for comparison'}
            
        filtered_df = self.df[self.df['VehicleType'].isin(vehicle_stats.index)]
        
        freq_table = pd.crosstab(filtered_df['VehicleType'], filtered_df['has_claim'])
        freq_result = self._run_chi2_test(freq_table)
        
        severity_result = self._run_anova_test('VehicleType', 'TotalClaims', filtered_df[filtered_df['has_claim']])
        
        return {
            'claim_frequency': freq_result,
            'claim_severity': severity_result
        }

    def _run_chi2_test(self, contingency_table: pd.DataFrame) -> Dict:
        """Run chi-square test of independence."""
        if contingency_table.size == 0:
            return {'p_value': np.nan, 'error': 'Empty contingency table'}
            
        try:
            chi2, p, dof, expected = chi2_contingency(contingency_table)
            return {
                'p_value': p,
                'chi2_statistic': chi2,
                'degrees_of_freedom': dof,
                'is_significant': p < 0.05
            }
        except Exception as e:
            return {'p_value': np.nan, 'error': str(e)}

    def _run_ttest(self, group1_mask: pd.Series, group2_mask: pd.Series, values: pd.Series) -> Dict:
        """Run t-test between two groups."""
        group1 = values[group1_mask]
        group2 = values[group2_mask]
        
        if len(group1) < 2 or len(group2) < 2:
            return {'p_value': np.nan, 'error': 'Insufficient data points'}
            
        try:
            t_stat, p = ttest_ind(group1, group2, equal_var=False)
            return {
                'p_value': p,
                't_statistic': t_stat,
                'group1_mean': group1.mean(),
                'group2_mean': group2.mean(),
                'is_significant': p < 0.05
            }
        except Exception as e:
            return {'p_value': np.nan, 'error': str(e)}

    def _run_anova_test(self, group_col: str, value_col: str, df: pd.DataFrame = None) -> Dict:
        """Run one-way ANOVA test."""
        df = df if df is not None else self.df
        groups = df[group_col].unique()
        
        if len(groups) < 2:
            return {'p_value': np.nan, 'error': 'Insufficient groups for ANOVA'}
            
        try:
            group_data = [df[df[group_col] == g][value_col] for g in groups]
            f_stat, p = f_oneway(*group_data)
            return {
                'p_value': p,
                'f_statistic': f_stat,
                'n_groups': len(groups),
                'is_significant': p < 0.05
            }
        except Exception as e:
            return {'p_value': np.nan, 'error': str(e)}

    def _filter_recent_data(self, months: int) -> pd.DataFrame:
        """Filter data to most recent N months."""
        if self.date_col not in self.df.columns:
            warnings.warn(f"Date column {self.date_col} not found - skipping time filter")
            return self.df
            
        max_date = self.df[self.date_col].max()
        cutoff_date = max_date - pd.DateOffset(months=months)
        return self.df[self.df[self.date_col] >= cutoff_date]

    def _get_metadata(self) -> Dict:
        """Get metadata about the current analysis."""
        return {
            'timestamp': datetime.now().isoformat(),
            'data_points': len(self.df),
            'start_date': self.df[self.date_col].min().isoformat() if self.date_col in self.df.columns else None,
            'end_date': self.df[self.date_col].max().isoformat() if self.date_col in self.df.columns else None,
            'claim_rate': self.df['has_claim'].mean() if 'has_claim' in self.df.columns else None
        }

    def generate_report(self, results: Dict) -> str:
        """Generate a comprehensive business report from test results."""
        report = [
            "Insurance Risk Analysis Report",
            "=" * 50,
            f"Analysis Date: {results['metadata']['timestamp']}",
            f"Policies Analyzed: {results['metadata']['data_points']:,}",
            f"Time Period: {results['metadata']['start_date']} to {results['metadata']['end_date']}",
            f"Overall Claim Rate: {results['metadata']['claim_rate']:.1%}",
            "\nKey Findings:"
        ]
        
        # Geographic Risk
        report.extend(self._format_geographic_results(results['geographic_risk']))
        
        # Demographic Risk
        report.extend(self._format_demographic_results(results['demographic_risk']))
        
        # Product Risk
        report.extend(self._format_product_results(results['product_risk']))
        
        # Recommendations
        report.append("\nRecommendations:")
        report.extend(self._generate_recommendations(results))
        
        return "\n".join(report)

    def _format_geographic_results(self, results: Dict) -> List[str]:
        """Format geographic risk results for reporting."""
        output = ["\n1. Geographic Risk Variations:"]
        
        # Province results
        prov = results.get('by_province', {})
        if 'error' in prov:
            output.append(f"   - Provinces: {prov['error']}")
        else:
            output.append("   - By Province:")
            if 'claim_frequency' in prov:
                p = prov['claim_frequency']['p_value']
                output.append(f"      • Claim Frequency: {'SIGNIFICANT' if p < 0.05 else 'No significant'} differences (p={p:.4f})")
            if 'claim_severity' in prov:
                p = prov['claim_severity']['p_value']
                output.append(f"      • Claim Severity: {'SIGNIFICANT' if p < 0.05 else 'No significant'} differences (p={p:.4f})")
        
        # Zip code results
        zip_risk = results.get('by_zip_code', {})
        if 'error' in zip_risk:
            output.append(f"   - Zip Codes: {zip_risk['error']}")
        else:
            output.append("   - By Zip Code (Highest vs. Lowest Risk):")
            if 'claim_frequency' in zip_risk:
                p = zip_risk['claim_frequency']['p_value']
                output.append(f"      • Claim Frequency: {'SIGNIFICANT' if p < 0.05 else 'No significant'} differences (p={p:.4f})")
            if 'claim_severity' in zip_risk:
                p = zip_risk['claim_severity']['p_value']
                output.append(f"      • Claim Severity: {'SIGNIFICANT' if p < 0.05 else 'No significant'} differences (p={p:.4f})")
            if 'high_risk_zips' in zip_risk:
                output.append(f"      • Highest Risk Zip Codes: {', '.join(map(str, zip_risk['high_risk_zips'][:5]))}...")
        
        return output

    def _format_demographic_results(self, results: Dict) -> List[str]:
        """Format demographic risk results for reporting."""
        output = ["\n2. Demographic Risk Variations:"]
        
        # Gender results
        gender = results.get('by_gender', {})
        if 'error' in gender:
            output.append(f"   - Gender: {gender['error']}")
        else:
            output.append("   - By Gender:")
            if 'claim_frequency' in gender:
                p = gender['claim_frequency']['p_value']
                output.append(f"      • Claim Frequency: {'SIGNIFICANT' if p < 0.05 else 'No significant'} differences (p={p:.4f})")
            if 'claim_severity' in gender:
                p = gender['claim_severity']['p_value']
                output.append(f"      • Claim Severity: {'SIGNIFICANT' if p < 0.05 else 'No significant'} differences (p={p:.4f})")
        
        return output

    def _format_product_results(self, results: Dict) -> List[str]:
        """Format product risk results for reporting."""
        output = ["\n3. Product Risk Variations:"]
        
        # Vehicle type results
        vehicle = results.get('by_vehicle_type', {})
        if 'error' in vehicle:
            output.append(f"   - Vehicle Types: {vehicle['error']}")
        else:
            output.append("   - By Vehicle Type:")
            if 'claim_frequency' in vehicle:
                p = vehicle['claim_frequency']['p_value']
                output.append(f"      • Claim Frequency: {'SIGNIFICANT' if p < 0.05 else 'No significant'} differences (p={p:.4f})")
            if 'claim_severity' in vehicle:
                p = vehicle['claim_severity']['p_value']
                output.append(f"      • Claim Severity: {'SIGNIFICANT' if p < 0.05 else 'No significant'} differences (p={p:.4f})")
        
        return output

    def _generate_recommendations(self, results: Dict) -> List[str]:
        """Generate business recommendations based on analysis results."""
        recommendations = []
        
        # Geographic recommendations
        geo = results.get('geographic_risk', {})
        prov = geo.get('by_province', {})
        zip_risk = geo.get('by_zip_code', {})
        
        if prov.get('claim_frequency', {}).get('is_significant', False):
            recommendations.append("- Implement province-specific premium adjustments")
        if zip_risk.get('claim_frequency', {}).get('is_significant', False):
            recommendations.append("- Review underwriting guidelines for high-risk zip codes")
        
        # Demographic recommendations
        demo = results.get('demographic_risk', {})
        gender = demo.get('by_gender', {})
        
        if not gender.get('claim_frequency', {}).get('is_significant', True):
            recommendations.append("- Maintain gender-neutral rating approach")
        
        # Product recommendations
        product = results.get('product_risk', {})
        vehicle = product.get('by_vehicle_type', {})
        
        if vehicle.get('claim_frequency', {}).get('is_significant', False):
            recommendations.append("- Review vehicle type risk factors and pricing")
        
        if not recommendations:
            recommendations.append("- No significant risk variations detected - current pricing approach appears appropriate")
        
        return recommendations


# Example Usage
if __name__ == "__main__":
    analyzer = InsuranceRiskAnalyzer(file_path="data/data.csv")
    
    if not analyzer.df.empty:
        print(f"Loaded {len(analyzer.df):,} policies")
        
        results = analyzer.run_risk_analysis(
            sampling_method='auto',
            recent_months=12
        )
        
        report = analyzer.generate_report(results)
        print("\n" + report)
        
        with open("insurance_risk_report.txt", "w") as f:
            f.write(report)
    else:
        print("Failed to load data. Please check the file path.")