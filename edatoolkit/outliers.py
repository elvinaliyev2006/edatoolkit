import pandas as pd
import numpy as np

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', None)


class OutlierAnalyzer:

    def __init__(self):
        
        self.line = '─' * 170


    def check_outlier(self,dataframe, num_cols, num_summary_df, iqr_th=1.5, z_score_th=3, remove=False, cap=False):

            """
            Detects outliers in numerical columns using Z-score (for normal distributions)
            or IQR method (for non-normal distributions).

            Parameters
            ----------
            iqr_th : float, optional
                IQR multiplier for outlier boundaries (default: 1.5).
            z_score_th : int, optional
                Z-score threshold for outlier detection (default: 3).
            remove : bool, optional
                If True, removes outlier rows and returns cleaned dataframe (default: False).
            cap : bool, optional
                If True, caps outliers at upper/lower limits instead of removing (default: False).
                Cannot be used together with remove=True.

            Returns
            -------
            dict or tuple
                Outlier report dict, or (outlier_report, cleaned_df) if remove=True or cap=True.
            """

            if cap and remove:
                raise ValueError('! remove and cap cannot both be True. Choose one.')
            if num_cols:
                print(f"\n{self.line}")
                print(" Outlier Detection ".center(170))
                print(self.line)
                outlier_report = {}
                cleaned_df = dataframe.copy()
                for col in num_cols:
                    data = cleaned_df[col]
                    n = len(data)
                    is_normal = num_summary_df[num_summary_df['Column'] == col]['Result']
                    is_normal = is_normal.values[0] == 'Normal'
                    if is_normal:
                        lower = data.mean() - z_score_th * data.std()
                        upper = data.mean() + z_score_th * data.std()
                        z_scores = np.abs((data - data.mean()) / data.std())
                        outliers = data[z_scores > z_score_th]
                        method = "z-score"
                    else:
                        q1 = data.quantile(0.25)
                        q3 = data.quantile(0.75)
                        iqr = q3 - q1
                        lower = q1 - iqr_th * iqr
                        upper = q3 + iqr_th * iqr
                        outliers = data[(data < lower) | (data > upper)]
                        method = "IQR"
                    outlier_ratio = outliers.shape[0] / n
                    outlier_report[col] = {"method": method,
                                        "outlier_count": outliers.shape[0],
                                        "outlier_ratio": outlier_ratio}
                    print(f"Column: {col} | Method: {method} | Outliers: {outliers.shape[0]} ({outliers.shape[0] / n:.2%})")
                    if remove and outliers.shape[0] > 0:
                        cleaned_df = cleaned_df.drop(outliers.index)
                    elif cap and outliers.shape[0] > 0:
                        cleaned_df[col] = cleaned_df[col].clip(lower=lower, upper=upper)
                        print(f"  → '{col}' capped at [{lower:.4f}, {upper:.4f}]")
                if remove:
                    print("! Outliers removed. Run num_summary() again before further analysis.")
                    return outlier_report, cleaned_df.reset_index(drop=True)
                elif cap:
                    print("\n! Outliers capped. Run num_summary() again before further analysis.")
                    return outlier_report, cleaned_df
                else:
                    return outlier_report
            else:
                raise ValueError('! num_cols is empty')