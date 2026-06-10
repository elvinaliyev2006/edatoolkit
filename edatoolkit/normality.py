import pandas as pd
import numpy as np
import scipy
import matplotlib.pyplot as plt


pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', None)


class NormalityAnalyzer:
    
    def __init__(self):
        self.line = '─' * 170
    

    def descriptive_analysis(self,dataframe,num_cols):

        """
        Prints a detailed descriptive statistics table for all numerical columns,
        including percentiles, median, coefficient of variation, skewness, and kurtosis.
        """

        if num_cols:
            df_desc = dataframe[num_cols].describe([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).T
            cv = (dataframe[num_cols].std() / dataframe[num_cols].mean().replace(0,
                                                                                                     np.nan) * 100).round(
                3)
            cv = cv.replace([np.inf, -np.inf], np.nan)
            median = dataframe[num_cols].median()
            skew = dataframe[num_cols].skew()
            kurtosis = dataframe[num_cols].kurtosis()
            cv_col = pd.DataFrame(cv).T
            cv_col.index = ['cv%']
            median_col = pd.DataFrame(median).T
            median_col.index = ['median']
            skew_col = pd.DataFrame(skew).T
            skew_col.index = ['skewness']
            kurtosis_col = pd.DataFrame(kurtosis).T
            kurtosis_col.index = ['kurtosis']
            df_desc = pd.concat([df_desc.T, median_col, cv_col, skew_col, kurtosis_col]).T
            df_desc.columns = ['count', 'mean', 'std', 'min', '1%', '5%', '25%', '50%', '75%', '95%', '99%', 'max',
                               'median', 'cv%', 'skewness', 'kurtosis']
            print(f'\n{self.line}')
            print(' Descriptive Analysis '.center(170))
            print(self.line)
            print(df_desc)
            print(self.line)
        else:
            raise ValueError('! num_cols is empty')
        

    def check_num(self, dataframe, num_cols, alpha=0.05, plot=False, width_for_graph=15, height_for_graph=5):

        """
        Displays visual diagnostics and normality test results for each numerical column.

        For each column, optionally generates a combined plot consisting of a Q-Q plot,
        histogram, and box plot. Runs Shapiro-Wilk test for n <= 2500 or
        D'Agostino K² test for n > 2500, and reports the p-value alongside
        the test conclusion.

        Columns flagged as non-normal by the test should be verified visually
        before making a final decision. Use the returned list to construct the
        normality dictionary for num_summary().

        Parameters
        ----------
        plot : bool, optional
            If True, displays Q-Q plot, histogram, and box plot for each column (default: False).
        width_for_graph : int, optional
            Width of each figure in inches (default: 15).
        height_for_graph : int, optional
            Height of each figure in inches (default: 5).

        Returns
        -------
        list
            List of column names flagged as non-normal by the statistical test.

        Example
        -------
        non_normals = eda.check_num()
        # Visually inspect the plots, then:
        eda.num_summary(result_dict={'age': 'Normal', 'salary': 'Non-normal'})

        # To enable plots:
        non_normals = eda.check_num(plot=True)
        """

        if num_cols:
            print(f'\n{self.line}')
            print(' Numerical Variable Summary '.center(170))
            print(self.line)
            result = []
            for col in num_cols:
                data = dataframe[col].dropna()

                if plot:
                    fig, axes = plt.subplots(1, 3, figsize=(width_for_graph, height_for_graph))
                    fig.patch.set_facecolor('white')
                    for ax in axes:
                        ax.set_facecolor('white')
                        ax.grid(False)
                        for spine in ax.spines.values():
                            spine.set_edgecolor('#cccccc')
                    fig.suptitle(col, fontsize=14, fontweight='bold')

                    (osm, osr), (slope, intercept, r) = scipy.stats.probplot(data)
                    axes[0].scatter(osm, osr, color='#4682B4', s=15, alpha=0.7)
                    axes[0].plot(osm, slope * np.array(osm) + intercept, color='red', linewidth=2)
                    axes[0].set_title('Q-Q Plot')
                    axes[0].set_xlabel('Theoretical Quantiles')
                    axes[0].set_ylabel(f'{col} (Sample Quantiles)')

                    axes[1].hist(data, bins='auto', color='#4682B4', alpha=0.7, edgecolor='white')
                    axes[1].set_title('Histogram')
                    axes[1].set_xlabel(col)
                    axes[1].set_ylabel('Frequency')

                    axes[2].boxplot(data, vert=False, patch_artist=True,
                                    boxprops=dict(facecolor='#4682B4', alpha=0.7),
                                    medianprops=dict(color='red', linewidth=2),
                                    whiskerprops=dict(color='#4682B4'),
                                    capprops=dict(color='#4682B4'),
                                    flierprops=dict(marker='o', color='#4682B4', alpha=0.5))
                    axes[2].set_title('Box Plot')
                    axes[2].set_xlabel(col)
                    axes[2].set_yticks([])

                    plt.tight_layout()
                    plt.show()

                n = len(data)
                if n <= 2500:
                    test_stat, p_value = scipy.stats.shapiro(data)
                    test_name = "Shapiro-Wilk"
                else:
                    test_stat, p_value = scipy.stats.normaltest(data)
                    test_name = "D'Agostino K²"
                print(f"Column: {col}")
                print(f"Test: {test_name}")
                print(f"Test Statistic: {test_stat:.4f}, p-value: {p_value:.4f}")
                if p_value > alpha:
                    print(f"Result: Based on the {test_name} test (p={p_value:.4f} > {alpha}), "
                          f"\nthe sample appears Gaussian. However, please verify using the visuals above before making a final decision. "
                          f"\nTo override, pass result_dict={{'col_name': 'Normal/Non-normal'}} to num_summary().")
                else:
                    print(f"Result: Based on the {test_name} test (p={p_value:.4f} ≤ {alpha}), "
                          f"\nthe sample does not appear Gaussian. However, please verify using the visuals above before making a final decision. "
                          f"\nTo override, pass result_dict={{'col_name': 'Normal/Non-normal'}} to num_summary().")
                    result.append(col)
                print(self.line)
            print(f"\nNote: The results above are based on statistical tests only. "
                  f"\nPlease verify using the visuals before making a final decision. "
                  f"\nTo manually set normality, pass result_dict={{'col_name': 'Normal/Non-normal'}} "
                  f"\nto num_summary().")
            print(self.line)
            if result:
                print(f"\nColumns flagged as Non-normal by the test — please verify visually: {result}")
            else:
                print(f"\nAll columns appear Gaussian according to the test.")
            return result
        else:
            raise ValueError('! num_cols is empty')
        

    def num_summary(self,num_cols, result_dict):

        """
         Creates the normality summary DataFrame used across all analysis methods.

         Accepts a dictionary of column-level normality decisions and maps them
         to all numerical columns. Columns not present in result_dict are assumed
         to be normally distributed.

         Run check_num() first to visually inspect each column and identify
         non-normal distributions before constructing result_dict.

         Parameters
         ----------
         result_dict : dict
             Dictionary mapping column names to normality decisions.
             Accepted values: 'Normal' or 'Non-normal'.
             Columns not included default to 'Normal'.

         Returns
         -------
         pd.DataFrame
             DataFrame with columns ['Column', 'Result'] stored as self.num_summary_df.

         Example
         -------
         eda.check_num()
         eda.num_summary(result_dict={
             'age'    : 'Normal',
             'salary' : 'Non-normal',
             'height' : 'Non-normal'
         })
         """

        result_df = pd.DataFrame({
            'Column': num_cols,
            'Result': [result_dict.get(col, 'Normal') for col in num_cols]
        })
        return  result_df 