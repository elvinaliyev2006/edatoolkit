import pandas as pd
import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import scipy


pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', None)


class EDA:
    """
    A comprehensive Exploratory Data Analysis (EDA) class that provides
    statistical summaries, visualizations, and hypothesis tests for a given dataset.

    Parameters
    ----------
    dataframe : pd.DataFrame
        The dataset to analyze.
    target_col : str
        The target column name for supervised analysis.
    cat_th : int, optional
        Threshold for treating a numeric column as categorical (default: 10).
    car_th : int, optional
        Threshold for treating a categorical column as high-cardinality (default: 20).
    alpha : float, optional
        Significance level for hypothesis tests (default: 0.05).
    """

    line = '─' * 170

    def __init__(self, dataframe, target_col, cat_th=10, car_th=20, alpha=0.05,):
        self.dataframe = dataframe
        self.target_col = target_col
        self.cat_th = cat_th
        self.car_th = car_th
        self.alpha = alpha
        self.cat_cols, self.num_cols, self.num_but_cat, self.cat_but_car = self.get_columns_types()
        self.num_summary_df = None
        self.outlier_report = None

    def update_dataframe(self, new_df):
        """
        Updates the dataframe and automatically recalculates all column types.

        Parameters
        ----------
        new_df : pd.DataFrame
            The new dataset to be assigned to the class instance.
        """
        self.dataframe = new_df
        self.cat_cols, self.num_cols, self.num_but_cat, self.cat_but_car = self.get_columns_types()
        self.num_summary_df = None
        print("Dataframe and column types have been successfully updated.")


    def check_dataframe(self, n=5):

        """
        Prints a general overview of the dataframe including head, sample, tail,
        shape, info, missing values, and duplicate counts.

        Parameters
        ----------
        n : int, optional
            Number of rows to display for head/sample/tail (default: 5).
        """

        print(f'\n{self.line}')
        print(' Head '.center(170))
        print(self.line)
        print(self.dataframe.head(n))
        print(f'\n{self.line}')
        print(' Sample '.center(170))
        print(self.line)
        print(self.dataframe.sample(n))
        print(f'\n{self.line}')
        print(' Tail '.center(170))
        print(self.line)
        print(self.dataframe.tail(n))
        print(f'\n{self.line}')
        print(' Shape '.center(170))
        print(self.line)
        print('Rows: ', self.dataframe.shape[0])
        print('Columns: ', self.dataframe.shape[1])
        print(f'\n{self.line}')
        print(' Info '.center(170))
        print(self.line)
        print(self.dataframe.info())
        print(f'\n{self.line}')
        print(' NA '.center(170))
        print(self.line)
        print(self.dataframe.isnull().mean())
        print(f'\n{self.line}')
        print(' Duplicate Values '.center(170))
        print(self.line)
        print('Count: ', self.dataframe.duplicated().sum())
        print('Ratio: ', (self.dataframe.duplicated().sum()) / (self.dataframe.shape[0]))

    def get_columns_types(self):

        """
        Identifies and categorizes columns into categorical, numerical,
        numeric-but-categorical, and categorical-but-high-cardinality groups.

        Returns
        -------
        tuple
            (cat_cols, num_cols, num_but_cat, cat_but_car)
        """

        cat_cols = [col for col in self.dataframe.columns if
                    str(self.dataframe[col].dtype) in ['object', 'bool', 'category','str', 'string']]
        num_but_cat = [col for col in self.dataframe.columns if
                       pd.api.types.is_numeric_dtype(self.dataframe[col]) and self.dataframe[
                           col].nunique() < self.cat_th]
        cat_but_car = [col for col in self.dataframe.columns if
                       str(self.dataframe[col].dtype) in ['object', 'category'] and self.dataframe[
                           col].nunique() > self.car_th]
        c_c = [col for col in num_but_cat if col not in cat_cols ]
        cat_cols = cat_cols + c_c
        cat_cols = [col for col in cat_cols if col not in cat_but_car]
        num_cols = [col for col in self.dataframe.columns if
                    pd.api.types.is_numeric_dtype(self.dataframe[col]) and col not in num_but_cat]
        return cat_cols, num_cols, num_but_cat, cat_but_car

    def descriptive_analysis(self):

        """
        Prints a detailed descriptive statistics table for all numerical columns,
        including percentiles, median, coefficient of variation, skewness, and kurtosis.
        """

        if self.num_cols:
            df_desc = self.dataframe[self.num_cols].describe([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).T
            cv = (self.dataframe[self.num_cols].std() / self.dataframe[self.num_cols].mean().replace(0,
                                                                                                     np.nan) * 100).round(
                3)
            cv = cv.replace([np.inf, -np.inf], np.nan)
            median = self.dataframe[self.num_cols].median()
            skew = self.dataframe[self.num_cols].skew()
            kurtosis = self.dataframe[self.num_cols].kurtosis()
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

    def check_num(self, width_for_graph=1308, height_for_graph=500):

        """
        Displays visual diagnostics and normality test results for each numerical column.

        For each column, generates a combined plot consisting of a Q-Q plot,
        histogram, and box plot. Runs Shapiro-Wilk test for n <= 2500 or
        D'Agostino K² test for n > 2500, and reports the p-value alongside
        the test conclusion.

        Columns flagged as non-normal by the test should be verified visually
        before making a final decision. Use the returned list to construct the
        normality dictionary for num_summary().

        Parameters
        ----------
        width_for_graph : int, optional
            Width of each plot in pixels (default: 1308).
        height_for_graph : int, optional
            Height of each plot in pixels (default: 500).

        Returns
        -------
        list
            List of column names flagged as non-normal by the statistical test.

        Example
        -------
        non_normals = eda.check_num()
        # Visually inspect the plots, then:
        eda.num_summary(result_dict={'age': 'Normal', 'salary': 'Non-normal'})
        """

        if self.num_cols:
            print(f'\n{self.line}')
            print(' Numerical Variable Summary '.center(170))
            print(self.line)
            result = []
            for col in self.num_cols:
                data = self.dataframe[col].dropna()
                fig_combined = make_subplots(rows=2, cols=2, subplot_titles=['Q-Q Plot', 'Histogram', '', 'Box Plot'],
                                             row_heights=[0.7, 0.3],
                                             specs=[[{"rowspan": 2}, {}], [None, {}]])
                (osm, osr), (slope, intercept, r) = scipy.stats.probplot(data)
                fig_combined.add_trace(go.Scatter(x=osm, y=osr, mode='markers',
                                                  marker=dict(color='#4682B4', size=5, opacity=0.7),
                                                  name='Sample'), row=1, col=1)
                fig_combined.add_trace(go.Scatter(x=osm, y=slope * np.array(osm) + intercept,
                                                  mode='lines', line=dict(color='red', width=2),
                                                  name='Normal Reference'), row=1, col=1)
                fig_combined.update_xaxes(title_text='Theoretical Quantiles', row=1, col=1)
                fig_combined.update_yaxes(title_text=f'{col} (Sample Quantiles)', row=1, col=1)
                fig_combined.add_trace(go.Histogram(x=data, opacity=0.7, marker_color='#4682B4'), row=1, col=2)
                fig_combined.add_trace(go.Box(x=data, marker_color='#4682B4'), row=2, col=2)
                fig_combined.update_layout(title_text=f'{col}', title_x=0.5, width=width_for_graph,
                                           height=height_for_graph)
                fig_combined.show(renderer="png", width=width_for_graph, height=height_for_graph, scale=2)
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
                if p_value > self.alpha:
                    print(f"Result: Based on the {test_name} test (p={p_value:.4f} > {self.alpha}), "
                          f"\nthe sample appears Gaussian. However, please verify using the visuals above before making a final decision. "
                          f"\nTo override, pass result_dict={{'col_name': 'Normal/Non-normal'}} to num_summary().")
                else:
                    print(f"Result: Based on the {test_name} test (p={p_value:.4f} ≤ {self.alpha}), "
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

    def num_summary(self, result_dict):

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
            'Column': self.num_cols,
            'Result': [result_dict.get(col, 'Normal') for col in self.num_cols]
        })
        self.num_summary_df = result_df
        return self.num_summary_df

    def check_outlier(self, iqr_th=1.5, z_score_th=3, remove=False, cap=False):

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
        if self.num_cols:
            if self.num_summary_df is None:
                raise RuntimeError("Run num_summary() first.")
            print(f"\n{self.line}")
            print(" Outlier Detection ".center(170))
            print(self.line)
            outlier_report = {}
            cleaned_df = self.dataframe.copy()
            for col in self.num_cols:
                data = cleaned_df[col]
                n = len(data)
                is_normal = self.num_summary_df[self.num_summary_df['Column'] == col]['Result']
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
                self.dataframe = cleaned_df.reset_index(drop=True)
                self.num_summary_df = None
                print("! Outliers removed. Run num_summary() again before further analysis.")
                return outlier_report, self.dataframe
            elif cap:
                self.dataframe = cleaned_df
                self.num_summary_df = None
                print("\n! Outliers capped. Run num_summary() again before further analysis.")
                return outlier_report, self.dataframe
            else:
                return outlier_report
        else:
            raise ValueError('! num_cols is empty')

    def cat_summary(self, width_for_graph=1308, height_for_graph=500):

        """
        Prints value counts and ratios for each categorical column
        and displays a bar chart.

        Parameters
        ----------
        width_for_graph : int, optional
            Width of the plot in pixels (default: 1308).
        height_for_graph : int, optional
            Height of the plot in pixels (default: 500).
        """

        if self.cat_cols:
            print(f'\n{self.line}')
            print(' Categorical Variable Summary '.center(170))
            print(self.line)
            for col in self.cat_cols:
                val_c = pd.DataFrame({
                    'Count': self.dataframe[col].value_counts(),
                    'Ratio': 100 * self.dataframe[col].value_counts() / len(self.dataframe)})
                print(f"\nColumn: {col}")
                print(val_c)
                fig = px.bar(
                    val_c,
                    x=val_c.index,
                    y='Count',
                    text='Count',
                    labels={'x': col, 'Count': 'Count'},
                    title=f'{col}')
                fig.update_layout(width=width_for_graph, height=height_for_graph, title_x=0.5)
                fig.update_traces(marker_color='#2A9D8F', textposition='auto')
                fig.show(renderer="png", width=width_for_graph, height=height_for_graph, scale=2)
            print(self.line)
        else:
            raise ValueError('! cat_cols is empty')

    def target_summary_with_cat(self, width_for_graph=1308, height_for_graph=500):

        """
        Analyzes the relationship between the target column and categorical columns.
        - If target is categorical: Chi-Square test and Cramer's V with heatmap visualization.
        - If target is numerical: ANOVA / Kruskal-Wallis / T-test / Mann-Whitney U test
        with box plot visualization. Normality is assessed independently per group
        using Shapiro-Wilk (n ≤ 2500) or D'Agostino K² (n > 2500).

        Parameters
        ----------
        width_for_graph : int, optional
            Width of the plot in pixels (default: 1308).
        height_for_graph : int, optional
            Height of the plot in pixels (default: 500).
        """

        if self.cat_cols:
            print(f'\n{self.line}')
            print(' Target Analysis with Categorical Variables '.center(170))
            print(self.line)
            if self.target_col in self.cat_cols:
                cols_to_analyze = [col for col in self.cat_cols if col != self.target_col]
                for col in cols_to_analyze:
                    ct = pd.crosstab(self.dataframe[self.target_col].astype(str), self.dataframe[col])
                    ct_pct = pd.crosstab(self.dataframe[self.target_col].astype(str), self.dataframe[col],
                                         normalize='index') * 100
                    chi2, p, dof, expected = scipy.stats.chi2_contingency(ct)
                    fig_combined = make_subplots(rows=1, cols=2,
                                                 subplot_titles=['Countplot', "Crosstab Heatmap Percentage"],
                                                 horizontal_spacing=0.15)
                    teal_palette = ['#355c7d', '#43aa8b', '#c77dff', '#f67280', '#f8961e', '#ef476f', '#00b4d8',
                                    '#9b5de5']
                    fig = px.histogram(self.dataframe, x=self.target_col, color=col, barmode='group',
                                       color_discrete_sequence=teal_palette)
                    for trace in fig.data:
                        fig_combined.add_trace(trace, row=1, col=1)
                    fig_combined.add_trace(
                        go.Heatmap(
                            z=ct_pct.round(1).values,
                            x=ct_pct.columns.tolist(),
                            y=ct_pct.index.tolist(),
                            colorscale='Teal',
                            zmin=0,
                            zmax=100,
                            text=ct_pct.round(1).values,
                            texttemplate='%{text:.1f}%',
                            showscale=True,
                            colorbar=dict(x=1.01, xanchor='left', y=0, yanchor='bottom', thickness=15, len=0.9)), row=1,
                        col=2)
                    fig_combined.update_layout(
                        title_text=f'Relationship: {self.target_col} vs {col}',
                        title_x=0.5,
                        width=width_for_graph,
                        height=height_for_graph,
                        barmode='group',
                        margin=dict(r=120, l=120),
                        legend=dict(x=-0.12, xanchor='left', y=0, yanchor='bottom'))
                    fig_combined.show(renderer="png", width=width_for_graph, height=height_for_graph, scale=2)
                    print(f"\n{self.line}")
                    print(' Chi-Square Test '.center(170))
                    print(self.line)
                    print(f"χ²      = {chi2:.4f}")
                    print(f"p-value = {p:.6f}")
                    print(f"df      = {dof}")
                    print(f"\nExpected Values:")
                    expected_df = pd.DataFrame(expected, index=ct.index, columns=ct.columns).round(1)
                    print(expected_df, '\n')
                    print('Result:')
                    if p < self.alpha:
                        print(
                            f"\n→ H₀ REJECTED: There is a significant relationship between {self.target_col} and {col} (p < 0.05)")
                    else:
                        print("\n→ H₀ ACCEPTED: No statistically significant difference found")
                    print(f"\n{self.line}")
                    print(" Cramer's V ".center(170))
                    print(self.line)
                    n = ct.values.sum()
                    min_dim = min(ct.shape) - 1
                    if min_dim > 0:
                        cramers_v = np.sqrt(chi2 / (n * min_dim))
                    else:
                        cramers_v = 0
                    print(f"V = {cramers_v:.4f}")
                    print('Result:')
                    if cramers_v < 0.1:
                        strength = "Very weak"
                    elif cramers_v < 0.3:
                        strength = "Weak-moderate"
                    elif cramers_v < 0.5:
                        strength = "Moderate-strong"
                    else:
                        strength = "Strong"
                    print(f"\nAssociation strength: {strength}\n")
                    print(self.line)

            elif self.target_col in self.num_cols:
                for col in self.cat_cols:
                    df_pivot = self.dataframe.pivot_table(index=col, values=self.target_col,
                                                          aggfunc=['mean', 'median', 'count'])
                    print(df_pivot)
                    groups = []
                    normality_for_group=[]
                    for i in self.dataframe[col].unique():
                        val = self.dataframe[self.dataframe[col] == i][self.target_col]
                        groups.append(val)
                        if len(val)<=2500:
                            _,p_val_group=scipy.stats.shapiro(val)
                        else:
                            _, p_val_group = scipy.stats.normaltest(val)
                        normality_for_group.append(p_val_group)
                    normality_for_group=np.array(normality_for_group)
                    is_normal=bool((normality_for_group>self.alpha).all())
                    unique_count = len(groups)
                    if unique_count < 2:
                        print(f"Skipping {col}: Not enough groups for comparison.")
                        continue
                    fig = px.box(self.dataframe, x=col, y=self.target_col, color=col,
                                 title=f'{self.target_col} by {col}')
                    fig.update_layout(width=width_for_graph, height=height_for_graph, title_x=0.5)
                    fig.show(renderer="png", width=width_for_graph, height=height_for_graph, scale=2)
                    print(f"\n{self.line}")
                    if unique_count == 2:
                        if is_normal and min(len(g) for g in groups) > 30:
                            print(' Independent T-Test '.center(170))
                            print(self.line)
                            _, p_value = scipy.stats.ttest_ind(*groups, equal_var=False)
                        else:
                            print(' Mann-Whitney U Test '.center(170))
                            print(self.line)
                            _, p_value = scipy.stats.mannwhitneyu(*groups, alternative='two-sided')
                    elif unique_count >= 3:
                        if is_normal and min(len(g) for g in groups) > 30:
                            _, p_levene = scipy.stats.levene(*groups)
                            if p_levene > 0.05:
                                print(' One-Way ANOVA Test '.center(170))
                                print(self.line)
                                _, p_value = scipy.stats.f_oneway(*groups)
                            else:
                                print(' Kruskal-Wallis Test '.center(170))
                                print(self.line)
                                _, p_value = scipy.stats.kruskal(*groups)
                        else:
                            print(' Kruskal-Wallis Test '.center(170))
                            print(self.line)
                            _, p_value = scipy.stats.kruskal(*groups)
                    if p_value < self.alpha:
                        print(f"P-value: {p_value:.6f}.\n H₀ REJECTED: Significant difference found.")
                    else:
                        print(f"P-value: {p_value:.6f}.\n H₀ ACCEPTED: No significant difference.")
                    print(self.line)
        else:
            raise ValueError('! cat_cols is empty')

    def target_summary_with_num(self, width_for_graph=1308, height_for_graph=500):

        """
        Analyzes the relationship between the target column and numerical columns.
        - If target is categorical: T-test / Mann-Whitney U / ANOVA / Kruskal-Wallis
        with box plot visualization. Normality is assessed independently per group
        using Shapiro-Wilk (n ≤ 2500) or D'Agostino K² (n > 2500).
        - If target is numerical: Pearson or Spearman correlation with scatter plot.
        Requires num_summary() to be run first to determine the correlation method.

        Parameters
        ----------
        width_for_graph : int, optional
            Width of the plot in pixels (default: 1308).
        height_for_graph : int, optional
            Height of the plot in pixels (default: 500).
        """

        if self.num_cols:
            print(f'\n{self.line}')
            print(' Target Analysis with Numerical Variables '.center(170))
            print(self.line)
            if self.target_col in self.cat_cols:
                for col in self.num_cols:
                    df_pivot = self.dataframe.pivot_table(index=self.target_col, values=col,
                                                          aggfunc=['mean', 'median', 'count'])
                    print(df_pivot)
                    groups = []
                    normality_for_group = []
                    for i in self.dataframe[self.target_col].unique():
                        val = self.dataframe[self.dataframe[self.target_col] == i][col]
                        groups.append(val)
                        if len(val)<=2500:
                            _,p_val_group=scipy.stats.shapiro(val)
                        else:
                            _, p_val_group = scipy.stats.normaltest(val)
                        normality_for_group.append(p_val_group)
                    normality_for_group=np.array(normality_for_group)
                    is_normal=bool((normality_for_group>self.alpha).all())
                    unique_count = len(groups)
                    if unique_count < 2:
                        print(f"Skipping {col}: Not enough groups for comparison.")
                        continue
                    fig = px.box(self.dataframe, x=self.target_col, y=col, color=self.target_col,
                                 title=f'{col} by {self.target_col}')
                    fig.update_layout(width=width_for_graph, height=height_for_graph, title_x=0.5)
                    fig.show(renderer="png", width=width_for_graph, height=height_for_graph, scale=2)
                    print(f"\n{self.line}")
                    if unique_count == 2:
                        if is_normal and min(len(g) for g in groups) > 30:
                            print(' Independent T-Test '.center(170))
                            print(self.line)
                            _, p_value = scipy.stats.ttest_ind(*groups, equal_var=False)
                        else:
                            print(' Mann-Whitney U Test '.center(170))
                            print(self.line)
                            _, p_value = scipy.stats.mannwhitneyu(*groups, alternative='two-sided')
                    elif unique_count >= 3:
                        if is_normal and min(len(g) for g in groups) > 30:
                            _, p_levene = scipy.stats.levene(*groups)
                            if p_levene > 0.05:
                                print(' One-Way ANOVA Test '.center(170))
                                print(self.line)
                                _, p_value = scipy.stats.f_oneway(*groups)
                            else:
                                print(' Kruskal-Wallis Test '.center(170))
                                print(self.line)
                                _, p_value = scipy.stats.kruskal(*groups)
                        else:
                            print(' Kruskal-Wallis Test '.center(170))
                            print(self.line)
                            _, p_value = scipy.stats.kruskal(*groups)
                    if p_value < self.alpha:
                        print(f"P-value: {p_value:.6f}.\n H₀ REJECTED: Significant difference found.")
                    else:
                        print(f"P-value: {p_value:.6f}.\n H₀ ACCEPTED: No significant difference.")
                    print(self.line)

            elif self.target_col in self.num_cols:
                if self.num_summary_df is None:
                    raise RuntimeError("Run num_summary() first.")
                cols_to_analyze = [col for col in self.num_cols if col != self.target_col]
                for col in cols_to_analyze:
                    is_normal_target = \
                    self.num_summary_df[self.num_summary_df['Column'] == self.target_col]['Result'].values[
                        0] == 'Normal'
                    is_normal_col = self.num_summary_df[self.num_summary_df['Column'] == col]['Result'].values[
                                        0] == 'Normal'
                    if is_normal_target and is_normal_col:
                        method = 'Pearson'
                        corr_value, p_value = scipy.stats.pearsonr(self.dataframe[self.target_col], self.dataframe[col])
                    else:
                        method = 'Spearman'
                        corr_value, p_value = scipy.stats.spearmanr(self.dataframe[self.target_col],
                                                                    self.dataframe[col])
                    print(f'\nTarget: {self.target_col} ←→ {col}')
                    print(f'Method: {method} Correlation')
                    print(f'ρ: {corr_value}')
                    print(f'p-value: {p_value}')
                    if p_value < self.alpha:
                        print(f"→ H₀ REJECTED: Significant correlation (p < {self.alpha})")
                    else:
                        print(f"→ H₀ ACCEPTED: No significant correlation")
                    if abs(corr_value) < 0.1:
                        strength = "Negligible"
                    elif abs(corr_value) < 0.3:
                        strength = "Weak"
                    elif abs(corr_value) < 0.5:
                        strength = "Moderate"
                    elif abs(corr_value) < 0.7:
                        strength = "Strong"
                    else:
                        strength = "Very strong"
                    direction = 'positive' if corr_value > 0 else 'negative'
                    print(f'Strength: {strength} {direction} correlation\n')
                    fig = px.scatter(self.dataframe, x=self.target_col, y=col, trendline='ols',
                                     title=f'{self.target_col} vs {col} | Method: {method} ρ={corr_value:.3f}',
                                     color_discrete_sequence=['#4682B4'], opacity=0.5)
                    fig.update_traces(line=dict(color="red", width=3), selector=dict(mode="lines"))
                    fig.update_layout(title_x=0.5, width=width_for_graph, height=height_for_graph)
                    fig.show(renderer="png", width=width_for_graph, height=height_for_graph, scale=2)
                    print(self.line)
        else:
            raise ValueError('! num_cols is empty ')

    def correlation_heatmap(self, method="spearman", width_for_graph=900, height_for_graph=900):

        """
        Displays a correlation heatmap for all numerical columns.

        Parameters
        ----------
        method : str, optional
            Correlation method: 'spearman', 'pearson', or 'kendall' (default: 'spearman').
        width_for_graph : int, optional
            Width of the plot in pixels (default: 900).
        height_for_graph : int, optional
            Height of the plot in pixels (default: 900).
        """

        if self.num_cols:
            corr = self.dataframe[self.num_cols].corr(method=method)
            fig = px.imshow(
                corr,
                text_auto=True,
                aspect="equal",
                color_continuous_scale="RdBu_r",
                zmin=-1,
                zmax=1,
                width=width_for_graph,
                height=height_for_graph,
                title=f"{method.capitalize()} Correlation Heatmap")
            fig.update_layout(xaxis_tickangle=45, title_x=0.5)
            fig.show(renderer="png", width=width_for_graph, height=height_for_graph, scale=2)
        else:
            raise ValueError('! num_cols is empty')
