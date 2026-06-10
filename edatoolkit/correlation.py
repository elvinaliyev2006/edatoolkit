import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 


class Correlation:
    
    def __init__(self):
        pass


    def correlation_heatmap(self, dataframe, num_cols, method="spearman",  width_for_graph=9, height_for_graph=9):

            """
            Displays a correlation heatmap for all numerical columns.

            Parameters
            ----------
            method : str, optional
                Correlation method: 'spearman', 'pearson', or 'kendall' (default: 'spearman').
            plot : bool, optional
                If True, displays the correlation heatmap (default: False).
            width_for_graph : int, optional
                Width of the figure in inches (default: 9).
            height_for_graph : int, optional
                Height of the figure in inches (default: 9).
            """

            if num_cols:
                corr = dataframe[num_cols].corr(method=method)

                
                fig, ax = plt.subplots(figsize=(width_for_graph, height_for_graph))
                fig.patch.set_facecolor('white')
                ax.set_facecolor('white')
                ax.grid(False)
                sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu_r',
                                vmin=-1, vmax=1, ax=ax,
                                linewidths=0.5, linecolor='white',
                                annot_kws={'size': 9},
                                square=True)
                ax.set_title(f'{method.capitalize()} Correlation Heatmap', fontsize=13, fontweight='bold')
                plt.xticks(rotation=45, ha='right')
                plt.yticks(rotation=0)
                plt.tight_layout()
                plt.show()
            else:
                raise ValueError('! num_cols is empty')