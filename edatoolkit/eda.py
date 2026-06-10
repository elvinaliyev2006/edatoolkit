from .inspector import Inspector
from .normality import NormalityAnalyzer
from .outliers import OutlierAnalyzer
from .categorical import CategoricalAnalyzer
from .target import TargetAnalyzer
from .correlation import Correlation


class EDA:
    def __init__(self, dataframe, target_col, cat_th=10, car_th=20, alpha=0.05):
        self.dataframe=dataframe
        self.target_col=target_col
        self.car_th=car_th
        self.cat_th=cat_th
        self.alpha=alpha
        self.inspector=Inspector()
        self.normality=NormalityAnalyzer()
        self.outliers=OutlierAnalyzer()
        self.categorical=CategoricalAnalyzer()
        self.target=TargetAnalyzer()
        self.correlation=Correlation()
        self.cat_cols, self.num_cols, self.num_but_cat, self.cat_but_car = self.inspector.get_columns_types(
                                                                                        dataframe=self.dataframe, car_th=self.car_th, cat_th=self.cat_th)
        self.num_summary_df = None
        self.outlier_report = None
    
    def update_dataframe(self,new_df):
        """
        Updates the dataframe and automatically recalculates all column types.

        Parameters
        ----------
        new_df : pd.DataFrame
            The new dataset to be assigned to the class instance.
        """
        self.dataframe=new_df
        self.cat_cols, self.num_cols, self.num_but_cat, self.cat_but_car = self.inspector.get_columns_types(dataframe=self.dataframe, 
                                                                                                            car_th=self.car_th, cat_th=self.cat_th)
        self.num_summary_df = None
        print("Dataframe and column types have been successfully updated.")  

    def check_dataframe(self,n=5):
        self.inspector.check_dataframe(dataframe=self.dataframe, n=n)
    
    def get_columns_types(self):
        return self.inspector.get_columns_types(dataframe=self.dataframe,car_th=self.car_th,cat_th=self.cat_th)

    def descriptive_analysis(self):
        self.normality.descriptive_analysis(dataframe=self.dataframe,num_cols=self.num_cols)
    
    def check_num(self, plot=False, width_for_graph=15, height_for_graph=5):
        self.normality.check_num(dataframe=self.dataframe,num_cols=self.num_cols,alpha=self.alpha,width_for_graph=width_for_graph,
                                 height_for_graph=height_for_graph)
    
    def num_summary(self,result_dict):
        self.num_summary_df=self.normality.num_summary(num_cols=self.num_cols,result_dict=result_dict)
        return self.num_summary_df
    
    def check_outlier(self,iqr_th=1.5, z_score_th=3, remove=False, cap=False):
        result=self.outliers.check_outlier(dataframe=self.dataframe,num_cols=self.num_cols,num_summary_df=self.num_summary_df,
                                           iqr_th=iqr_th,z_score_th=z_score_th,remove=remove,cap=cap)
        if isinstance(result, tuple):
            outlier_report, self.dataframe = result
            self.num_summary_df = None
            return outlier_report, self.dataframe
        return result
    
    def cat_summary(self,plot=False,width_for_graph=13, height_for_graph=5 ):
        self.categorical.cat_summary(dataframe=self.dataframe,cat_cols=self.cat_cols,plot=plot,
                                     width_for_graph=width_for_graph,height_for_graph=height_for_graph)

    def target_summary_with_cat(self, plot=False, width_for_graph=13, height_for_graph=5):
        self.target.target_summary_with_cat(dataframe=self.dataframe,cat_cols=self.cat_cols,num_cols=self.num_cols,
                                            target_col=self.target_col,alpha=self.alpha,plot=plot,width_for_graph=width_for_graph,height_for_graph=height_for_graph)
        
    def target_summary_with_num(self, plot=False, width_for_graph=13, height_for_graph=5):
        self.target.target_summary_with_num(dataframe=self.dataframe,num_cols=self.num_cols,cat_cols=self.cat_cols,
                                            target_col=self.target_col,num_summary_df=self.num_summary_df,alpha=self.alpha,plot=plot,
                                            width_for_graph=width_for_graph,height_for_graph=height_for_graph)

    def correlation_heatmap(self, method="spearman",  width_for_graph=9, height_for_graph=9):
        self.correlation.correlation_heatmap(dataframe=self.dataframe,num_cols=self.num_cols,method=method,
                                            width_for_graph=width_for_graph,height_for_graph=height_for_graph )