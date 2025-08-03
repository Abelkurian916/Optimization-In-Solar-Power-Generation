# -*- coding: utf-8 -*-
"""
Created on Sun Mar 23 23:21:01 2025

@author: abelk
"""

import pandas as pd
import numpy as np

df = pd.read_csv("C:\\projects\\main project (solar energy)\\python + sql + EDA\\Transformed_Inverter_Data - Copy.csv")
df2 = pd.read_csv("C:\\projects\\main project (solar energy)\\python + sql + EDA\\Transformed_WMS_REPORT1.csv")


#####--ydata_profiling--####

import ydata_profiling
#Pandas Profiling (YData Profiling) of inverter dataset
profile = df.profile_report(title="Inverter Data - Pandas Profiling Report")
profile.to_file("pandas_profiling_report.html")
print("Pandas Profiling Report saved as 'pandas_profiling_report.html'")

#Pandas Profiling (YData Profiling) of wms report
profile = df2.profile_report(title="WMS Report - Pandas Profiling Report")
profile.to_file("pandas_profiling_report_wms.html")
print("Pandas Profiling Report saved as 'pandas_profiling_report_wms.html'")


#####--sweetviz--####

import sweetviz as sv
#Sweetviz Report of inverter data set
sweet_report = sv.analyze(df)
sweet_report.show_html("sweetviz_report_invert.html")
print(" Sweetviz Report saved as 'sweetviz_report_invert.html'")


#Sweetviz Report of wms report
sweet_report = sv.analyze(df2)
sweet_report.show_html("sweetviz_report_wms.html")
print(" Sweetviz Report saved as 'sweetviz_report_wms.html'")




#####--D-Tale--####

import dtale
#D-Tale report of inverter

dtale.show(df)
#D-Tale report of wms report
dtale.show(df2)

#####--Dataprep--####

from dataprep.eda import create_report
df = pd.read_csv("your_dataset.csv")

#Dataprep of inverter
report = create_report(df2)
report.show_browser()
report.save("dataprep_eda_report_wms.html")


#Dateprep of wms report
report = create_report(df)
report.show_browser()
report.save("dataprep_eda_report_wms.html")
