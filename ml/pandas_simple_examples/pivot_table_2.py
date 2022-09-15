# -*- coding: utf-8 -*-
"""
more on Pivot table

"""

import pandas as pd

employees = {
    'Name of Employee': ['Jon','Mark','Tina','Maria','Bill','Jon','Mark','Tina','Maria','Bill','Jon','Mark','Tina','Maria','Bill','Jon','Mark','Tina','Maria','Bill'],
    'Sales': [1000,300,400,500,800,1000,500,700,50,60,1000,900,750,200,300,1000,900,250,750,50],
    'Quarter': [1,1,1,1,1,2,2,2,2,2,3,3,3,3,3,4,4,4,4,4],
    'Country': ['US','Japan','Brazil','UK','US','Brazil','Japan','Brazil','US','US','US','Japan','Brazil','UK','Brazil','Japan','Japan','Brazil','UK','US']
    }

df = pd.DataFrame(employees, columns= ['Name of Employee','Sales','Quarter','Country'])

print (df,'\n\n')



# Total sales by country

pivot = df.pivot_table(index = ['Country'], values = ['Sales'], aggfunc=('sum'))

print(pivot,'\n\n')



# Sales by both employee and country

pivot = df.pivot_table(index = ['Name of Employee','Country'], values = ['Sales'], aggfunc=('sum'))

print(pivot,'\n\n')

pivot = df.pivot_table(index = ['Country','Name of Employee'], values = ['Sales'], aggfunc=('sum'))
print(pivot,'\n\n')



# Maximum individual sale by country

pivot = df.pivot_table(index=['Country'], values=['Sales'], aggfunc='max')

print (pivot,'\n\n')


# Mean, median and minimum sales by country

pivot = df.pivot_table(index=['Country'], values=['Sales'], aggfunc={'median','mean','min'})

print (pivot, '\n\n')
