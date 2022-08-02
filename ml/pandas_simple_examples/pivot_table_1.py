# -*- coding: utf-8 -*-
"""
Dataframe in Pandas
Create a simple pivot table.
"""

import pandas as pd

employees = {
    'Name of Employee': ['Jon', 'Mark','Tina','Maria','Bill','Jon','Mark','Tina','Maria','Bill','Jon','Mark','Tina','Maria','Bill','Jon','Mark','Tina','Maria','Bill'],
    'Sales': [1000,300,400,500,800,1000,500,700,50,60,1000,900,750,200,300,1000,900,250,750,50],
    'Quarter': [1,1,1,1,1,2,2,2,2,2,3,3,3,3,3,4,4,4,4,4],
    'Country': ['US','Japan','Brazil','UK','US','Brazil','Japan','Brazil','US','US','US','Japan','Brazil','UK','Brazil','Japan','Japan','Brazil','UK','US']
    }

df = pd.DataFrame(employees, columns= ['Name of Employee','Sales','Quarter','Country'])

print (df,'\n\n')


# Once you have your DataFrame ready, you’ll be able to "pivot" your data.
# pivot means:
#   to turn or twist
#   to change something so that they are different to what they were before

# In Excel
# A Pivot Table is used to summarise, sort, reorganise, group, count,
# total or average data stored in a table. It allows us to transform
# columns into rows and rows into columns. It allows grouping by any
# field (column), and using advanced calculations on them.


# Total sales per employee

pivot = df.pivot_table(index=['Name of Employee'], values=['Sales'], aggfunc='sum')

print (pivot,'\n\n')

