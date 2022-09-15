# -*- coding: utf-8 -*-
"""
read from data file and create a dataframe

"""

import pandas as pd

df = pd.read_csv('./data.csv')

print (df,'\n\n')


# Total sales per employee

pivot = df.pivot_table(index=['Name of Employee'], values=['Sales'], aggfunc='sum')

print (pivot,'\n\n')

