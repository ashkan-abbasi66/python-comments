"""
Formatted Output
https://python-course.eu/python-tutorial/formatted-output.php

"""

import os

# dirs = os.listdir('E:\POSTDOC\GITHUB\python-comments\prog\simple-tests\data') # absolute path
dirs = os.listdir('./data') # relative path
# dirs = os.listdir('../')

for e in dirs:
    print(e)
    if e == 'baby.png':
        os.rename('data/dir0_noJPG/baby.png', './data/baby2.png')

print('line 1') # \n
print('line 2')
print('line 3', end = '')
print('line 4')
print(10)
print('Your score is %d.'%10) # %d: integer; %f: float; %g: float

for file_number in range(5000,5005):
    filename = 'myfile{:0>9}'.format(file_number)
    print(filename)

print('{:0>3}'.format(2))

variable_name = 10.258975
print(f'{variable_name:.3f}')

print('%.2f'%(10.258975))
print('%.10f'%(10.25897545))
print('%g'%(10.258975))


print(os.getcwd()) # get current working directory.

