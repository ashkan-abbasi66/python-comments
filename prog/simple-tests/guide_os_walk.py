import os


home = os.path.expanduser("~")
print("Your home directory is:")
print(home, '\n')

print("Your current directory is:")
print("Relative:", os.path.curdir)
print("Absolute:", os.getcwd(), '\n')

# Given a path, let's split it
fpath = './guide_os_walk.py'
print("file path:", fpath)
print("file name:", os.path.basename(fpath))
print("dir. path:", os.path.dirname(fpath), '\n')

# Let's make it absolute
fpath_abs = os.path.abspath(fpath)
print("dir. path:", os.path.dirname(fpath_abs), '\n')


"""
os.listdir
"""
print('======> ')
# data_dir = os.path.abspath('./data')
# E:\POSTDOC\GITHUB\python-comments\prog\simple-tests\data
# data_dir = 'E:\\POSTDOC\\GITHUB\\python-comments\\prog\\simple-tests\\data'
data_dir = 'E:/POSTDOC/GITHUB/python-comments/prog/simple-tests/data'
print('\n')
print('a\tb')
print('Here\'s my book')

# data_dir = './data'

file_list = os.listdir(data_dir)
print(file_list)
print()


"""
os.walk
"""

for root, dir_names, file_names in os.walk(data_dir):
    print(root, ", ", dir_names, ", ", file_names)
print()


# Exercise: Create a full path for each file
# for root, dir_names, file_names in os.walk(data_dir):
#     for file_name in file_names:
#         print(os.path.join(root, file_name))
