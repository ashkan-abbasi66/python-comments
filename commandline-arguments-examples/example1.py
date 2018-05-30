# command-line arguments
# How to parse command line arguments

# keywords:
# sys.argv
# string.startswith
# string.split


# Usage:
#   python example1.py --help
#   python example1.py model=Hello task=greeting number:10

# Note: Can't run python on the Terminal window in PyCharm?
#   File>Settings>Tools>Terminal, then
#   add "cmd.exe "/K" C:\ProgramData\Anaconda3\Scripts\activate.bat"
#

import sys

def process_this_script_args(arglist):
    for a in arglist:
        if len(arglist)==1 or a.startswith(("--help", "-h")):
            print("Run this command as an example:")
            print("\tpython example1.py model=Hello task=greeting number:10")
            return

    print("Relative path to this script: ", arglist[0])
    print("# of args: ", len(arglist))
    print("List of arguments: " , str(arglist))

    for a in arglist:
        if a.startswith("model"):
            print("model name=",a.split("=")[1])
        if a.startswith("task="):
            print("The {0} is {1}".format(a.split("=")[0],a.split("=")[1]))
        if a.startswith("number:"):
            t=a.split(":")
            argname=t[0]
            num=int(t[1])*100
            print("Input ''{0}'' multiplied by 100 is {1}".format(argname,num))


# sys.argv is a list which contains the command-line arguments.
process_this_script_args(sys.argv)

str = "this is string example."
print(str.startswith( 'example', 15, 22 ))# determine start and end positions

text = "programming is easy"
result = text.startswith(('python', 'programming'))

