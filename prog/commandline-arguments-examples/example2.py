# Usage:
#   python example2.py  # ArgumentParser automatically generates a good error message.
#   python example2.py -h   # display help messages
#   python example2.py -n ashkan
#   python example2.py -n ashkan -2nd-name ali

# keywords:
# argparse.ArgumentParser()
# add_argument
# vars(ap.parse_args())

import argparse # makes it easy to write user-friendly command-line interfaces.
# official example can be found here: https://docs.python.org/3.3/library/argparse.html

ap = argparse.ArgumentParser()
ap.add_argument("-n", # Or ("-n", "--name",... => In this case, the key should be "name"
                required=True,# a required argument
                help="name of the user")
ap.add_argument("-a","-2nd-name",
                required=False,
                help="name of the second user")
args = vars(ap.parse_args()) # this is a dictionary object - <class 'dict'>
# obj.parse_args() parse the command line arguments.
# vars on this object turns the parsed arguments into a Python dictionary

print('dictionary elements:')
print(args)

print("Hi {}, Nice to meet you!".format(args["n"]))
if "a" in args:
    print("Hi {}".format(args["a"]))

# [Ref: https://www.pyimagesearch.com/2018/03/12/python-argparse-command-line-arguments/]
# see more:
#   https://stackoverflow.com/questions/1009860/how-to-read-process-command-line-arguments?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa