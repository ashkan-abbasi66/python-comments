# NOTE: see main.ipynb
#
#
#
#
#
#





# from folder.file import a class

from staff import Staff

officeStaff1 = Staff('Basic', 'Ali', 0) # We do not need to pass 'self'

#change variable position
officeStaff1.position = 'Manager'

print(officeStaff1.calculatePay())

print(officeStaff1) # Python will call the __str__ method

# we may accidentally change the property of an instance to an incorrect value
# Let's change 'position' from an instance variable to a property
print("\n")
from staff_2 import Staff_2
officeStaff2=Staff_2("Manager","Mohammad",0)
officeStaff2.position = 'Basic' # setter method
print(officeStaff2)
officeStaff2.position = 'WRONG VALUE'
print(officeStaff2)
# although we created properties, one can still modify '_position'
officeStaff2._position='WRONG VALUE'
print(officeStaff2)

# Name mangling
# mangle variable's name to make it harder to access
print("\n")
from staff_3 import Staff_3
officeStaff3=Staff_3("Manager","Ashkan",0)
officeStaff3._Staff_3__position='WRONG VALUE'
print(officeStaff3)

## what is 'self' keyword?
# self essentially represents an instance of the class.
from programmingStaff import ProgStaff
peter = ProgStaff(2500)
john = ProgStaff(2500)
print(peter.salary)
print(ProgStaff.companyName)

print('----- call an instance method: -----')
peter.printInfo()
ProgStaff.printInfo(peter)

# There are two other method types which are used rarely.
# Class method
# A class method is a method that has a class object (instead of self) as the first
# parameter. 'cls' is commonly used to represent that class object.
ProgStaff.classMeth()
peter.classMeth()

# Static method
ProgStaff.staticMeth()
john.staticMeth()

# Inheritance
# import staffs  ==> arman=staffs.ManagementStaff('Arman',0,1000,0)
from staffs import *
arman=ManagementStaff('Arman',0,1000,0)
arman.calculatePay()
print(arman)

afshin=BasicStaff('Afshin',800)
print(afshin)

# special function
# __init__, __str__
# __add__, __sub__, __mul__ and __div__ can be used for overloading operators (+,-,*,/)
total=afshin+arman
print(total)

# Python Built-in Functions for Objects
print(isinstance(afshin, Staff_2))
print(isinstance(afshin,BasicStaff))
print(isinstance(afshin,ManagementStaff))
