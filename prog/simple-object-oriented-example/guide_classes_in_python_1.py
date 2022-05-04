"""
    https://www.w3schools.com/python/python_classes.asp
"""

"""
A class with one property
"""
print('---------------------------------')
class Student:
  age = 5

ali = Student()
print("ali.age", ali.age)
ali.age = 10
print("ali.age", ali.age)

print("Student.age", Student.age)


"""
A class with a constructor

All classes have a function called __init__(), 
 which is always executed when the class is being initiated.
"""
print('---------------------------------')
class StudentAli:
  def __init__(self):
    self.name = 'ali' # self is a reference to the current instance of the class
    self.age = 5


Ali = StudentAli()
print("Ali.name: ", Ali.name)

Ahmad = StudentAli()
Ahmad.name = "ahmad"
print("Ahmad.name: ", Ahmad.name)

# print(StudentAli.name) # It is not possible!

"""
A class with a constructor
"""
print('---------------------------------')
class Student:
  def __init__(self, name, age):
    self.name = name
    self.age = age


Ali = Student("ali", 5)
print(Ali.name)


"""
A class with a constructor, and a method
"""
print('---------------------------------')
class Student:
  def __init__(self, name, age):
    self.name = name
    self.age = age

  def say_hello(self):
    print("Hello, my name is ")
    print(self.name)

Ali = Student("ali", 5)
print(Ali.say_hello(), '\n')


"""

"""
print('---------------------------------')
class Student:
  teacher_name = "Parvin" # class attribute / property

  def __init__(self, name, age):
    self.name = name # instance attribute / property
    self.age = age

  def say_hello(self):
    print("Hello, my name is ", self.name)
    print('--------')


Ali = Student("ali", 5)
print(Ali.say_hello())
print('Teacher name: ', Ali.teacher_name)

Jafar = Student("jafar", 5)
print(Jafar.say_hello())
print('Teacher name: ', Jafar.teacher_name, '\n')
