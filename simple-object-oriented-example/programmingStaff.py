class ProgStaff:
    companyName = 'ProgrammingLab' # class variable

    def __init__(self, pSalary):
        self.salary = pSalary # instance variable

    # 'self' parameter indicates that this is an instance method
    def printInfo(self):
        print("Company name is", ProgStaff.companyName) # access to a class var.
        print("Salary is", self.salary) # access to an instance var.

    @classmethod
    def classMeth(cls): # cls or anything else. however, cls is a common name.
        print("This is a class method.")
        print(" we can usu. access to a class variable using \'cls\'")
        print("cls.companyName=%s"%cls.companyName)

    @staticmethod
    def staticMeth():
        print("This is an static method")