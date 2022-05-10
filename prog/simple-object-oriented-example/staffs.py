from staff_2 import Staff_2

class ManagementStaff(Staff_2): # ManagementStaff is a subclass of Staff_2
    def __init__(self, pName, pPay, pAllowance, pBonus):
        # Call a method in the base class using 'super()' method
        # super() returns a temporary object that allows reference to a parent class by the keyword super.
        super().__init__('Manager',pName,pPay)
        # Staff_2.__init__(self,'Manager',pName,pPay)
        # Python 2 syntax:
        # super(ManagementStaff, self).__init__('Manager', pName, pPay)
        self.allowance=pAllowance
        self.bonus=pBonus

        # overriding
    def calculatePay(self):
        basicPay = super().calculatePay()
        self.pay = basicPay + self.allowance
        return self.pay

class BasicStaff(Staff_2):
    def __init__(self, pName, pPay):
        super().__init__('Basic', pName, pPay)