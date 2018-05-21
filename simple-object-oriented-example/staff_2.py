# reference: book_Python (2nd Edition)_ Learn Python in One Day.pdf - page: 68

class Staff_2:
    # Look at how to add a property for the _position variable
    def __init__(self,pPosition, pName, pPay):
        self._position=pPosition # '_' indicates that do not access directly.
        self.name=pName
        self.pay=pPay
        print("Creating Staff_2 object")

    # @property is a 'decorator' which alters the functionality of a method
    # it changes a method to a property. Instead of obj.position(), use obj.position
    @property
    def position(self):
        print("Get Method")
        return self._position

    @position.setter
    def position(self, value):
        if value=='Manager' or value=='Basic':
            self._position=value
        else:
            print('Position is invalid. No changes made.')

    def __str__(self):
        return "Position=%s, Name=%s, Pay=%d"%(self._position, self.name, self.pay)

    def calculatePay(self):
        prompt="\nEnter # of hours worked for %s: "%(self.name)
        hours=input(prompt)
        prompt = 'Enter the hourly rate for %s: ' % (self.name)
        hourlyRate=input(prompt)
        self.pay = int(hours) * int(hourlyRate)
        return self.pay

    def __add__(self, other):
        return self.pay+other.pay