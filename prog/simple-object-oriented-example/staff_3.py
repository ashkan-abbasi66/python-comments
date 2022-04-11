# reference: book_Python (2nd Edition)_ Learn Python in One Day.pdf - page: 68

class Staff_3:
    def __init__(self,pPosition, pName, pPay):
        self.___position=pPosition # '__' it makes harder for others to access this var
        self.name=pName
        self.pay=pPay
        print("Creating Staff_3 object")

    @property
    def position(self):
        print("Get Method")
        return self.__position

    @position.setter
    def position(self, value):
        if value=='Manager' or value=='Basic':
            self.__position=value
        else:
            print('Position is invalid. No changes made.')

    def __str__(self):
        return "Position=%s, Name=%s, Pay=%d"%(self.__position, self.name, self.pay)

    def calculatePay(self):
        prompt="\nEnter # of hours worked for %s: "%(self.name)
        hours=input(prompt)
        prompt = 'Enter the hourly rate for %s: ' % (self.name)
        hourlyRate=input(prompt)
        self.pay = int(hours) * int(hourlyRate)
        return self.pay