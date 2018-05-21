# reference: book_Python (2nd Edition)_ Learn Python in One Day.pdf - page: 68

class Staff:
    # special method
    def __init__(self,pPosition, pName, pPay):
        self.position=pPosition #self.position is an instance variable
        self.name=pName
        self.pay=pPay
        print("Creating Staff object")

    # special method
    def __str__(self):
        return "Position=%s, Name=%s, Pay=%d"%(self.position, self.name, self.pay)

    def calculatePay(self):
        prompt="\nEnter # of hours worked for %s: "%(self.name)
        hours=input(prompt)
        prompt = 'Enter the hourly rate for %s: ' % (self.name)
        hourlyRate=input(prompt)
        self.pay = int(hours) * int(hourlyRate)
        return self.pay