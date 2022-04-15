principal = 500000.0
payment = 3000
total_paid = 0.0

while principal > 0:
    principal = principal - payment
    total_paid = total_paid + payment
    print(principal)

print('Total paid', total_paid)
