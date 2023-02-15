def calculate_loan_payment_months(principal, annual_interest_rate, monthly_payment):
    monthly_interest_rate = annual_interest_rate / 1200
    months = round(-1 * math.log(1 - ((monthly_interest_rate * principal) / monthly_payment), 10) / math.log(1 + monthly_interest_rate), 0)
    return int(months)

import math

principal = float(input("Enter the principal amount: "))
annual_interest_rate = float(input("Enter the annual interest rate: "))
monthly_payment = float(input("Enter the monthly payment amount: "))

months = calculate_loan_payment_months(principal, annual_interest_rate, monthly_payment)

print("It will take", months, "months to pay off the loan.")
