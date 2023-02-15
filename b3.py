# Prompt user to input loan amount, annual interest rate, and monthly payment
loan_amount = float(input("Enter loan amount: "))
annual_interest_rate = float(input("Enter annual interest rate (%): "))
monthly_payment = float(input("Enter monthly payment: "))

# Convert annual interest rate to monthly interest rate
monthly_interest_rate = annual_interest_rate / 1200

# Initialize variables
months = 0
loan_amount_remaining = loan_amount

# Loop until remaining balance is 0 or less
while loan_amount_remaining > 0:
    # Calculate interest for the month
    interest = loan_amount_remaining * monthly_interest_rate
    
    # If monthly payment is less than the interest, notify the user and break out of loop
    if monthly_payment < interest:
        print("Monthly payment is too low to pay off loan. Increase the payment amount.")
        break
    
    # Calculate payment for the month
    payment = min(loan_amount_remaining + interest, monthly_payment)
    
    # Subtract payment from remaining balance
    loan_amount_remaining -= payment - interest
    
    # Increment the number of months
    months += 1

# Print the number of months it takes to pay off the loan
print("It takes", months, "months to pay off the loan.")
