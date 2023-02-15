def calculate_bill(account_num, service_code, minutes_used):
    if service_code.lower() == 'r':
        base_price = 10.00
        free_minutes = 50
        price_per_minute = 0.20
    elif service_code.lower() == 'p':
        base_price = 25.00
        if 6 <= int(input("What is the hour? ")) < 18:
            free_minutes = 75
        else:
            free_minutes = 100
        if minutes_used > free_minutes:
            price_per_minute = 0.10
        else:
            price_per_minute = 0.05

    total_minutes = minutes_used - free_minutes
    if total_minutes < 0:
        total_minutes = 0
    total_price = base_price + (total_minutes * price_per_minute)

    print("Account Number:", account_num)
    print("Service Code:", service_code)
    print("Minutes Used:", minutes_used)
    print("Total Amount Due: $", format(total_price, ".2f"))

account_num = input("Enter Account Number: ")
service_code = input("Enter Service Code (R or P): ")
minutes_used = int(input("Enter Minutes Used: "))

calculate_bill(account_num, service_code, minutes_used)
