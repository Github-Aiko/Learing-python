# Enter account number and service code
account_number = input("Enter account number: ")
service_code = input("Enter service code (r for basic service; p for premium service): ")

# Enter the number of minutes of service used
used_minutes = int(input("Enter the number of minutes of service used: "))

# Calculate tariff
if service_code == "r" or service_code == "R":
    # Basic service
    base_price = 10.00
    free_minutes = 50
    additional_minutes_price = 0.20
    if used_minutes <= free_minutes:
        total_price = base_price
    else:
        total_price = base_price + (used_minutes - free_minutes) * additional_minutes_price
else:
    # Premium service
    base_price = 25.00
    day_free_minutes = 75
    night_free_minutes = 100
    day_additional_minutes_price = 0.10
    night_additional_minutes_price = 0.05
    day_minutes = int(input("Enter the number of minutes of service used during the day: "))
    night_minutes = int(input("Enter the number of minutes of service used at night: "))
    if used_minutes <= day_free_minutes + night_free_minutes:
        total_price = base_price
    else:
        day_minutes_price = max(0, day_minutes - day_free_minutes) * day_additional_minutes_price
        night_minutes_price = max(0, night_minutes - night_free_minutes) * night_additional_minutes_price
        total_price = base_price + day_minutes_price + night_minutes_price

# Print invoice
print("Account number: ", account_number)
if service_code == "r" or service_code == "R":
    print("Service type: Basic service")
else:
    print("Service type: Premium service")
print("Number of minutes of service used: ", used_minutes)
print("Amount to be paid: $", format(total_price, ".2f"))