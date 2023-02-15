def calculate_profit(num_apartments, rent_price, increase_price, maintenance_cost):
    max_profit = 0
    max_num_apartments = 0

    for i in range(num_apartments):
        current_profit = (rent_price + increase_price * i) * (num_apartments - i) - maintenance_cost * num_apartments
        if current_profit > max_profit:
            max_profit = current_profit
            max_num_apartments = num_apartments - i

    return max_num_apartments

# Prompt user for input
num_apartments = int(input("Enter the number of apartments: "))
rent_price = int(input("Enter the rent price per month: "))
increase_price = int(input("Enter the price increase that leads to one vacant apartment: "))
maintenance_cost = int(input("Enter the maintenance cost per apartment per month: "))

max_num = calculate_profit(num_apartments, rent_price, increase_price, maintenance_cost)

print("The maximum number of apartments to rent for maximum profit is:", max_num)
