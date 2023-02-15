# Ask user for input
num_apartments = int(input("Enter the number of apartments: "))
rental_price = int(input("Enter the monthly rental price of an apartment: "))
rent_increase = int(input("Enter the increase in rent that leads to an apartment being vacated: "))
maintenance_cost = int(input("Enter the amount of money required to maintain an apartment: "))

profits = []

# Loop through the number of apartments and calculate the profit for each scenario
for num_rented in range(num_apartments, -1, -1):
	profit = (num_rented * (rental_price + (rent_increase * (num_apartments - num_rented)))) - (num_rented * maintenance_cost)
	profits.append(profit)

# Get the index of the maximum profit value in profits
max_profit_index = profits.index(max(profits))

# Print the number of rented apartments that leads to the maximum profit
print("To maximize profit, rent", num_apartments - max_profit_index, "apartments.")
