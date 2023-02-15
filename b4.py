# Ask user for input
num_apartments = int(input("Enter the number of apartments: "))
rental_price = int(input("Enter the monthly rental price of an apartment: "))
rent_increase = int(input("Enter the increase in rent that leads to an apartment being vacated: "))
maintenance_cost = int(input("Enter the amount of money required to maintain an apartment: "))

num_rented = []
num_rented.append(num_apartments) # Initially, all apartments are rented

# Loop through the number of apartments and append each value to num_rented
while num_apartments > 0:
	num_apartments -= 1
	num_rented.append(num_apartments)

profits = []

# Loop through each value in num_rented to calculate the profit for each scenario
for i in range(len(num_rented)):
	profit = (num_rented[i] * (rental_price + (rent_increase * i))) - (num_rented[i] * maintenance_cost)
	profits.append(profit)

# Get the index of the maximum profit value in profits
max_profit_index = profits.index(max(profits))

# Print the number of rented apartments that leads to the maximum profit
print("To maximize profit, rent", num_rented[max_profit_index], "apartments.")
