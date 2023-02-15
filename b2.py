def get_valid_input(prompt):
    while True:
        value = input(prompt)
        try:
            value = int(value)
            if value <= 0:
                print("Please enter a positive integer.")
            else:
                return value
        except ValueError:
            print("Please enter a valid integer.")

A_population = get_valid_input("Please enter the population of City A: ")
growth_rate_A = get_valid_input("Please enter the growth rate of City A: ")
B_population = get_valid_input("Please enter the population of City B: ")
growth_rate_B = get_valid_input("Please enter the growth rate of City B: ")

num_years = 0
while A_population < B_population:
    A_population *= (1 + growth_rate_A / 100)
    B_population *= (1 + growth_rate_B / 100)
    num_years += 1

print(f"Population Growth Simulation Results:\n"
      f"{'City':<10}{'Population':>15}\n"
      f"{'A':<10}{round(A_population):>15,}\n"
      f"{'B':<10}{round(B_population):>15,}\n"
      f"\n"
      f"City A will have a larger or equal population to City B in {num_years:,} years.")
