def calculate_population_growth(a_pop, a_growth_rate, b_pop, b_growth_rate):
    years = 0
    while a_pop < b_pop:
        a_pop *= 1 + a_growth_rate
        b_pop *= 1 + b_growth_rate
        years += 1
    return years, int(a_pop), int(b_pop)

a_pop = int(input("Enter population of city A: "))
a_growth_rate = float(input("Enter growth rate of city A (in decimal): "))
b_pop = int(input("Enter population of city B: "))
b_growth_rate = float(input("Enter growth rate of city B (in decimal): "))

years, a_pop, b_pop = calculate_population_growth(a_pop, a_growth_rate, b_pop, b_growth_rate)

print(f"It will take {years} years for city A to surpass or equal city B's population.")
print(f"At that time, the population of city A will be {a_pop} and the population of city B will be {b_pop}.")
