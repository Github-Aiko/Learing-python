# Taking input from user 
A_population = int(input("Please enter the population of City A: ")) 
growth_rate_A = int(input("Please enter the growth rate of City A: ")) 
B_population = int(input("Please enter the population of City B: ")) 
growth_rate_B = int(input("Please enter the growth rate of City B: ")) 
  
# Calculating the number of years 
num_years = 0
while A_population < B_population: 
    A_population += A_population * growth_rate_A / 100 
    B_population += B_population * growth_rate_B / 100 
    num_years += 1
  
# Printing the result 
print("Number of years after which City A population will be greater than or equal to City B population is", num_years) 
print("At this time, City A population is", round(A_population), "and City B population is", round(B_population))