
num = input("Enter a large integer: ")
num = num[::-1] 
result = ""
for i in range(0, len(num), 3):
    result += num[i:i+3][::-1] + "." 
result = result[::-1].strip(".")
print("result: ", result)
