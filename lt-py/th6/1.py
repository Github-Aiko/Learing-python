feet = float(input("Nhập chiều dài tính bằng feet: "))
options = ["inch", "yard", "dặm", "milimét", "centimet", "mét", "km"]
print("Chọn đơn vị muốn chuyển đổi:")
for i, option in enumerate(options):
    print(f"{i+1}. {option}")
choice = int(input("Nhập lựa chọn của bạn: "))
result = ""
try:
    result = eval(f"feet*{[12, 1/3, 1/5280, 304.8, 30.48, 0.3048, 0.0003048][choice-1]}")
    print(f"{feet} feet = {result} {options[choice-1]}")
except:
    print("Lựa chọn không hợp lệ!")
