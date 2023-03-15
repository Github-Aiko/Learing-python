import random

def one_time_pad_encrypt(message):
    # Chuyển đổi chuỗi thành chữ thường
    message = message.lower()
    # Tạo một list chứa các số ngẫu nhiên từ 1 đến 26 cho mỗi ký tự trong tin nhắn
    random_numbers = [random.randint(1, 26) for _ in range(len(message))]
    # Khởi tạo chuỗi mã hóa
    encrypted_message = ""
    # Duyệt qua từng ký tự trong tin nhắn
    for i in range(len(message)):
        # Nếu ký tự là một khoảng trắng hoặc dấu chấm câu thì giữ nguyên
        if message[i] == " " or message[i] in ",.!?":
            encrypted_message += message[i]
        else:
            # Chuyển đổi ký tự thành mã ASCII và trừ đi 97 để chuyển đổi thành số từ 0 đến 25
            ascii_code = ord(message[i]) - 97
            # Thực hiện phép mã hóa bằng cách cộng thêm số ngẫu nhiên tương ứng
            new_ascii_code = (ascii_code + random_numbers[i]) % 26
            # Chuyển đổi số vừa mã hóa thành ký tự và thêm vào chuỗi mã hóa
            encrypted_message += chr(new_ascii_code + 97)
    return encrypted_message

# Yêu cầu người dùng nhập tin nhắn và thực hiện mã hóa
message = input("Nhập tin nhắn cần mã hóa: ")
encrypted_message = one_time_pad_encrypt(message)
# In ra tin nhắn đã được mã hóa
print("Tin nhắn đã được mã hóa: ", encrypted_message)
