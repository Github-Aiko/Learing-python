def one_time_pad_decrypt(encrypted_message, random_numbers):
    # Chuyển đổi chuỗi mã hóa thành chữ thường
    encrypted_message = encrypted_message.lower()
    # Khởi tạo chuỗi giải mã
    decrypted_message = ""
    # Duyệt qua từng ký tự trong chuỗi mã hóa
    for i in range(len(encrypted_message)):
        # Nếu ký tự là một khoảng trắng hoặc dấu chấm câu thì giữ nguyên
        if encrypted_message[i] == " " or encrypted_message[i] in ",.!?":
            decrypted_message += encrypted_message[i]
        else:
            # Chuyển đổi ký tự mã hóa thành mã ASCII và trừ đi 97 để chuyển đổi thành số từ 0 đến 25
            ascii_code = ord(encrypted_message[i]) - 97
            # Thực hiện phép giải mã bằng cách trừ đi số ngẫu nhiên tương ứng
            new_ascii_code = (ascii_code - random_numbers[i]) % 26
            # Chuyển đổi số vừa giải mã thành ký tự và thêm vào chuỗi giải mã
            decrypted_message += chr(new_ascii_code + 97)
    return decrypted_message

# Yêu cầu người dùng nhập chuỗi mã hóa và dãy số ngẫu nhiên
encrypted_message = input("Nhập chuỗi mã hóa: ")
random_numbers = [int(num) for num in input("Nhập dãy số ngẫu nhiên (cách nhau bằng dấu cách): ").split()]
# Thực hiện giải mã
decrypted_message = one_time_pad_decrypt(encrypted_message, random_numbers)
print("Chuỗi đã được giải mã: ", decrypted_message)
