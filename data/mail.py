import csv
import re

# Tạo danh sách chứa các địa chỉ email
emails = []

# Mở tệp CSV để đọc
with open('data.csv', 'r') as file:
    # Đọc dữ liệu từ tệp CSV và chuyển đổi nó thành một danh sách các hàng
    rows = csv.reader(file)
    
    # Lặp lại các hàng trong tệp CSV
    for row in rows:
        # Sử dụng biểu thức chính quy để tìm kiếm email trong mỗi hàng
        email = re.findall('\S+@\S+', row[0])
        
        # Nếu tìm thấy email, thêm nó vào danh sách
        if email:
            emails.append(email[0])

# Mở tệp CSV để ghi
with open('emails.csv', 'w', newline='') as file:
    # Tạo một đối tượng csv.writer
    writer = csv.writer(file)
    
    # Lặp lại các địa chỉ email trong danh sách và viết chúng vào tệp CSV mới
    for email in emails:
        writer.writerow([email])
