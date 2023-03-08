
def count_words(string):
    # Tách chuỗi thành các từ bằng cách sử dụng hàm split()
    words = string.split()

    # Đếm số từ trong danh sách các từ
    count = len(words)

    return count

string = "This is a sample string\nwith multiple lines."
word_count = count_words(string)
print(f"The string has {word_count} words.")  # Output: The string has 8 words.
