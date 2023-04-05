# câu 1 
def is_roman_numeral(s):
    roman_numerals = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}
    for i in range(len(s)):
        if s[i] not in roman_numerals:
            return False
        if i > 0 and roman_numerals[s[i]] > roman_numerals[s[i-1]]:
            return False
    return True

print(is_roman_numeral('XIV'))

# Câu 2 - a
person = {}
person['name'] = input('Enter name: ')
person['phone'] = input('Enter phone number: ')
person['email'] = input('Enter email address: ')
d = []
d.append(person)

# câu 2 - b
for person in d:
    print(person['phone'])
    
# câu 2 - c
def is_valid_email(person):
    email = person['email']
    return '@' in email

print(is_valid_email(d[1]))

# câu 2 - d
for person in d:
    person['name'] = person['name'].upper()

# câu 2 - e
def remove_person(name_or_phone_or_email):
    for person in d:
        if person['name'] == name_or_phone_or_email or person['phone'] == name_or_phone_or_email or person['email'] == name_or_phone_or_email:
            d.remove(person)

# Xóa người có tên là "Todd" ra khỏi danh sách d
remove_person('Todd')

# Xóa người có số điện thoại là "555-1618" ra khỏi danh sách d
remove_person('555-1618')

# Xóa người có địa chỉ email là "lj@mail.net" ra khỏi danh sách d
remove_person('lj@mail.net')

