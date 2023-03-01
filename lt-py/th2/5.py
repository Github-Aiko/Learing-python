# a/
def encode_rail_fence_3(msg):
    # Chia chuỗi thành 3 nhóm
    group1 = ""
    group2 = ""
    group3 = ""
    for i in range(len(msg)):
        if i % 4 == 0:
            group1 += msg[i]
        elif i % 4 == 1 or i % 4 == 3:
            group2 += msg[i]
        else:
            group3 += msg[i]
    # Ghép các nhóm lại với nhau
    encoded_msg = group1 + group2 + group3
    return encoded_msg

# b/
def decode_rail_fence_3(msg):
    # Tính toán độ dài của các nhóm
    n = len(msg)
    len_group1 = n // 4 + (n % 4 >= 1)
    len_group2 = n // 4 * 2 + (n % 4 >= 2)
    len_group3 = n // 4 + (n % 4 == 3)
    # Tách chuỗi thành các nhóm
    group1 = msg[:len_group1]
    group2 = msg[len_group1:len_group1+len_group2]
    group3 = msg[len_group1+len_group2:]
    # Ghép các ký tự từ các nhóm lại với nhau
    decoded_msg = ""
    for i in range(max(len(group1), len(group2), len(group3))):
        if i < len(group1):
            decoded_msg += group1[i]
        if i < len(group2):
            decoded_msg += group2[i]
        if i < len(group3):
            decoded_msg += group3[i]
    return decoded_msg

# c/
def rail_fence_encode(text, rails):
    fence = [[] for _ in range(rails)]
    rail = 0
    direction = 1
    
    for char in text:
        fence[rail].append(char)
        rail += direction
        
        if rail == rails - 1:
            direction = -1
        elif rail == 0:
            direction = 1
    
    encoded = ""
    for rail in fence:
        encoded += "".join(rail)
    
    return encoded



# main program
def main():
    # Get the message input from the user
    message = input("Enter a message to encode: ")

    # Encode the message
    encoded_message = encode_rail_fence_3(message)

    # Print the encoded message
    print("The encoded message is:", encoded_message)

    # Decode the message
    decoded_message = decode_rail_fence_3(encoded_message)

    # Print the decoded message
    print("The decoded message is:", decoded_message)
    
    # string to encode 
    text = "Hello World"
    # number of rails
    rails = 3
    # encoded string
    encoded = rail_fence_encode(text, rails)
    
    print("The encoded message is:", encoded)
    
if __name__ == "__main__":
    main()
    