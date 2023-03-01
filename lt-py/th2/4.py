# a/
def encode_message(message):
    even_chars = message[::2]
    odd_chars = message[1::2]
    encoded = even_chars + odd_chars
    return encoded

# b/
def decode_message(encoded_message):
    midpoint = len(encoded_message) // 2
    even_chars = encoded_message[:midpoint]
    odd_chars = encoded_message[midpoint:]
    decoded = ""
    for i in range(midpoint):
        decoded += even_chars[i] + odd_chars[i]
    if len(encoded_message) % 2 == 1:
        decoded += encoded_message[-1]
    return decoded


# main program
def main():
    # Get the message input from the user
    message = input("Enter a message to encode: ")

    # Encode the message
    encoded_message = encode_message(message)

    # Print the encoded message
    print("The encoded message is:", encoded_message)

    # Decode the message
    decoded_message = decode_message(encoded_message)

    # Print the decoded message
    print("The decoded message is:", decoded_message)
    
if __name__ == "__main__":
    main()