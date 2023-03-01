def encode_message(message):
    even_chars = message[::2]
    odd_chars = message[1::2]
    encoded = even_chars + odd_chars
    return encoded

# Get the message input from the user
message = input("Enter a message to encode: ")

# Encode the message
encoded_message = encode_message(message)

# Print the encoded message
print("The encoded message is:", encoded_message)
