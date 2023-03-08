# Description: Tách câu thành danh sách các từ
def split_sentence(sentence):
    # Tách câu thành danh sách các từ
    words = sentence.split()

    return words

sentence = "This is a sample sentence."
word_list = split_sentence(sentence)
print(word_list)  # Output: ['This', 'is', 'a', 'sample', 'sentence.']
