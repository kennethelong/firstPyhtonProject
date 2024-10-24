# Function to reverse a sentence
def reverse_sentence(sentence):
    return sentence[::-1]

# Input: 3 sentences
sentences = [
    "The quick brown fox jumps over the lazy dog",
    "Python programming is fun",
    "I am learning machine learning"
]

# Output each sentence in reverse order
for sentence in sentences:
    reversed_sentence = reverse_sentence(sentence)
    print(f"Original: {sentence}")
    print(f"Reversed: {reversed_sentence}\n")
