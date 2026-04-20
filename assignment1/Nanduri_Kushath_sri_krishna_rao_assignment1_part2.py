# %% [markdown]
# # Assignment 1 - Part 2: Text Preprocessing with NLTK
# 
# **Course:** Natural Language Processing
# 
# **Total Points:** 10 points (contributes to 50% of Assignment 1)
# 
# ---
# 
# ## Instructions
# 
# 1. Complete all the functions marked with `# YOUR CODE HERE`
# 2. **DO NOT** change the function names or their signatures
# 3. Each function must return the exact type specified
# 4. Test your functions by running the test cells
# 5. When finished:
#    - Export this notebook as a Python file (.py)
#    - **Name the file:** `LASTNAME_FIRSTNAME_assignment1_part2.py`
#    - Example: `DUPONT_Jean_assignment1_part2.py`
#    - Push to your GitHub repository
#    - Send the .py file by email to: **yoroba93@gmail.com**
# 
# ---
# 
# ## Assignment Overview
# 
# In this assignment, you will use NLTK to analyze the Herman Melville novel **Moby Dick**.
# 
# You will practice:
# - Tokenization
# - Frequency analysis
# - Stop word removal
# - Stemming and lemmatization
# - Building a preprocessing pipeline
# 
# ---

# %% [markdown]
# ## Setup

# %%
import nltk
import pandas as pd
import numpy as np
import re

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('averaged_perceptron_tagger_eng', quiet=True)

from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords

# Load the novel
with open('moby.txt', 'r') as f:
    moby_raw = f.read()

# Create NLTK Text object
moby_tokens = nltk.word_tokenize(moby_raw)
text1 = nltk.Text(moby_tokens)

print(f"Loaded Moby Dick")
print(f"Raw text length: {len(moby_raw)} characters")
print(f"First 200 characters: {moby_raw[:200]}")

# %% [markdown]
# ---
# 
# ## Example Functions
# 
# These examples show you how to work with the text:

# %%
# Example 1: Count total tokens
def example_one():
    return len(nltk.word_tokenize(moby_raw))

print(f"Total tokens: {example_one()}")

# Example 2: Count unique tokens
def example_two():
    return len(set(nltk.word_tokenize(moby_raw)))

print(f"Unique tokens: {example_two()}")

# Example 3: Lemmatize verbs and count unique
def example_three():
    lemmatizer = WordNetLemmatizer()
    lemmatized = [lemmatizer.lemmatize(w, 'v') for w in text1]
    return len(set(lemmatized))

print(f"Unique tokens after verb lemmatization: {example_three()}")

# %% [markdown]
# ---
# 
# ## Question 1 (1 point)
# 
# **What is the lexical diversity of the text?**
# 
# Lexical diversity = ratio of unique tokens to total number of tokens
# 
# *This function should return a float.*

# %%
def question_one():
    """
    Calculate the lexical diversity of the text.
    
    Returns:
        float: Ratio of unique tokens to total tokens
    """
    tokens = nltk.word_tokenize(moby_raw)
    return len(set(tokens)) / len(tokens)

# Test your function
q1_result = question_one()
print(f"Lexical diversity: {q1_result}")
# Expected: approximately 0.08 (8%)

# %% [markdown]
# ---
# 
# ## Question 2 (1 point)
# 
# **What percentage of tokens is 'whale' or 'Whale'?**
# 
# *This function should return a float (percentage, e.g., 0.5 for 0.5%).*

# %%
def question_two():
    """
    Calculate the percentage of tokens that are 'whale' or 'Whale'.
    
    Returns:
        float: Percentage of whale tokens
    """
    tokens = nltk.word_tokenize(moby_raw)
    whale_count = tokens.count('whale') + tokens.count('Whale')
    return (whale_count / len(tokens)) * 100

# Test your function
q2_result = question_two()
print(f"Percentage of 'whale'/'Whale': {q2_result}%")

# %% [markdown]
# ---
# 
# ## Question 3 (1 point)
# 
# **What are the 20 most frequently occurring (unique) tokens in the text? What is their frequency?**
# 
# *This function should return a list of 20 tuples `(token, frequency)`, sorted in descending order of frequency.*

# %%
def question_three():
    """
    Find the 20 most frequent tokens and their frequencies.
    
    Returns:
        list: List of 20 tuples (token, frequency) sorted by frequency descending
    """
    fdist = nltk.FreqDist(text1)
    return fdist.most_common(20)

# Test your function
q3_result = question_three()
print("20 most frequent tokens:")
for token, freq in q3_result:
    print(f"  {token}: {freq}")

# %% [markdown]
# ---
# 
# ## Question 4 (1 point)
# 
# **What tokens have a length greater than 5 and a frequency of more than 150?**
# 
# *This function should return an alphabetically sorted list of tokens.*

# %%
def question_four():
    """
    Find tokens with length > 5 and frequency > 150.
    
    Returns:
        list: Alphabetically sorted list of tokens
    """
    fdist = nltk.FreqDist(text1)
    return sorted([token for token, freq in fdist.items() if len(token) > 5 and freq > 150])

# Test your function
q4_result = question_four()
print(f"Found {len(q4_result)} tokens:")
print(q4_result)

# %% [markdown]
# ---
# 
# ## Question 5 (1 point)
# 
# **Find the longest word in text1 and its length.**
# 
# *This function should return a tuple `(longest_word, length)`.*

# %%
def question_five():
    """
    Find the longest word in the text.
    
    Returns:
        tuple: (longest_word, length)
    """
    longest = max(text1, key=len)
    return (longest, len(longest))

# Test your function
q5_result = question_five()
print(f"Longest word: '{q5_result[0]}' with length {q5_result[1]}")

# %% [markdown]
# ---
# 
# ## Question 6 (1 point)
# 
# **What unique words (only alphabetic tokens) have a frequency of more than 2000?**
# 
# Use `isalpha()` to check if the token is a word and not punctuation.
# 
# *This function should return a list of tuples `(frequency, word)` sorted in descending order of frequency.*

# %%
def question_six():
    """
    Find words with frequency > 2000.
    
    Returns:
        list: List of tuples (frequency, word) sorted by frequency descending
    """
    fdist = nltk.FreqDist(text1)
    result = [(freq, word) for word, freq in fdist.items() if word.isalpha() and freq > 2000]
    return sorted(result, reverse=True)

# Test your function
q6_result = question_six()
print("Words with frequency > 2000:")
for freq, word in q6_result:
    print(f"  {word}: {freq}")

# %% [markdown]
# ---
# 
# ## Question 7 (1 point)
# 
# **What is the average number of tokens per sentence?**
# 
# *This function should return a float.*

# %%
def question_seven():
    """
    Calculate the average number of tokens per sentence.
    
    Returns:
        float: Average tokens per sentence
    """
    sentences = sent_tokenize(moby_raw)
    return sum(len(word_tokenize(s)) for s in sentences) / len(sentences)

# Test your function
q7_result = question_seven()
print(f"Average tokens per sentence: {q7_result}")

# %% [markdown]
# ---
# 
# ## Question 8 (1 point)
# 
# **Remove stop words from the text and return the 10 most common remaining words.**
# 
# Only consider alphabetic tokens (use `isalpha()`).
# 
# *This function should return a list of 10 tuples `(word, frequency)` sorted by frequency descending.*

# %%
def question_eight():
    """
    Find 10 most common words after removing stop words.
    
    Returns:
        list: List of 10 tuples (word, frequency) sorted by frequency descending
    """
    stop_words = set(stopwords.words('english'))
    tokens = [w.lower() for w in text1 if w.isalpha() and w.lower() not in stop_words]
    fdist = nltk.FreqDist(tokens)
    return fdist.most_common(10)

# Test your function
q8_result = question_eight()
print("10 most common words (excluding stop words):")
for word, freq in q8_result:
    print(f"  {word}: {freq}")

# %% [markdown]
# ---
# 
# ## Question 9 (1 point)
# 
# **Apply Porter stemming to all words and return the 10 most common stems.**
# 
# Only consider alphabetic tokens.
# 
# *This function should return a list of 10 tuples `(stem, frequency)` sorted by frequency descending.*

# %%
def question_nine():
    """
    Find 10 most common stems using Porter stemmer.
    
    Returns:
        list: List of 10 tuples (stem, frequency) sorted by frequency descending
    """
    stemmer = PorterStemmer()
    stems = [stemmer.stem(w) for w in text1 if w.isalpha()]
    fdist = nltk.FreqDist(stems)
    return fdist.most_common(10)

# Test your function
q9_result = question_nine()
print("10 most common stems:")
for stem, freq in q9_result:
    print(f"  {stem}: {freq}")

# %% [markdown]
# ---
# 
# ## Question 10 (1 point)
# 
# **Create a complete preprocessing function that:**
# 1. Tokenizes the text
# 2. Converts to lowercase
# 3. Removes non-alphabetic tokens
# 4. Removes stop words
# 5. Applies lemmatization
# 
# Apply this function to the first 1000 characters of Moby Dick.
# 
# *This function should return a list of preprocessed tokens.*

# %%
def question_ten():
    """
    Preprocess the first 1000 characters of Moby Dick.
    
    Returns:
        list: List of preprocessed tokens
    """
    text = moby_raw[:1000]
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    tokens = word_tokenize(text)
    tokens = [t.lower() for t in tokens]
    tokens = [t for t in tokens if t.isalpha()]
    tokens = [t for t in tokens if t not in stop_words]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]

    return tokens

# Test your function
q10_result = question_ten()
print(f"Number of preprocessed tokens: {len(q10_result)}")
print(f"First 20 tokens: {q10_result[:20]}")

# %% [markdown]
# ---
# 
# ## Summary of Functions for Grading
# 
# Make sure all these functions are properly implemented before exporting:

# %%
# Run this cell to verify all functions exist and return correct types
print("Checking functions...")

try:
    r1 = question_one()
    assert isinstance(r1, float), "question_one should return a float"
    print("✓ question_one: OK")
except Exception as e:
    print(f"✗ question_one: {e}")

try:
    r2 = question_two()
    assert isinstance(r2, float), "question_two should return a float"
    print("✓ question_two: OK")
except Exception as e:
    print(f"✗ question_two: {e}")

try:
    r3 = question_three()
    assert isinstance(r3, list) and len(r3) == 20, "question_three should return a list of 20 tuples"
    print("✓ question_three: OK")
except Exception as e:
    print(f"✗ question_three: {e}")

try:
    r4 = question_four()
    assert isinstance(r4, list), "question_four should return a list"
    print("✓ question_four: OK")
except Exception as e:
    print(f"✗ question_four: {e}")

try:
    r5 = question_five()
    assert isinstance(r5, tuple) and len(r5) == 2, "question_five should return a tuple of 2 elements"
    print("✓ question_five: OK")
except Exception as e:
    print(f"✗ question_five: {e}")

try:
    r6 = question_six()
    assert isinstance(r6, list), "question_six should return a list"
    print("✓ question_six: OK")
except Exception as e:
    print(f"✗ question_six: {e}")

try:
    r7 = question_seven()
    assert isinstance(r7, float), "question_seven should return a float"
    print("✓ question_seven: OK")
except Exception as e:
    print(f"✗ question_seven: {e}")

try:
    r8 = question_eight()
    assert isinstance(r8, list) and len(r8) == 10, "question_eight should return a list of 10 tuples"
    print("✓ question_eight: OK")
except Exception as e:
    print(f"✗ question_eight: {e}")

try:
    r9 = question_nine()
    assert isinstance(r9, list) and len(r9) == 10, "question_nine should return a list of 10 tuples"
    print("✓ question_nine: OK")
except Exception as e:
    print(f"✗ question_nine: {e}")

try:
    r10 = question_ten()
    assert isinstance(r10, list), "question_ten should return a list"
    print("✓ question_ten: OK")
except Exception as e:
    print(f"✗ question_ten: {e}")

print("\nDone! Export this notebook as .py file when all functions pass.")

# %% [markdown]
# ---
# 
# ## Submission Checklist
# 
# - [ ] All 10 functions are implemented
# - [ ] All functions return the correct type
# - [ ] Notebook exported as Python file
# - [ ] File named: `LASTNAME_FIRSTNAME_assignment1_part2.py`
# - [ ] Pushed to GitHub repository
# - [ ] Sent to **yoroba93@gmail.com**


