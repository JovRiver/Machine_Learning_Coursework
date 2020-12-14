# This program reads from a list of emails and splits them into a training set and test set, and then attempts
# to predict spam emails from among the test set.

import csv
import re
import math
import numpy as np
from collections import Counter

# THIS DETERMINES THE PERCENTAGE OF TRAINING EMAILS
percent_training = 0.8

#-------------------------------------------------------------------------------------------------------------------------------------------------------------
    # EMAIL PARSING SECTION
#-------------------------------------------------------------------------------------------------------------------------------------------------------------

# Open and read the csv file, process each email, and add them to a list of emails
file = open('spam-apache.csv', encoding="utf8")
csv_reader = csv.reader(file)
emails = [list(filter(None, re.split('[,> "</+=\[\])(*&^%$#_@!.\{\}|:\';?]', str(email)))) for email in csv_reader]

# Determine the size of the training set
num_training = int(len(emails) * percent_training)

# Training emails contains all of the emails from 0-num_training
training_emails = emails[:num_training]
# Test emails contains all of the emails from num_training-end
test_emails = emails[num_training:]

print('Percentage of training emails: {:.2g}%\nNumber of Training Emails: {}\nNumber of Test Emails: {}\n'.format(percent_training * 100, num_training, len(test_emails)))

# Find the number of spam emails and ham emails in both the training set and the test set
num_spam = len([email for email in training_emails if email[0] == '-1'])
num_ham = len([email for email in training_emails if email[0] == '1'])

#-------------------------------------------------------------------------------------------------------------------------------------------------------------
    # WORD SELECTION SECTION
#-------------------------------------------------------------------------------------------------------------------------------------------------------------

# I first grabbed all the words of length greater than 2 from spam and ham and placed them in their own dictionaries (Counter counts the number of times each word occurs)
spam_dict = dict(Counter([word.lower() for email in emails for word in email if email[0] == '-1' and word.isalpha() == True and len(word) > 2]))
ham_dict = dict(Counter([word.lower() for email in emails for word in email if email[0] == '1' and word.isalpha() == True and len(word) > 2]))
print(f'DICTIONARY LENGTHS:\nNumber of Spam Words: {len(spam_dict)}\nNumber of Ham Words: {len(ham_dict)}')

# I chose these words to ignore based off of small likelihood of indicators or outliers (with noticeable presence) in my final train_words
ignore = ["you", "she", "they", "your", "but", "and", "the", "that", "they", "their", "there", "they're", "can", "not", "this", "nfor", "then",
          "will", "wont", "from", "all", "are", "have", "out", "for", "with", "nto", "html", "nhttp", "www", "com", "out", "http", "nthis", "nthe", "nyour",
          "get", "just", "how", "nalso", "has", "nbut", "ftp", "nand", "only", "been", "had", "nis", "here", "asp", "nif", "nplease", "nvisit", "even", "better",
          "nof", "ncould", "nmail", "nyou", "npeople", "nwill", "nwith", "nout", "nwould", "njust", "nnow", "nwant", "nwe", "nthan", "nno", "nbeen", "nthat", "nthen",
          "nthere", "nat", "ncan", "our", "those", "email", "now", "sure", "via", "its", "too", "than", "few", "ilug", "find"]

# I chose all the words from the spam and ham dictionaries that met my criteria below
spam_words = [word for word in spam_dict if word in ham_dict and word in spam_dict and spam_dict.get(word) > ham_dict.get(word) and word not in ignore and spam_dict.get(word) > 2]
ham_words = [word for word in ham_dict if word in spam_dict and word in ham_dict and ham_dict.get(word) > spam_dict.get(word) and word not in ignore and ham_dict.get(word) > 2]
print(f'\nLIST LENGTHS AFTER PRUNING:\nNumber of Spam Words: {len(spam_words)}\nNumber of Ham Words: {len(ham_words)}\n')

# Grab an equal amount of words from spam words and ham words and add them as our training words
num_words = min(len(spam_words), len(ham_words))
train_words = spam_words[:num_words] + ham_words[:num_words]

#-------------------------------------------------------------------------------------------------------------------------------------------------------------
    # INITIAL PROBABILITIES SECTION
#-------------------------------------------------------------------------------------------------------------------------------------------------------------

# Create two lists to hold our word probabilities for words in spam emails and words in ham emails
spam_words_prob = [(spam_dict.get(word) / num_spam) for word in train_words]
ham_words_prob = [(ham_dict.get(word) / num_ham) for word in train_words]

# We find the probabilities of spam emails and ham emails from our training set
spam_email_prob = num_spam / num_training
ham_email_prob = num_ham / num_training

#-------------------------------------------------------------------------------------------------------------------------------------------------------------
    # FORMULA SECTION
#-------------------------------------------------------------------------------------------------------------------------------------------------------------

# These will hold our guesses
spam_guess = []
ham_guess = []

# This is based off of the formula given in our assignment
for email in test_emails:
    probability = math.log(spam_email_prob / ham_email_prob)
    for word in email:
        if word in train_words:
            i = train_words.index(word)
            prob = spam_words_prob[i] / ham_words_prob[i]
            probability += math.log(prob)
    # Check our probability and if it is greater than 0 we guess spam, else we guess ham
    if probability > 0:
        spam_guess.append(email)
    else:
        ham_guess.append(email)

#-------------------------------------------------------------------------------------------------------------------------------------------------------------
    # RESULTS SECTION
#-------------------------------------------------------------------------------------------------------------------------------------------------------------

# Find the number of true spam and false spam from spam_guess
num_true_spam = len([email for email in spam_guess if email[0] == '-1'])
num_false_spam = len([email for email in spam_guess if email[0] == '1'])

# Find the number of true ham and false ham from ham_guess
num_false_ham = len([email for email in ham_guess if email[0] == '-1'])
num_true_ham = len([email for email in ham_guess if email[0] == '1'])

# These are the results
print('Spam Guess : {:<3} | Ham Guess : {:<3}\nActual Spam: {:<3} | False Spam: {:<3} | Acc: {:<}%\nActual Ham : {:<3} | False Ham : {:<3} | Acc: {:<}%\n'.format
      (len(spam_guess), len(ham_guess), 
       num_true_spam, num_false_spam,  int(num_true_spam / len(spam_guess) * 100),
       num_true_ham, num_false_ham, int(num_true_ham / len(ham_guess) * 100)))

# Confusion matrix to show the spread of false hits and true hits
confusion_matrix = np.array([[num_true_spam, num_false_spam], [num_false_ham, num_true_ham]])
print('Confusion Matrix:\n{}\n'.format(confusion_matrix))