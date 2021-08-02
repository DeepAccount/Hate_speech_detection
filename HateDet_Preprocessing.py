import html
import os
import re

import pandas as pd
import preprocessor as p
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize


def clean(word, stop_word_removal, stemming, lowercasing):
    """
    Returns Preprocessed sentence
    :param word:
    :type word: String
    :param stop_word_removal:
    :type stop_word_removal: boolean
    :param stemming:
    :type stemming: boolean
    :param lowercasing:
    :type lowercasing: boolean
    :return:
    :rtype: String
    """
    # remove encoding
    word_1 = word.encode('ascii', 'ignore').decode('unicode-escape').encode('iso-8859-1').decode('utf-8')

    # escape HTML
    word_2 = html.unescape(word_1)

    # pip install tweet-preprocessor
    # tweet cleaning
    p.set_options(p.OPT.URL, p.OPT.MENTION, p.OPT.EMOJI, p.OPT.SMILEY)
    word_3 = p.clean(word_2)
    #    print(word_3)

    if word_3.startswith("b'") or word_3.startswith('b"'):
        word_4 = word_3[2:]
    else:
        word_4 = word_3

    if word_4.startswith("RT"):
        word_5 = word_4[2:]
    else:
        word_5 = word_4

    # Remove extra chars & numbers
    word_6 = re.sub('[^A-Za-z]+', ' ', word_5)
    #    print(word_6)

    # lowercase
    if lowercasing:
        word_7 = word_6.lower()
    else:
        word_7 = word_6

    if (not stop_word_removal) and (not stemming):
        return word_7

    # remove stop words and strip
    if stop_word_removal:
        text_tokens = word_tokenize(word_7)
        word_8_list = [word for word in text_tokens if not word in stopwords.words()]
        word_8 = ''
        for w in word_8_list:
            word_8 = word_8 + " " + w

    if not stemming:
        return word_8.strip()

    if stop_word_removal:
        text_tokens = word_tokenize(word_8)
    else:
        text_tokens = word_tokenize(word_7)

    # Stemming
    if stemming:
        stemmer = PorterStemmer()
        stems = [stemmer.stem(word) for word in text_tokens]
        word_9 = ''
        for w in stems:
            word_9 = word_9 + " " + w
        word_9 = word_9.strip()
        return word_9

    return word_7


def create_clean_file(input_file_name, clean_file_name, output_folder):
    """

    Returns Preprocessed dataframe for Hate and category identification
    Saves same in excel for reuse
    :param input_file_name:
    :type input_file_name: String
    :param clean_file_name:
    :type clean_file_name: String
    :param output_folder:
    :type output_folder: String
    :return:
    :rtype: Pandas dataframe
    """
    working_dir = os.getcwd() + '/' + output_folder + '/'
    cleanfilename = working_dir + clean_file_name

    cleandf = ''
    try:
        cleandf = pd.read_excel(cleanfilename)
        print("Loaded existing Clean file.")
        return cleandf
    except:
        print("Loading Raw file...")
        tweet_data = pd.read_excel(input_file_name)
        print('cleaning...')
        datarows = []
        print(tweet_data.shape)
        for index, row in tweet_data.iterrows():
            #        print(index)
            H = row['Hate or Not']
            if (H == 0 or H == 1):
                c = ""
                try:
                    c = clean(row['Text'], stop_word_removal=False, stemming=False, lowercasing=True)
                except:
                    print("Error reading " + str(index))
                if str(c).lower() != 'nan' and str(c).lower() != '':
                    datarows.append([c, H, row['Hate Category']])
        print("CLeaning done.")
        cleandf = pd.DataFrame(datarows)
        print("Saving Clean file...")
        cleandf.to_excel(cleanfilename, index=False)
        return cleandf


def create_clean_file_target_identification(input_file_name, clean_file_name, output_folder):
    """

    Returns Preprocessed dataframe for Target identification
    Saves same in excel for reuse
    :param input_file_name:
    :type input_file_name: String
    :param clean_file_name:
    :type clean_file_name: String
    :param output_folder:
    :type output_folder: String
    :return:
    :rtype: Pandas dataframe
    """
    working_dir = os.getcwd() + '/' + output_folder + '/'
    cleanfilename = working_dir + clean_file_name

    cleandf = ''
    try:
        cleandf = pd.read_excel(cleanfilename)
        print("Loaded existing Clean file.")
        return cleandf
    except:
        print("Loading Raw file...")
        tweet_data = pd.read_excel(input_file_name)
        print('cleaning...')
        datarows = []
        print(tweet_data.shape)
        for index, row in tweet_data.iterrows():
            #        print(index)
            H = row['Hate or Not']
            if (H == 0 or H == 1):
                c = ""
                try:
                    c = clean(row['Text'], stop_word_removal=True, stemming=False, lowercasing=False)
                except:
                    print("Error reading " + str(index))
                if str(c).lower() != 'nan' and str(c).lower() != '':
                    datarows.append([c, H, row['Hate Category']])
        print("CLeaning done.")
        cleandf = pd.DataFrame(datarows)
        print("Saving Clean file...")
        cleandf.to_excel(cleanfilename, index=False)
        return cleandf
