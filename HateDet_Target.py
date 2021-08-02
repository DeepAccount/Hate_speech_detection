import pandas as pd
import spacy

ner = spacy.load('en_core_web_md')

def generate_targets_fullfile(df):
    """
     Creates and Excel file with all targets generated
    :param df:
    :type df: pandas dataframe
    """
    x_texts = df[0]
    hate_non_hate = df[1]
    y = df[2]

    datarows = []
    for i in range(len(df)):
        if hate_non_hate.iloc[i] == 1:
            category = y.iloc[i]
            tweet_data = x_texts.iloc[i]
            target = get_target(tweet_data, category)

            print("Target " + target)
            datarows.append([tweet_data, 1, category, target])

    new_df = pd.DataFrame(datarows)
    print("Saving Clean file...")
    new_df.to_excel("Hate_targets_genrated.xlsx", index=False)


def get_target(tweet_data, category):
    """
    Finds hate target in a tweet
    :param tweet_data:
    :type tweet_data: String
    :param category:
    :type category: String
    :return:
    :rtype: String
    """

    remove_entity_type = ['MONEY', 'TIME', 'ORDINAL', 'PERCENT', 'DATE', 'CARDINAL']

    category_token = ner(category)
    candidates = []
    doc = ner(tweet_data)

    entity = doc.ents
    for e in entity:
        label = e.label_
        if label not in remove_entity_type:
            candidates.append(str(e))

    if len(candidates) == 0:
        target = 'Others'
    elif len(candidates) == 1:
        target = candidates[0]
    else:
        max_sim = 0
        target = 'Others'
        for can in candidates:
            sim = category_token.similarity(ner(can))
            if sim > max_sim:
                target = can

    if target == 'Others' :
        np_labels = set(['amod', 'dobj', 'pobj'])
        word_list = []
        for word in doc:
            if word.dep_ in np_labels:
                word_list.append(word.text)

        actual_word = ''
        for w in word_list:
            actual_word = actual_word + ' ' + w

        if actual_word.strip() == '':
            actual_word = 'Others'

        target = actual_word

    return target
