import os.path

import HateDet_Preprocessing
import HateDet_Target
import joblib

"""
Goal: Running SVC Model
HATE CLASSIFICATION APP USING SVC
"""

working_dir = os.getcwd() + '/output/'

# Load any trained model to be used. Below models can be changed by any of the models
# Different models for finding hate and Category
# Here We are choosing Support vector classifier with TF-IDF to find hate
vectorizerfname = working_dir + 'TfidfVectorizer.vec'
vectorizer = joblib.load(vectorizerfname)
modelfname = working_dir + 'SVC TfidfVectorizer.mdl'
model_hate = joblib.load(modelfname)

# Logistic Regression with Count vectoriser to find category
vectorizerfname_cat = working_dir + 'CountVectorizerCategory.vec'
vectorizer_cat = joblib.load(vectorizerfname_cat)
modelfname_cat = working_dir + 'Logistic Regression CountVectorizer.mdlCategory'
model_category = joblib.load(modelfname_cat)


print("######################## HATE CLASSIFICATION APP USING SVC######################")
print("")
print(" Sample tweets" )
print("1) Dear asshole Hu Xi ji we are not hiding how many are affected by Chinesevirus and how many are dead by Chinesevirus. Do you have guts to tell truth how many died in China due to Chinesevirus?? ")
print("2) I am Indian and I am proud of this")
print("3) Waiting for Corona to end so that we can go back to college")
print("")
while (1):
    print("")
    print("Type a statement to test the Hate Classification. Type '0' to come out")
    inp = input()
    if inp == '0':
        break
    elif inp == '':
        continue
    else:
        preprocessed = HateDet_Preprocessing.clean(word=inp, stop_word_removal=False, stemming=False,
                                                   lowercasing=True)
        tweet = []
        tweet.append(preprocessed)
        val = str(model_hate.predict(vectorizer.transform(tweet)))
        if val == "[1]":
            print("Hate Non-Hate : " + "Hate")
        else:
            print("Hate Non-Hate : " + "Non Hate")

        category = str(model_category.predict(vectorizer_cat.transform(tweet)))
        print("Category : " + category)

        preprocessed_target = HateDet_Preprocessing.clean(word=inp, stop_word_removal=True, stemming=False,
                                                          lowercasing=False)
        print("Target : " + HateDet_Target.get_target(preprocessed_target, category))
