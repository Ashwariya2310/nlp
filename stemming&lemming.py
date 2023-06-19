import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

para ="""When mice are kept at high population densities, their behaviour changes in a number of ways. 
        Aggressive activity within populations of mice rises as density increases. 
        Cannibalism of young also goes up, and so does aberrant sexual activity. 
        Communal nesting, frequent in natural mouse populations, increases abnormally. 
        In one example, 58 mice one to three days old (from several litters) were found in one nest, most unusual communal living. 
        None survived because most of the mothers deserted them immediately after birth."""

sentence = nltk.sent_tokenize(para)

stemmer = PorterStemmer()
print(len(set(stopwords.words('english'))))
for i in range(len(sentence)):
    words = nltk.word_tokenize(sentence[i])
    stemmed_words = [stemmer.stem(word) for word in words if word not in stopwords.words('english')]
    sentence[i] = " ".join(stemmed_words)

print(sentence)
print(' ')

#####################

sentence = nltk.sent_tokenize(para)
lemmatizer = WordNetLemmatizer()
for i in range(len(sentence)):
    words =  nltk.word_tokenize(sentence[i])
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words if word not in stopwords.words('english')]
    sentence[i] = ' '.join(lemmatized_words)

print(sentence)