import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import WordNetLemmatizer
para ="""When mice are kept at high population densities, their behaviour changes in a number of ways. 
        Aggressive activity within populations of mice rises as density increases. 
        Cannibalism of young also goes up, and so does aberrant sexual activity. 
        Communal nesting, frequent in natural mouse populations, increases abnormally. 
        In one example, 58 mice one to three days old (from several litters) were found in one nest, most unusual communal living. 
        None survived because most of the mothers deserted them immediately after birth."""

sentence = nltk.sent_tokenize(para)

corpus = []
lemmatizer = WordNetLemmatizer()
for i in range(len(sentence)):
    words =  nltk.word_tokenize(sentence[i])
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words if word not in stopwords.words('english')]
    sentence[i] = ' '.join(lemmatized_words)
    corpus.append(sentence[i])

cv = TfidfVectorizer()
X = cv.fit_transform(corpus).toarray()
#your actual bag of words lol
feature_names = cv.get_feature_names_out()
print(feature_names)
