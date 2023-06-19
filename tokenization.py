import nltk
# nltk.download()
# nltk.download('punkt')

para ="""When mice are kept at high population densities, their behaviour changes in a number of ways. 
        Aggressive activity within populations of mice rises as density increases. 
        Cannibalism of young also goes up, and so does aberrant sexual activity. 
        Communal nesting, frequent in natural mouse populations, increases abnormally. 
        In one example, 58 mice one to three days old (from several litters) were found in one nest, most unusual communal living. 
        None survived because most of the mothers deserted them immediately after birth."""
sent = nltk.sent_tokenize(para)

# print(sent)

print("No of sentences", len(sent))

word = nltk.word_tokenize(para)

print("No of words", len(word))



