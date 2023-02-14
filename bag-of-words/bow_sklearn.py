from sklearn.feature_extraction.text import CountVectorizer

corpus = ["John likes to watch movies. Mary likes movies too.",
          "Mary also likes to watch football games.",
          "John's favourite movie is Gravity.", 
          "Mary can't stand romantic comedies."]

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus)

print("##### Vocabulary #####")
print(vectorizer.get_feature_names_out())

print("#### Representation #####")
print(X.toarray())
