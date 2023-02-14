from bow import BagofWords
import numpy as np

def cosine_similarity(v1, v2):
    dot_product = np.dot(v1, v2)
    magnitude_v1 = np.sqrt(np.dot(v1, v1))
    magnitude_v2 = np.sqrt(np.dot(v2, v2))
    return dot_product / (magnitude_v1 * magnitude_v2)

if __name__ == '__main__':
    a = "I love playing soccer in the park." 
    b = "Playing soccer in the park is one of my favorite things to do."
    c =  "I love to read books in the library."
    s = [a,b,c]
    bow = BagofWords(s)

    print("\n##### Vocabulary #####")
    print(bow.vocab)

    print("\n#### Representation #####")
    rep = bow.get_all_vectors()
    print(rep)

    cos1 = cosine_similarity(rep[0], rep[1])
    cos2 = cosine_similarity(rep[1], rep[2])

    print("\nCosine Similarities")
    print("1 & 2: ", cos1)
    print("2 & 3: ", cos2)

    t = ["John's favourite movie is Gravity.", "Mary can't stand romantic comedies."]
    bow.add_sentences(t)

    print("\n##### New Vocabulary #####")
    print(bow.vocab)

    print("\n#### New Representation #####")
    print(bow.get_all_vectors())

