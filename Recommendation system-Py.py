#name: sharannbasappa
#batch: 050121 10AM

# import os
import pandas as pd

# import Dataset 
game_rec = pd.read_csv("C:\\Users\\shilpa\\Desktop\\Datasets_Recommendation Engine\\game.csv", encoding = 'utf8')
game_rec.shape # shape
game_rec.columns
game_rec.game # genre columns

from sklearn.feature_extraction.text import TfidfVectorizer #term frequencey- inverse document frequncy is a numerical statistic that is intended to reflect how important a word is to document in a collecion or corpus

# Creating a Tfidf Vectorizer to remove all stop words
tfidf = TfidfVectorizer(stop_words = "english")    # taking stop words from tfid vectorizer 

# replacing the NaN values in overview column with empty string
game_rec["game"].isnull().sum() 
game_rec["game"] = game_rec["game"].fillna(" ")

# Preparing the Tfidf matrix by fitting and transforming
tfidf_matrix = tfidf.fit_transform(game_rec.game)   #Transform a count matrix to a normalized tf or tf-idf representation
tfidf_matrix.shape #12294, 46

# with the above matrix we need to find the similarity score
# There are several metrics for this such as the euclidean, 
# the Pearson and the cosine similarity scores

# For now we will be using cosine similarity matrix
# A numeric quantity to represent the similarity between 2 movies 
# Cosine similarity - metric is independent of magnitude and easy to calculate 

# cosine(x,y)= (x.y⊺)/(||x||.||y||)

from sklearn.metrics.pairwise import linear_kernel

# Computing the cosine similarity on Tfidf matrix
cosine_sim_matrix = linear_kernel(tfidf_matrix, tfidf_matrix)

# creating a mapping of game_rec name to index number 
game_rec_index = pd.Series(game_rec.index, index = game_rec['game']).drop_duplicates()

game_rec_id = game_rec_index["Super Mario Galaxy"]
game_rec_id

def get_recommendations(Name, topN):    
    # topN = 10
    # Getting the movie index using its title 
    game_rec_id = game_rec_index[Name]
    
    # Getting the pair wise similarity score for all the game_rec's with that 
    # game_rec
    cosine_scores = list(enumerate(cosine_sim_matrix[game_rec_id]))
    
    # Sorting the cosine_similarity scores based on scores 
    cosine_scores = sorted(cosine_scores, key=lambda x:x[1], reverse = True)
    
    # Get the scores of top N most similar movies 
    cosine_scores_N = cosine_scores[0: topN+1]
    
    # Getting the movie index 
    game_rec_idx  =  [i[0] for i in cosine_scores_N]
    game_rec_scores =  [i[1] for i in cosine_scores_N]
    
    # Similar movies and scores
    game_rec_similar_show = pd.DataFrame(columns=["game", "Score"])
    game_rec_similar_show["game"] = game_rec.loc[game_rec_idx, "game"]
    game_rec_similar_show["Score"] = game_rec_scores
    game_rec_similar_show.reset_index(inplace = True)  
    # game_rec_similar_show.drop(["index"], axis=1, inplace=True)
    print (game_rec_similar_show)
      #return (game_rec_similar_show)

    
# Enter your game_rec and number of game_rec's to be recommended 
get_recommendations("Super Mario Galaxy", topN = 10)
game_rec_index["Super Mario Galaxy"]

##############################QUESTION-2############################################
                       ######################
# import os
import pandas as pd

# import Dataset 
entetirement_rec = pd.read_csv("C:\\Users\\shilpa\\Desktop\\Datasets_Recommendation Engine\\Entertainment.csv" , encoding = 'utf8')
entetirement_rec.shape # shape
entetirement_rec.columns
entetirement_rec.Titles
 # genre columns

from sklearn.feature_extraction.text import TfidfVectorizer #term frequencey- inverse document frequncy is a numerical statistic that is intended to reflect how important a word is to document in a collecion or corpus

# Creating a Tfidf Vectorizer to remove all stop words
tfidf = TfidfVectorizer(stop_words = "english")    # taking stop words from tfid vectorizer 

# replacing the NaN values in overview column with empty string
entetirement_rec["Titles"].isnull().sum() 
entetirement_rec["Titles"] = entetirement_rec["Titles"].fillna(" ")

# Preparing the Tfidf matrix by fitting and transforming
tfidf_matrix = tfidf.fit_transform(entetirement_rec.Titles)   #Transform a count matrix to a normalized tf or tf-idf representation
tfidf_matrix.shape #12294, 46

# with the above matrix we need to find the similarity score
# There are several metrics for this such as the euclidean, 
# the Pearson and the cosine similarity scores

# For now we will be using cosine similarity matrix
# A numeric quantity to represent the similarity between 2 movies 
# Cosine similarity - metric is independent of magnitude and easy to calculate 

# cosine(x,y)= (x.y⊺)/(||x||.||y||)

from sklearn.metrics.pairwise import linear_kernel

# Computing the cosine similarity on Tfidf matrix
cosine_sim_matrix = linear_kernel(tfidf_matrix, tfidf_matrix)

# creating a mapping of entetirement_rec name to index number 
entetirement_rec_index = pd.Series(entetirement_rec.index, index = entetirement_rec['Titles']).drop_duplicates()

entetirement_rec_id = entetirement_rec_index["Waiting to Exhale (1995)"]
entetirement_rec_id

def get_recommendations(Name, topN):    
    # topN = 10
    # Getting the movie index using its title 
    entetirement_rec_id = entetirement_rec_index[Name]
    
    # Getting the pair wise similarity score for all the entetirement_rec's with that 
    # entetirement_rec
    cosine_scores = list(enumerate(cosine_sim_matrix[entetirement_rec_id]))
    
    # Sorting the cosine_similarity scores based on scores 
    cosine_scores = sorted(cosine_scores, key=lambda x:x[1], reverse = True)
    
    # Get the scores of top N most similar movies 
    cosine_scores_N = cosine_scores[0: topN+1]
    
    # Getting the movie index 
    entetirement_rec_idx  =  [i[0] for i in cosine_scores_N]
    entetirement_rec_scores =  [i[1] for i in cosine_scores_N]
    
    # Similar movies and scores
    entetirement_rec_similar_show = pd.DataFrame(columns=["Titles", "Score"])
    entetirement_rec_similar_show["Titles"] = entetirement_rec.loc[entetirement_rec_idx, "Titles"]
    entetirement_rec_similar_show["Score"] = entetirement_rec_scores
    entetirement_rec_similar_show.reset_index(inplace = True)  
    # entetirement_rec_similar_show.drop(["index"], axis=1, inplace=True)
    print (entetirement_rec_similar_show)
      #return (entetirement_rec_similar_show)

    
# Enter your entetirement_rec and number of entetirement_rec's to be recommended 
get_recommendations("Jumanji (1995)", topN = 10)
entetirement_rec_index["Jumanji (1995)"]
