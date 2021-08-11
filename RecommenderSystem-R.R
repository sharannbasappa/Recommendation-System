
 NAME: SHARANNBASAPPA
 BATCH:050121 10AM







#########################################QUESTION-1############################################
                              #################################


library(recommenderlab)
library(reshape2)

#loading game dataset to R
games <-read.csv("C:\\Users\\shilpa\\Desktop\\Datasets_Recommendation Engine\\game.csv",header=TRUE)
head(game_1)
dim(game_1)

#coverting data into matrix format
game_matrix <- as.matrix(acast(games, userId~game, fun.aggregate = mean))
dim(game_matrix)

## recommendarlab realRatingMatrix format
Rating <- as(game_matrix, "realRatingMatrix")

rec1 = Recommender(Rating, method="UBCF") ## User-based collaborative filtering
rec2 = Recommender(Rating, method="IBCF") ## Item-based collaborative filtering
rec3 = Recommender(Rating, method="SVD")
rec4 = Recommender(Rating, method="POPULAR")
rec5 = Recommender(binarize(Rating,minRating=2), method="UBCF") ## binarize all 2+ rating to 1

## create n recommendations for a user
uid = "3"
gamen <- subset(games, game_1$userId==uid)
print("You have rated:")
gamen
print("recommendations for you:")
prediction <- predict(rec1, Rating[uid], n=1) 
as(prediction, "list")
prediction <- predict(rec2, R[uid], n=2)
as(prediction, "list")
prediction <- predict(rec3, R[uid], n=2)
as(prediction, "list")
prediction <- predict(rec4, R[uid], n=2) 
as(prediction, "list")
prediction <- predict(rec5, R[uid], n=2) 
as(prediction, "list")


#------------------------------Problem 2------------------------------------------#




library(recommenderlab)
library(reshape2)


entertainment <-read.csv("C:\\Users\\shilpa\\Desktop\\Datasets_Recommendation Engine\\Entertainment.csv",header=TRUE)
head(entertainment)
entertainment <- entertainment[,2:4]
head(entertainment)
dim(entertainment)

## covert to matrix format
?acast
ratings_matrix <- as.matrix(acast(entertainment,Titles~Category, fun.aggregate = mean))
dim(ratings_matrix)

## recommendarlab realRatingMatrix format
R <- as(ratings_matrix, "realRatingMatrix")

rec1 = Recommender(R, method="UBCF") ## User-based collaborative filtering
rec2 = Recommender(R, method="IBCF") ## Item-based collaborative filtering
rec3 = Recommender(R, method="SVD")
rec4 = Recommender(R, method="POPULAR")
rec5 = Recommender(binarize(R,minRating=2), method="UBCF") ## binarize all 2+ rating to 1

## create n recommendations for a user
uid = "Father of the Bride Part II (1995)"
movies <- subset(ratings_list, ratings_list$user==uid)
print("You have rated:")
movies
print("recommendations for you:")
prediction <- predict(rec1, R[uid], n=2) ## you may change the model here
as(prediction, "list")
prediction <- predict(rec2, R[uid], n=2) ## you may change the model here
as(prediction, "list")
prediction <- predict(rec3, R[uid], n=2) ## you may change the model here
as(prediction, "list")
prediction <- predict(rec4, R[uid], n=2) ## you may change the model here
as(prediction, "list")
prediction <- predict(rec5, R[uid], n=2) ## you may change the model here
as(prediction, "list")
