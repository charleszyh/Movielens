##########################################################
#' Algorithm.R 
#' Author: Yuanhang Zhang (Charles)
#' This file is created to run the prediction algorithm and give RMSE
#' Please refer to the Report-new for explanation
##########################################################


##########################################################
# Create edx and final_holdout_test sets 
##########################################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

options(timeout = 120)

dl <- "ml-10M100K.zip"
if(!file.exists(dl))
  download.file("https://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings_file <- "ml-10M100K/ratings.dat"
if(!file.exists(ratings_file))
  unzip(dl, ratings_file)

movies_file <- "ml-10M100K/movies.dat"
if(!file.exists(movies_file))
  unzip(dl, movies_file)

ratings <- as.data.frame(str_split(read_lines(ratings_file), fixed("::"), simplify = TRUE),
                         stringsAsFactors = FALSE)
colnames(ratings) <- c("userId", "movieId", "rating", "timestamp")
ratings <- ratings %>%
  mutate(userId = as.integer(userId),
         movieId = as.integer(movieId),
         rating = as.numeric(rating),
         timestamp = as.integer(timestamp))

movies <- as.data.frame(str_split(read_lines(movies_file), fixed("::"), simplify = TRUE),
                        stringsAsFactors = FALSE)
colnames(movies) <- c("movieId", "title", "genres")
movies <- movies %>%
  mutate(movieId = as.integer(movieId))

movielens <- left_join(ratings, movies, by = "movieId")

# Final hold-out test set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.6 or later
# set.seed(1) # if using R 3.5 or earlier
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in final hold-out test set are also in edx set
final_holdout_test <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from final hold-out test set back into edx set
removed <- anti_join(temp, final_holdout_test)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

##########################################################
# Generating test set and training set
##########################################################
set.seed(1) # To ensure the code and results are reproducible
test_indices <- createDataPartition(y = edx$rating, times = 1, p = 0.1, list = FALSE)

test <- edx[test_indices,]
train <- edx[-test_indices,]


##########################################################
# Deriving models
##########################################################

#####
# Average rating
mu <- train %>% summarise(mu = mean(rating)) %>% #Calculating the mean ratings
  select(mu) #extract the mu column

mu <- mu[1,1]

train_n <- train %>% mutate(mu = mu)
final_holdout_test_n <- final_holdout_test %>% mutate(mu = mu)

#####
# Movie effect

b_i <- train %>% mutate(mu = mu) %>% 
  mutate(b_i = rating - mu) %>% 
  group_by(movieId) %>% #Calclulating b_i at the aggregation of movie level
  summarise(b_i = mean(b_i))

train_n <- train_n %>% 
  left_join(b_i, by = "movieId")

final_holdout_test_n <- final_holdout_test_n %>% 
  left_join(b_i, by = "movieId") %>% 
  mutate(b_i = ifelse(is.na(b_i), 0, b_i)) #fill the NA with 0

#####
# User effect

b_u <- train_n %>% 
  mutate(b_u = rating - mu - b_i) %>% 
  group_by(userId) %>% #Calclulating b_u at the aggregation of movie level
  summarise(b_u = mean(b_u))

train_n <- train_n %>% 
  left_join(b_u, by = "userId")

final_holdout_test_n <- final_holdout_test_n %>% 
  left_join(b_u, by = "userId") %>% 
  mutate(b_u = ifelse(is.na(b_u), 0, b_u))

#####
# Age effect

# Extract birth year from the title

mv_time <- train %>% group_by(movieId, title) %>% # group by movie to extract birth year
  summarise(ave_rat = mean(rating)) %>% # make sure the grouping is executed
  mutate(rls_yr = str_extract(title, "\\((\\d{4})\\)$", group = 1)) %>% 
  # extract the continuous 4 numerical value (standing for the year) from the title
  mutate(rls_yr = as.numeric(rls_yr)) # transform the year into numeric

# Transform the timestamp

train_n <- train_n %>% mutate(timestamp = as_datetime(timestamp))
final_holdout_test_n <- final_holdout_test_n %>% mutate(timestamp = as_datetime(timestamp))

# Calculating age effect

age_effect <- train_n %>% 
  mutate(age_effect = rating - mu - b_u - b_i) %>% # calculate residual
  left_join(mv_time, by = "movieId") %>% # merge the birth year data
  mutate(age = year(timestamp) - rls_yr) %>% # compute age
  group_by(age) %>% # group the data by different ages
  summarise(age_effect = mean(age_effect)) # calculate average rating

# Merge age effect into data sets
train_n <- train_n %>% 
  left_join(mv_time, by = "movieId") %>% # First merge the birth year
  mutate(age = year(timestamp) - rls_yr) %>% 
  left_join(age_effect, by = "age") %>% # Then merge the age effect
  mutate(age_effect = ifelse(is.na(age_effect), 0, age_effect)) # Fill the NA

final_holdout_test_n <- final_holdout_test_n %>% 
  left_join(mv_time, by = "movieId") %>% # First merge the birth year
  mutate(age = year(timestamp) - rls_yr) %>% 
  left_join(age_effect, by = "age") %>% # Then merge the age effect
  mutate(age_effect = ifelse(is.na(age_effect), 0, age_effect)) # Fill the NA

#####
# Genre effect

# Count all the genres:

movie_genres <- train %>% group_by(movieId, title, genres) %>% 
  summarise(ave_rat = mean(rating)) # First group the training set into the movie-level data

all_genres <- unlist(str_split(movie_genres$genres, "\\|")) 
# Use str_split() to split the genre string into some individual words.

movie_genres <- as.data.frame(table(all_genres)) %>%  
  # Count the frequency of the genres and transform it into the data frame
  rename(genres = all_genres) # Rename a column

movie_genres <- movie_genres[-1,] #exclude the first row "no genres listed"

# Generate dummy variables for each genre

for (i in movie_genres$genres){ #for all the generes
  
  genre_str <- as.character(i) # turn the genre into a string
  genre_sym <- sym(genre_str) # generate the symbol of this genre
  
  train_n <- train_n %>% mutate(!!genre_sym := ifelse(str_detect(genres, genre_str), 1, 0)) 
  #if the genre column include the specific pattern, we make the identifyer 1
  
}

# Calculate the genre effect

train_n <- train_n %>% mutate(genre_effect = 0) # generate the genre effect column

for (i in movie_genres$genres){
  
  genre_str <- as.character(i) # turn the genre into a string
  genre_sym <- sym(genre_str) # generate the symbol of this genre
  
  genre_effect_val <- train_n %>% filter(!!genre_sym == 1) %>% 
    mutate(genre_effect_tmp = rating - mu - b_i - b_u - age_effect - genre_effect) %>% 
    # calculate the residual
    summarise(mean(genre_effect_tmp))  %>% 
    as.numeric() # extract the residual for this genre
  
  
  train_n <- train_n %>% 
    mutate(!!genre_sym := ifelse(!!genre_sym == 1, genre_effect_val, 0))  %>% 
    # put the genre effect value in the appropriate column
    mutate(genre_effect = genre_effect + !!genre_sym) 
  # put the genre effect we calculated added in the genre_effect column
}

# Extract genre effect and merge that into the test set
genre_effect <- train_n %>% group_by(movieId, genre_effect) %>% 
  summarise(mean(rating)) %>% 
  select(movieId, genre_effect)

final_holdout_test_n <- final_holdout_test_n %>% left_join(genre_effect, by = "movieId") %>% 
  mutate(genre_effect = replace_na(genre_effect, 0))

##########################################################
# Test out the final model
##########################################################

#Generate final prediction

y_hat_final <- final_holdout_test_n %>%   
  mutate(y_hat = mu + b_i + b_u + age_effect + genre_effect) %>% 
  .$y_hat

#Calculate RMSE of the final prediction:

RMSE(y_hat_final, final_holdout_test_n$rating)
# FINAL RMSE: 0.8652576
##########################################################




