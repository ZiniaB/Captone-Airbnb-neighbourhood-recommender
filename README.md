

Capstone Project 

# <Project Title: Using Machine Learning to develop a recommendation system for London neighbourhoods>
## <Student Name: Zinia Bhattacharya>

==============================

## Project Description and Context

Airbnb currently has a number of features to help personalize the property based on guest requirements e.g.<br>
- Price range
- Type of stay and property (Entire apartment/private room etc or House/Flat/Guest House etc)
- Number of rooms & beds
- Amenities
- Host details

For listings and experiences in the countryside, Airbnb also provides additional recommendations to personalize one's stay e.g. Vineyards, Farms, Countryside, Surfing, Lakes, Beaches, Designer homes, Tiny homes etc.
**However, when it comes to city stays, currently consumers are shown all available properties on a map without any location recommendations or filters</mark>.** <br>

Potential airbnb guests have to scroll through comments to get a 'feel' of the neighbourhood or, they need to research separately about the suitability of the location, before narrowing down on a property. Looking at Google Search trends, searches around 'Where to stay in London' have an average relative popularity score of 70. For comparison, Goggle's own relative score is 91 and Amazon is approx 72 in the United in the same time period of last 1 year. So evidently, people are seeking this information to guide their decision on where to stay in London during their visit.

Adding such recommendation to the Airbnb site, can enhance the user-experience without the guest having to leave the Airbnb ecosystem for location guidance. It can also pave the way for sustainable tourism and drive tourism revenue for ‘under-the radar’ neighbourhoods and ease tourist overcrowding in central London <br>

This recommendation system also has the potential to be extended to other short-term rental booking sites and hotels and expanded to other top cities globally

----------------------

## Table of Contents 

[1] README.md
[2] Data Source and link to data -Google link to dataset: https://drive.google.com/file/d/1IM1CwhKhf1StfN7DCZ23Z_PDEWDo-D_D/view?usp=sharing

Notebooks
[1] Capstone Notebook 1-Data load and Pre-processing.ipynb
[2] Capstone Notebook 2- EDA.ipynb
[3] Capstone Notebook 3 - Modelling Part 1.pynb
[4] Capstone Notebook 4 - Modelling Part 2.pynb
[5] Capstone Notebook 5 - Modelling Part 3.pynb


Tableau files
[1] Airbnb EDA - "../notebooks/Airbnb EDA.twb"
[2] Airbnb model visualization - "../notebooks/Airbnb EDA.twb"

Requirements
[1] Requirements.txt
[2] save_env2.yml

------------------------

## Data sourcing and pre-processing
The data pre-processing step was a critical part of the project due to the nature of the data required - `suburb` level data being key and the dataset did not come pre-populated with this.<br>

The key steps in the data-sourcing and data pre-processing stage were:

(1) Identifying the right dataset from the inside airbnb source -Having evaluated a few different datasets, we finalized on the dataset used here as it provided a detailed level of data on the properties listed, key being `neighborhood_overview` which provides a short description of the neighborhood where the property is located <br>
(2) The next step was to narrow down the dataset to include only relevant columns. As the purpose of the project was to identify neighborhood profiles and recommend properties accordingly, the key columns retained in the final dataset were related to the property location <br>
(3) Imputing missing values for `name` and `review_scores_location` based on the property details that we could glean from the other columns - e.g. `name` details were completed by adding details from the property location and neighborhood overview and `review_scores_location` were filled in with average review scores for the corresponding London borough <br>
(4) Reverse geocoding to map `suburb` names as per latitude and longitude - We used 'nominatim open street' to request for suburb names based on the property location. It returned values for all 40,605 listings barring 2619 of them where nominatim had no suburb details available
(5) Inputing missing values for `suburbs` by requesting for `postcode` data and then manually mapping suburbs based on outer postcodes - The process for imputing missing `suburb` values was slightly time-consuming as we first went down the route of requesting for 'postcode' information which were then mapped to 'ward' level data for the 'City of London' borough as a test. However, the 'ward' level data was too narrow for the purposes of our project where the end objective is to profile and recommend broad 'suburbs' to the airbnb guests. We therefore changed the approach and looked at 'outer' level postcode e.g. BR2 in the postcode BR2 6AN to establish the suburbs against each postcode and map them against the relevant listing. For certain boroughs which are closely knit (e.g. City of London) or outer boroughs where the suburbs within are broadly similar (e.g. Richmond) we used the borough name as the `suburb` for the purposes of this project

-------------------------

## EDA

The EDA has been primarily carried out in Tableau but we have also used python Jupyter notebook for some of the visualizations as appropriate for the output required. We have included some of the screenshots from the Tableau files in the EDA notebook for reference. The link to the full Tableau EDA file has also been provided separately.


Key insights from the EDA:

(1) Central London boroughs had the highest concentration property listings with Westminister being significantly higher than the rest at 4500 listings and likely because of the popular suburbs like Hyde Park, Soho, Mayfair, Covent Garden etc that fall within it. This could suggest a strong demand for locations close to the tourist hubs but also from the host's POV, it is possible that many central London properties have been repurposed for airbnb listings and are 'commercially managed' by professional hosts <br>
(2) When the data is broken down by suburb, Whitechapel comes out as the suburb with the highest number of listings at 1000 but interestingly it does not fall in the Westminister borough but within Tower Hamlets which has the 3rd highest number of listings at a borough level <br>
(3) The Average #night price varies distinctly between boroughs e.g. $225+ for Westminister vs. ~$170 for Tower Hamlets and ~$75 for Bexley <br>
(4) The length of stay also tends to differ with some of the centrally located popular boroughs having a longer stay and Greenwich being the highest at 8+ nights <br>
(5) The Review Scores for location are very strong and skewed highly towards 5. This suggests that the guests were broadly satisfied with the location they chosee based on their individual preferences. This is good news for our project as there is no particular neighborhood that stands out as 'negative' in terms of scores. We can go ahead and include all neighborhoods in our recommendation system to be able to ensure that all neighborhood listings are fairly represented and are considered by the guests.

Overall, our EDA has established that there are many factors that come into play when choosing a property e.g. availability (number of listings within the timeframe), room type, price, length of stay etc. Also, in terms of location, different neighborhoods appear to serve different customer needs (accesibility, affordability, experience) and our hypothesis that the customer could benefit from an initial recommendation on neighborhoods, very much holds true given all the aforementioned factors that they need to consider while choosing a property.

--------------------------

## Modelling Part 1

This being an unsupervised NLP project, processing the text data in the right way to then test the outcome against different models was the basis of our machine learning task.

**Steps in text processing and modelling:**<br>
(1) Cleaning and preparing the text data for modelling:<br>
--(a) Removing strings<br>
--(b) Defining functions to process text data - remove punctuation, convert to lower case, apply stemming and lemmatization<br>
--(c) Defining function for custom tokenizer that applies relevant text processing from the revious step<br>

(2) Topic modelling with LDA (Latent Dirichlet Allocation)<br>
(3) KMeans modelling on vectorized data- Count Vectorizer and TFIDF<br>
(4) Wordembedding on sample dataset of 500 rows<br>
(5) KMeans modelling on sample dataset<br>

**LDA modelling approach and outcome:**<br>
Latent Dirichlet Allocation model is usually used in topic modelling. Here we have applied it to model the distribution of words in the document 'neighborhood_overview`. The intuition behind LDA is that each document in a corpus is a mixture of several topics, and each topic is a probability distribution over words used to describe the topic. LDA was our first choice because of its ability to infer the underlying topics and their associated word distributions from the observed documents <br>
 
We utilized the Gensim library for LDA topic modelling. Gensim stands for “Generate Similar” and is a popular open source natural language processing library used for unsupervised topic modeling. We also used the pylDavis library for visualizing and interpreting the topics in the LDA model.


The LDA model performed relatively well in allocating neighbourhood descriptions broadly across 5 profiles based on frequency of suburb names, proximity to specific attractions and unique descriptions of the neighborhood e.g. trendy, open etc

Profile 1: Tourist attractions and restaurants <br>
Profile 2: Neighbourhoods with easy transport accessibility <br>
Profile 3: Family friendly, community feel<br>
Profile 4: Tourist hub and riverside neighbourhoods <br>
Profile 5: Trendy, independent cafes, nightlife suburbs <br>


Challenges with the LDA model:
In our LDA model, we have not used bi-gram or tri-grams (e.g. spaces, open and green - all evaluated separately) because running with bi-grams is computationally expensive. The absence of bi-grams and tri-grams might have impacted the accuracy and interpretability of the profile allocation <br>

The allocation is also highly dependent on the occurence of different words within the `neighborhood_overview` and it may not be able to assess the context of the words necessarily e.g. a description for a property located in Dulwich suggesting that it is only a 20 minute train ride to London Bridge and the river, might lead the model to allocate Dulwich to neighbourhood profile that have riverside location (hypothetical example)

Overall, the LDA model is a very good baseline model to start profiling the neighborhoods but we will explore further models that can go beyond uni-word analysis and account for context, where possible.


**KMeans modelling approach and outcome:**<br>

Here we are doing to vectorize the data by applying CountVectorizer and then TF-IDF to see if there are any differences in the top tokens as per the two methods. Both methods are used to covert text data to vectors but each slightly different in their approach.<br>
CountVectorizer simply counts the number of times a word appears in a document, while TF-IDF (Term Frequency Inverse Document Frequency) takes into account not only how many times a word appears in a document but also how important that word is to the whole corpus.This is done by penalizing words that appear frequently across the corpus but attaching higher importance to words that occur more frequently within the document itself.

We have used the TF-IDF ventorized data for our KMeans modelling since it takes the relative importance of the tokens into account. However, the KMeans model did not perform too well in terms of creating distinct clusters of neighbourhood characteristics. Looking at the top terms per cluster, we could see a lot of overlap across clusters leading us to the conclusion that KMeans with TF-IDF is perhaps not the right model for our dataset.

**KMeans modelling with word embedding:**<br>
We then applied word-embedding, given their proven effectiveness in NLP tasks as they are able to capture the semantic meaning and contextual relationships of the words. We used the distilBERT embedding model which is pretrained for sentence embeddings and we tested it on a small subset of our dataframe. The results, although not enirely robust, are encouraging as we can start to see some distinct neighbourhood profiles.<br>

In conclusion, KMeans with word-embedding appears to work better than the other models and can result in clearer clusters with more fine-tuning of the corpus.


## Modelling Part 2
In part 2 of our modelling task, we introduce an additional data source. Here, we have used ChatGPT to source neighbourhood characteristics of the top 100 suburbs (sorted by number of airbnb listings).
From our initial modelling, we recognized that the way neighbourhoods are described in the airbnb listings can be limiting in profiling the suburbs appropriately. It is primarily because neighbourhood descriptions are written in a way that highlight accessibility of the property to Central London or to tourist attractions and parks and restaurants resulting in very similar terms across majority of listings.
The project's emphasis was to identify unique neighbourhood vibes and match neighbourhoods to distinct guest preferences. For this purpose, Chat GPT's neighbourhood descriptions proved very useful and relevant.

Following on from our work in Part 1, we applied diltilBERT sentence embedding on ChatGPT's neighbourhood descriptions(suburb_tags) and ran KMeans and DBSCAN clustering models. DBSCAN did not perform well but KMeans results were very encouraging. We identified some distinct neighbourhood clusters that can form a strong basis to for our neighbourood recommender system.


## Modelling Part 3
In the last part of our modelling task, we ran diltilBERT sentence embedding on the full dataset of 40605 rows on the column `neighborhood_overview`.
It was computationally extremely expensive but the objective was to look at the result in its entirety. Unfortunately, this did not perform well due to the reasons noted earier about the nature of the neighborhood descriptions in the dataset.

We will disregard the outcome of this exercise in our final recommendations but will retain the work done on this, for future reference

------------------------------

## Closing notes

In conclusion, word or sentence embedding is a really effective way to process and analyze text data, as long as the data is specific to the task at hand. DiltilBERT model proved very effective in clustering profiles based on ChatGPT data but fairly inconclusive on airbnb listing's neighborhood descriptions due to the significant overlap in words across all properties.

Sentence-embedding and KMeans on ChatGPT neighborhood profiles led to a solid baseline clustering of nLondon neighbourhoods and as next steps, we will fine-tune and expand that model to include all suburbs, beyond the Top100 for a complete neighbourhood recommender system for airbnb London



------------------------------
## Credits
With guidance from my educators at BrainStation - Mark Wentink, Shifath Nafis and Emanuel Lowcock.

Medium articles, tutorials and documentation
LDA
https://github.com/priya-dwivedi/Deep-Learning/blob/master/topic_modeling/LDA_Newsgroup.ipynb <br>
https://radimrehurek.com/gensim/models/ldamodel.html <br>
https://pyldavis.readthedocs.io/en/latest/modules/API.html <br>




------------