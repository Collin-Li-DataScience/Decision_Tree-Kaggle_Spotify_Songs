from collections import Counter
import math
import pandas as pd

"""
artists and release_dates have been removed, but the artist and the month where the songs are released might be important features. Just make sure you don't add the year as a feature for the decade prediction, since this is what you are trying to predict.
"""

class PreprocessSongsDecadePrediction:
     # TODO: Complete functions to improve results

    def __init__(self, min_artist_count=10):
        pass
        
    
    def fit(self, X, y=None):
        return self
        
    
    def transform(self, X, y=None):
        X = X.drop(['id', 'name','artists'], axis=1) # min change to make the model run. You can change this
        # Dropped in notebook: ['year', 'popularity']
        
        '''
        # 1st
        X = X.drop(['energy', 'loudness'], axis = 1)
        
        # 2nd: not doing it
        '''
        
        # 3rd
        X['release_date'] = pd.to_datetime(X['release_date'])
        X['day_of_the_week'] = X['release_date'].dt.dayofweek
        
        X = X.drop(columns = ['release_date'])
        
        
        
        # 4th
        duration_outlier = X['duration_ms'].quantile(.95)
        X['duration_ms'] = X['duration_ms'].apply(lambda x: min(x,duration_outlier))
        
        speechiness_outlier = X['speechiness'].quantile(.95)
        X['speechiness'] = X['speechiness'].apply(lambda x: min(x,speechiness_outlier))
        
        loudness_outlier = X['loudness'].quantile(.05)
        X['loudness'] = X['loudness'].apply(lambda x: max(x,loudness_outlier))
        
        
        return X
    
    
    def fit_transform(self, X, y=None):
        #function that applies both functions to make it easier. You don't have to use nor modify it
        self.fit(X,y)
        return self.transform(X, y)
        

class PreprocessSongsPopularityPrediction:
    # TODO: Complete functions to improve results
    def __init__(self):
        pass
        
    
    def fit(self, X, y=None):
        return self
        
    
    def transform(self, X, y=None):
        X = X.drop(['id', 'name', 'artists'], axis=1)  # min change to make the model run. You can change this
        
        # 1st: added year since it has high correlation with popularity
        
        # 2nd - get day of the week:
        X['release_date'] = pd.to_datetime(X['release_date'])
        X['day_of_the_week'] = X['release_date'].dt.dayofweek
        
        X = X.drop(columns = ['release_date'])
       
        # 3rd - outlier:
        duration_outlier = X['duration_ms'].quantile(.95)
        X['duration_ms'] = X['duration_ms'].apply(lambda x: min(x,duration_outlier))
        
        speechiness_outlier = X['speechiness'].quantile(.95)
        X['speechiness'] = X['speechiness'].apply(lambda x: min(x,speechiness_outlier))
        
        loudness_outlier = X['loudness'].quantile(.05)
        X['loudness'] = X['loudness'].apply(lambda x: max(x,loudness_outlier))
        
        # 4th - StandardScaler
        # X = StandardScaler().fit(X)
       
        return X
