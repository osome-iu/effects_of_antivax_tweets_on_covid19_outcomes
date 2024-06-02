import pandas as pd
import pickle
from simpletransformers.classification import ClassificationModel
import time

from torch import cuda
use_cuda = cuda.is_available()

import sys
print('using_cuda '+str(use_cuda), file=sys.stderr)

#load data and model
df = pd.read_parquet('./models_etc/all_geotagged_tweets.parquet')
tweet_texts = list(df['text'].drop_duplicates().reset_index(drop=True))
model = pickle.load(open('./models_etc/good_model_apr21.pkl', 'rb'))

#run classifier inference on texts
start = time.time()
predictions = list()
for i in range(0, len(tweet_texts), 50000): #predict in superbatches. prevents memory leak overflow in bertweet tokenizer
    pred, _ = model.predict(tweet_texts[i:i + 50000])
    predictions.append(pred)
predictions = [item for sublist in predictions for item in sublist] #flatten list of lists into list
end = time.time()
print('elapsed_time',(end-start))

#append labels to df
labels = pd.DataFrame(list(zip(tweet_texts,predictions)), columns = ['text','is_antivax'])
df = df.merge(labels, on='text',how='inner')

#save
df.to_parquet('./models_etc/labeled_geotagged_tweets.parquet')
print(df.info())
print(df.head())
