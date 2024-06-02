import pandas as pd
import pickle 
import json
import numpy as np
import sklearn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef
pd.options.display.max_colwidth = 350

import subprocess
from torch import cuda
import sys

from text_clean import normalizeTweet

from simpletransformers.classification import ClassificationModel, ClassificationArgs



n_test = 700
consensus_criterion = 'all3' # requires all three mturk labelers to agree for the monsted dataset. This gave best results.
seed = 1



# our_data = pd.read_csv('./labeled_tweets/antivax_labels.csv').dropna()
our_data = pd.read_parquet('./labeled_tweets/antivax_labels_rebalanced.parquet').dropna()
monsted_data = pd.read_csv('./labeled_tweets/monsted_mturk_vaxx_labels.tsv', delimiter='	')[['tweetid','text','rating1','rating2','rating3']]
test2 = pd.read_parquet('./labeled_tweets/test_set.parquet')



#reformat monsted data
monsted_data['1=2'] = monsted_data['rating1']==monsted_data['rating2']
monsted_data['1=3'] = monsted_data['rating1']==monsted_data['rating3']
monsted_data['2=3'] = monsted_data['rating2']==monsted_data['rating3']
monsted_data['all3'] = [monsted_data['1=2'].iloc[i] and monsted_data['1=3'].iloc[i] for i in range(len(monsted_data))]
monsted_data['any2'] = [monsted_data['1=2'].iloc[i] or monsted_data['1=3'].iloc[i] or monsted_data['2=3'].iloc[i] for i in range(len(monsted_data))]
monsted_data['label'] = [np.median([monsted_data['rating1'].iloc[i], monsted_data['rating2'].iloc[i], monsted_data['rating3'].iloc[i]]) for i in range(len(monsted_data))]
monsted_data = monsted_data[['tweetid','text','label','any2','all3']].rename(columns={'tweetid':'tweet_id'})
monsted_data['is_antivax']=monsted_data['label']==1
monsted_data = monsted_data[monsted_data[consensus_criterion]][['text','is_antivax']].reset_index(drop=True)




#shuffle, make test-train split, rename label column
monsted_data = monsted_data.sample(frac=1, replace=False, random_state=seed).rename(columns={'is_antivax':'label'})[['text', 'label']]
monsted_data_test = monsted_data[:n_test]
monsted_data_train = monsted_data[n_test:]

our_data = our_data.rename(columns={'is_antivax':'label'})[['text', 'label']]
our_data_test = our_data[:n_test]
our_data_train = our_data[n_test:].sample(frac=1, replace=False, random_state=seed)



#train
use_cuda = cuda.is_available()
print('using_cuda '+str(use_cuda), file=sys.stderr)
subprocess.run('rm -r outputs', shell=True)
model_args = ClassificationArgs(num_train_epochs=2, train_batch_size = 8, learning_rate = 4e-5,
                                no_save = True, no_cache=True, overwrite_output_dir = True)
model = ClassificationModel("bertweet", "vinai/bertweet-covid19-base-uncased", args=model_args, use_cuda=use_cuda,
                            weight = [our_antivax_frac, 1-our_antivax_frac],
                           )
# model = ClassificationModel("roberta", "roberta-base", args=model_args, use_cuda=use_cuda,
#                             weight = [our_antivax_frac, 1-our_antivax_frac],
#                            )
training_data = pd.concat([our_data_train, monsted_data_train.sample(int(len(our_data_train)*.4))])
model.train_model(training_data)
pickle.dump(model, open('my_model.pkl', 'wb'))



#evaluate


#load model
model = pickle.load(open('my_model.pkl', 'rb'))

def get_ul(data, q=0.95):
    margin = (1-q)/2
    upper = data.quantile(1-margin)
    lower = data.quantile(margin)
    ul = (lower,upper)
    return ul

def get_pm(data, q=0.95):
    get_ul(data, q)
    mean = data.mean()
    pm = max(mean-lower, upper-mean)
    return pm

def bootstrap_results(res, y, n=1000):
    
    y_hat_prob = res[1][:,1]
    y_hat = y_hat_prob > 0.5
    idx = pd.Series(list(range(0,len(y))))
    
    acc = list()
    precision = list()
    recall = list()
    f1 = list()
    mcc = list()
    auroc = list()
    
    for i in range(0,n):
        samples = list(idx.sample(frac=1,replace=True))
        acc.append(accuracy_score(y[samples],y_hat[samples]))
        precision.append(precision_score(y[samples],y_hat[samples]))
        recall.append(recall_score(y[samples],y_hat[samples]))
        f1.append(f1_score(y[samples],y_hat[samples]))
        mcc.append(matthews_corrcoef(y[samples], y_hat[samples]))
        auroc.append(roc_auc_score(y[samples],y_hat_prob[samples]))
        
    acc = pd.Series(acc)
    precision = pd.Series(precision)
    recall = pd.Series(recall)
    f1 = pd.Series(f1)
    mcc = pd.Series(mcc)
    auroc = pd.Series(auroc)
    
    print('accuracy:', acc.mean(), '+/-', get_pm(acc))
    print('precision:', precision.mean(), '+/-', get_pm(precision))
    print('recall:', recall.mean(), '+/-', get_pm(recall))
    print('f1:', f1.mean(), '+/-', get_pm(f1))
    print('mcc:', mcc.mean(), '+/-', get_pm(mcc))
    print('auroc:', auroc.mean(), '+/-', get_pm(auroc))

    return

test_df = pd.concat([our_data_test,test2]).reset_index(drop=True)
res = model.eval_model(test_df)
print(bootstrap_results(res, y = np.array(test_df['label']), n=1000))

#save
pickle.dump(model, open('./models_etc/good_model_apr21.pkl', 'wb'))





