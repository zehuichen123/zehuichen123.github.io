---
layout: default
title: Relearning Tabular Data Competition--Kaggle IEEE Fraud Detection
date: 2019-10-13 13:43 +0800
---

> Content of this article
>
> - Intro
> - Main Section
>   - Identity Clients, to find magic UID
>   - Validation strategies 
>   - Quick visualization of feature importance
>   - Feature Engineering
>     - Feature Generation
>     - Feature Selection -- Time consistency
>   - Speed and Memory Optimization
> - Summary

## Intro

With the help of my friend Mono(actually he managed all stuffs at the last period of competition), I got my first bronze medal in Kaggle. In order to get better rankings next time, I decide to revisiting this competition and dive deeper to learn those tricks and insights behind tabular data competition. Most of them comes from kernels/ discussions published by those top  winners in this competition. 

## Main Section

### Identity(client) is the most important, not the time

Fraud detection is actually one time series-based task since transactions varies through time. However, in this competition, time is not important since most of users with the same `uid` doesn't appear in the test dataset. So the solution is to identify those users by constructing features which identity them. And that's why all top winners mentioned `uid` in their methods sharing parts. 

The intuition of this idea is to manually group transactions which helps the model to identity different clients. You can refer to [these figures](https://www.kaggle.com/cdeotte/xgb-fraud-with-magic-0-9600#How-the-Magic-Works) to see how it works in a toy example.

Actually, it can be divided into 3 steps,

- Construct UID, to identify those clients. In this competition, it can be: 

  ```python
  X_train['day'] = X_train.TransactionDT / (24*60*60)
  X_train['uid'] = X_train.card1_addr1.astype(str)+'_'+np.floor(X_train.day-X_train.D1).astype(str)
  
  X_test['day'] = X_test.TransactionDT / (24*60*60)
  X_test['uid'] = X_test.card1_addr1.astype(str)+'_'+np.floor(X_test.day-X_test.D1).astype(str)
  ```

- Group Aggregation Features

  With the help of uids, we were able to construct new features by group aggregation features based on uids.

- Remove UIDs

  Since we don't use uids in these uids to avoid overfitting.

### Validation Strategies

During the competition, we found that it's quite hard to establish one reliable local CV. Local CV is extremely import when performing FE. Also, removing strongly time-related features is quite important. Here I want to note several tricks here.

- **Adverserial Validation**: Find the features have different distributions in train and test dataset. Train one model to predict the cateogory of train/test. Check the features holding the top feature importances. 
- **Time-based Data Split**: Split data with the time axis, say, train the first several months skip one month and predict the last month.
- **GroupKFold**:  The training data are the months December 2017, January 2018, February 2018, March 2018, April 2018, and May 2018. We refer to these months as 12, 13, 14, 15, 16, 17. Fold one in GroupKFold will train on months 13 thru 17 and predict month 12. Note that the only purpose of month 12 is to tell XGB when to `early_stop` we don't actual care about the backwards time predictions. The model trained on months 13 thru 17 will also predict `test.csv` which is forward in time.

### Quickly Visualization Your Feature Importance

```python
feature_imp = pd.DataFrame(sorted(zip(clf.feature_importance(),features)), columns=['Value','Feature'])

plt.figure(figsize=(20, 10))
sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False).head(20))
plt.title('LightGBM Features')
plt.tight_layout()
plt.show()
plt.savefig('lgbm_importances-01.png')
```

### Feature Engineering

#### Feature Generatation Functions (Feature Encoding Functions)

Here are some feature encoding functions. (1) `encode_FE` does frequency encoding where it combines train and test first and then encodes. (2) `encode_LE` is a label encoded for categorical features (3) `encode_AG` makes aggregated features such as aggregated mean and std (4) `encode_CB` combines two columns (5) `encode_AG2` makes aggregated features where it counts how many unique values of one feature is within a group. Reference to <a href="https://www.kaggle.com/cdeotte/xgb-fraud-with-magic-0-9600#Encoding-Functions">Encoding Functions by cdeotte</a>.

```python
# FREQUENCY ENCODE TOGETHER
def encode_FE(df1, df2, cols):
    for col in cols:
        df = pd.concat([df1[col],df2[col]])
        vc = df.value_counts(dropna=True, normalize=True).to_dict()
        vc[-1] = -1
        nm = col+'_FE'
        df1[nm] = df1[col].map(vc)
        df1[nm] = df1[nm].astype('float32')
        df2[nm] = df2[col].map(vc)
        df2[nm] = df2[nm].astype('float32')
        print(nm,', ',end='')
        
# LABEL ENCODE
def encode_LE(col,train=X_train,test=X_test,verbose=True):
    df_comb = pd.concat([train[col],test[col]],axis=0)
    df_comb,_ = df_comb.factorize(sort=True)
    nm = col
    if df_comb.max()>32000: 
        train[nm] = df_comb[:len(train)].astype('int32')
        test[nm] = df_comb[len(train):].astype('int32')
    else:
        train[nm] = df_comb[:len(train)].astype('int16')
        test[nm] = df_comb[len(train):].astype('int16')
    del df_comb; x=gc.collect()
    if verbose: print(nm,', ',end='')
      
# GROUP AGGREGATION MEAN AND STD
# https://www.kaggle.com/kyakovlev/ieee-fe-with-some-eda
def encode_AG(main_columns, uids, aggregations=['mean'], train_df=X_train, test_df=X_test, 
              fillna=True, usena=False):
    # AGGREGATION OF MAIN WITH UID FOR GIVEN STATISTICS
    for main_column in main_columns:  
        for col in uids:
            for agg_type in aggregations:
                new_col_name = main_column+'_'+col+'_'+agg_type
                temp_df = pd.concat([train_df[[col, main_column]], test_df[[col,main_column]]])
                if usena: temp_df.loc[temp_df[main_column]==-1,main_column] = np.nan
                temp_df = temp_df.groupby([col])[main_column].agg([agg_type]).reset_index().rename(
                                                        columns={agg_type: new_col_name})

                temp_df.index = list(temp_df[col])
                temp_df = temp_df[new_col_name].to_dict()   

                train_df[new_col_name] = train_df[col].map(temp_df).astype('float32')
                test_df[new_col_name]  = test_df[col].map(temp_df).astype('float32')
                
                if fillna:
                    train_df[new_col_name].fillna(-1,inplace=True)
                    test_df[new_col_name].fillna(-1,inplace=True)
                
                print("'"+new_col_name+"'",', ',end='')
                
# COMBINE FEATURES
def encode_CB(col1,col2,df1=X_train,df2=X_test):
    nm = col1+'_'+col2
    df1[nm] = df1[col1].astype(str)+'_'+df1[col2].astype(str)
    df2[nm] = df2[col1].astype(str)+'_'+df2[col2].astype(str) 
    encode_LE(nm,verbose=False)
    print(nm,', ',end='')
    
# GROUP AGGREGATION NUNIQUE
def encode_AG2(main_columns, uids, train_df=X_train, test_df=X_test):
    for main_column in main_columns:  
        for col in uids:
            comb = pd.concat([train_df[[col]+[main_column]],test_df[[col]+[main_column]]],axis=0)
            mp = comb.groupby(col)[main_column].agg(['nunique'])['nunique'].to_dict()
            train_df[col+'_'+main_column+'_ct'] = train_df[col].map(mp).astype('float32')
            test_df[col+'_'+main_column+'_ct'] = test_df[col].map(mp).astype('float32')
            print(col+'_'+main_column+'_ct, ',end='')
```

#### Feature Selection -- Time Consistancy

Apart from adverserial validation to find strongly time-related features, we provide another way to detect those features without time consistancy. 

> One interesting trick called "time consistency" is to train a single model using a single feature (or small group of features) on the first month of train dataset and predict `isFraud` for the last month of train dataset. This evaluates whether a feature by itself is consistent over time. 95% were but we found 5% of columns hurt our models. They had training AUC around 0.60 and validation AUC 0.40. In other words some features found patterns in the present that did not exist in the future. 
>
> We added 28 new feature above. We have already removed 219 V Columns from correlation analysis done [here](https://www.kaggle.com/cdeotte/eda-for-columns-v-and-id). So we currently have 242 features now. We will now check each of our 242 for "time consistency". We will build 242 models. Each model will be trained on the first month of the training data and will only use one feature. We will then predict the last month of the training data. We want both training AUC and validation AUC to be above `AUC = 0.5`. It turns out that 19 features fail this test so we will remove them. Additionally we will remove 7 D columns that are mostly NAN.

### Speed and Memory Optimization

- Loading data with pickle format can be faster than csv format about 60 times in this dataset.
- Downcasting type for some of data to reduce your memory usage.

## Summary

I am looking forward to next tabular data competition, haha!

## Reference

[1] [xgb-fraud-with-magic-0-9600, kaggle kernel](https://www.kaggle.com/cdeotte/xgb-fraud-with-magic-0-9600)

[2] [How to Find UIDs - (Unique Identification), kaggle discussion](https://www.kaggle.com/c/ieee-fraud-detection/discussion/111510)

[3] [1st Place Solution - Part 1, kaggle discussion](https://www.kaggle.com/c/ieee-fraud-detection/discussion/111284#latest-647701)

[4] [1st Place Solution - Part 2, kaggle discussion](https://www.kaggle.com/c/ieee-fraud-detection/discussion/111308)