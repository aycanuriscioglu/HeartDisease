import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# !pip install statsmodels
import statsmodels.stats.api as sms
from scipy.stats import ttest_1samp, shapiro, levene, ttest_ind, mannwhitneyu,pearsonr, spearmanr, kendalltau, f_oneway, kruskal
from statsmodels.stats.proportion import proportions_ztest
import math
import scipy.stats as st
from sklearn.preprocessing import MinMaxScaler



pd.set_option('display.width', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

#seti inceleme/veriyi hazırlama
df_ = pd.read_csv("new data sets/heart_2020_cleaned.csv")
df=df_.copy()

def check_df(dataframe, head=5):
    print("############Shape############")
    print(dataframe.shape)
    print("############Types############")
    print(dataframe.dtypes)
    print("############Tail############")
    print(dataframe.tail(head))
    print("############Head############")
    print(dataframe.head(head))
    print("############NA############")
    print(dataframe.isnull().sum())
    print("############Quantiles############")
    print(dataframe.describe([0,0.05, 0.25, 0.50, 0.75, 0.95,0.99,1]).T)

check_df(df)

for i in df.columns:
    print(df[i].value_counts())
    print('***************************************\n')


def grab_col_names(dataframe, cat_th=10, car_th=20):
    cat_cols= [col for col in df.columns if str(df[col].dtypes) in ["category","object", "bool"]]
    num_but_cat=[col for col in df.columns if df[col].nunique()<10 and df[col].dtypes in ["int", "float"]]
    cat_but_car= [col for col in df.columns if
                   df[col].nunique()>20 and str(df[col].dtypes) in ["category", "object"]]
    cat_cols= cat_cols+num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    num_cols= [col for col in df.columns if df[col].dtypes in ["int", "float"]]
    num_cols= [col for col in num_cols if col not in cat_cols]

    print(f"Observations): {dataframe.shape[0]}")
    print(f"Veriables): {dataframe.shape[1]}")
    print(f"cat_cols): {len(cat_cols)}")
    print(f"num_cols): {len(num_cols)}")
    print(f"cat_but_car): {len(cat_but_car)}")
    print(f"num_but_cat): {len(num_but_cat)}")

    return cat_cols, num_cols,cat_but_car, num_but_cat


cat_cols, num_cols,cat_but_car,num_but_cat= grab_col_names(df)

def check_outlier_graph(dataframe, col_name, plot= False):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        print(True)
        if plot:
            sns.boxplot(dataframe[col_name])
            plt.show(block= True)
            print(dataframe[col_name])
            print("###########################################################")
    else:
        print(False)
        print(dataframe[col_name])
        print("###########################################################")

for col in num_cols:
    check_outlier_graph(df, col, plot=True)
for col in num_cols:
    replace_with_thresholds(df, col)
    

def num_summary(dataframe, numerical_col, plot= False):
    quantiles= [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    dataframe[numerical_col].describe().T
    print(dataframe[numerical_col].describe(quantiles))

    if plot:
        dataframe[numerical_col].hist()
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block= True)

for col in num_cols:
    num_summary(df, col, plot=True)


def cat_summary(dataframe, col_name, plot=False):
    if dataframe[col_name].dtypes == "bool":\
        dataframe[col_name] =dataframe[col_name].astype(int)
        print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                            "Ratio": 100*dataframe[col_name].value_counts()/ len(dataframe)}))
        print("###########################################################")
        if plot:
            sns.countplot(x=dataframe[col_name],data=dataframe)
            plt.show(block=True)
        else:
            print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                            "Ratio": 100*dataframe[col_name].value_counts()/ len(dataframe)}))
        print("###########################################################")

for col in cat_cols:
    cat_summary(df, col, plot=True)


def rare_analyser(dataframe, target, cat_cols):
    for col in cat_cols:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "RATIO": dataframe[col].value_counts() / len(dataframe),
                            "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end="\n\n\n")

rare_analyser(df, "HeartDisease", cat_cols)

def target_summary_with_num(dataframe, target, numarical_col):
    print(dataframe.groupby(target).agg({numarical_col:"mean"}))

target_summary_with_num(df, "HeartDisease", num_cols)

corr= df.corr()

plt.figure(figsize=(18,18))
cor = df_upsampled.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds, fmt='.2f')
plt.show()


df.loc[(df['AgeCategory'] <= "30-34"), 'age_cat'] = 'young'
df.loc[(df['AgeCategory'] >= "35-39") & (df['AgeCategory'] <= "45-49") , 'age_cat'] = 'mature'
df.loc[(df['AgeCategory'] >= "50-54") & (df['AgeCategory']  <= "60-64"), 'age_cat'] = 'senior'
df.loc[(df['AgeCategory'] >= "65-69") &  (df['AgeCategory'] <= "75-79"), 'age_cat'] = 'old'
df.loc[(df['AgeCategory'] == "80 or older"), 'age_cat'] = 'veryold'

def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

binary_cols = [col for col in df.columns if df[col].dtype not in [int, float]
               and df[col].nunique() == 2]

for col in binary_cols:
    label_encoder(df, col)
    
    
def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

ohe_cols = [col for col in df.columns if 13 > df[col].nunique() > 2]

df=one_hot_encoder(df, ohe_cols)

cat_cols, num_cols,cat_but_car,num_but_cat= grab_col_names(df)


mms = MinMaxScaler()
df["BMI_min_max"] = mms.fit_transform(df[["BMI"]])
df["Physical_min_max"] = mms.fit_transform(df[["PhysicalHealth"]])
df["Mental_min_max"] = mms.fit_transform(df[["MentalHealth"]])


from sklearn.ensemble import RandomForestClassifier

y = df["HeartDisease"]
X = df.drop(["HeartDisease", "AgeCategory", "BMI", "MentalHealth", "PhysicalHealth"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)

rf_model = RandomForestClassifier(random_state=46).fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
accuracy_score(y_pred, y_test)



def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(18, 18))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                      ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')


plot_importance(rf_model, X_train)
