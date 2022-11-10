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
pd.set_option('display.float_format', lambda x: '%.5f' % x)

#seti inceleme/veriyi hazırlama
df_ = pd.read_csv("new data sets/heart_2020_cleaned.csv")
df=df_.copy()
df.head()
df.shape
df.isnull().sum()
df.describe().T

for i in df.columns:
    print(df[i].value_counts())
    print('***************************************\n')


#Bir veri setindeki değişken verilerini düzenleyerek tiplerine göre liste oluşturma;
def grab_col_names(dataframe, cat_th=10, car_th=20):
    """
    veri setindeki kategorik, numerik ve kategorik fakar kardinal değişkenlerin isimlerini verir.

    Parameters
    ----------
    dataframe: dataframe
        değişken isimleri alınmak istenen dataframedir
    cat_th :int, float
        numerik fakat kategorik olan değişkenler için sınıf eşik değeri
    car_th: int, float
        kategorik fakat kardinaldeğişkenler için sınıf eşik değeri

    Returns
    -------
    cat_cols: list
         kategorik değişken listesi
    num_cols: list
        numerik değişken listesi
    cat_but_car: list
        kategorik görünümlü kardinal değişken listesi

    Notes
    -------
    cat_cols + num_cols+ cat_but_car = toplam değişken sayısı
    num_but_cat cat_cols'un içinde
    """
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

#yes/no değerleri integera çevirme
change_values = {'No':0, 'Yes':1, 'No, borderline diabetes': 3,'Yes (during pregnancy)':4}
for i in range(0, len(cat_cols)):
    df[cat_cols[i]] = df[cat_cols[i]].replace(change_values)

df.head()
df.dtypes
df.describe()

#num_col listesini düzenleme
num_cols = [col for col in df.columns if df[col].dtypes in ["int64", "float64"]]
cat_cols = [col for col in df.columns if str(df[col].dtypes) in ["category", "object", "bool"]]


#aykırı değer baskılama
def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.25)
    quartile2 = dataframe[variable].quantile(0.50)
    quartile3 = dataframe[variable].quantile(0.75)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

outlier_thresholds(df, "BMI")
outlier_thresholds(df, "PhysicalHealth")
outlier_thresholds(df, "MentalHealth")
outlier_thresholds(df, "SleepTime")


def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    # dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

replace_with_thresholds(df, "BMI")
replace_with_thresholds(df, "PhysicalHealth")
replace_with_thresholds(df, "MentalHealth")
replace_with_thresholds(df, "SleepTime")

df.describe()


#Veri setindeki numerik değişkenlerin çeyreklik dilimlerde grafik halinde incelenmesi;
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



#Veri setindeki kategorik değişkenlerin yüzdelik dilimlerde grafik halinde incelenmesi;
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


#Bir değişkenin o veri seti içerisindeki diğer kategorik değişkenlerle çaprazlama incelenmesi;
cat_cols
def target_summary_with_cat(dataframe, target, categorical_col):
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean()}))

target_summary_with_cat(df, "HeartDisease", "Sex")
target_summary_with_cat(df, "HeartDisease", "AgeCategory")
target_summary_with_cat(df, "HeartDisease", "Race")
target_summary_with_cat(df, "HeartDisease", "GenHealth")



df.head()

#Bir değişkenin o veri seti içerisindeki diğer numerik değişkenlerle çaprazlama incelenmesi;

def target_summary_with_num(dataframe, target, numarical_col):
    print(dataframe.groupby(target).agg({numarical_col:"mean"}))

target_summary_with_num(df, "HeartDisease", "Smoking")
target_summary_with_num(df, "HeartDisease", "AlcoholDrinking")    #a/b sonucuna göre değerlendirme yap
target_summary_with_num(df, "HeartDisease", "Stroke")
target_summary_with_num(df, "HeartDisease", "DiffWalking")
target_summary_with_num(df, "HeartDisease", "Diabetic")
target_summary_with_num(df, "HeartDisease", "PhysicalActivity")
target_summary_with_num(df, "HeartDisease", "PhysicalHealth")
target_summary_with_num(df, "HeartDisease", "MentalHealth")
target_summary_with_num(df, "HeartDisease", "SleepTime")
target_summary_with_num(df, "HeartDisease", "BMI")
target_summary_with_num(df, "HeartDisease", "Asthma")
target_summary_with_num(df, "HeartDisease", "KidneyDisease")
target_summary_with_num(df, "HeartDisease", "SkinCancer")


#korelasyon

corr= df.corr()

sns.set(rc={"figure.figsize": (9, 9)})
sns.heatmap(corr, cmap="RdBu")
plt.show(block=True)


