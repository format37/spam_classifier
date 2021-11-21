import pandas as pd
import re
from catboost import CatBoostClassifier

def data_augmentation(df):
    df['latin'] = df['name'].apply(lambda x: len(re.findall(r'[a-zA-Z]', x)))
    df['cyrilic'] = df['name'].apply(lambda x: len(re.findall(r'[а-яА-Я]', x)))
    df['numbers'] = df['name'].apply(lambda x: len(re.findall(r'[0-9]', x)))
    df['others'] = df['name'].apply(lambda x: len(re.findall(r'[^\x00-\x7Fа-яА-ЯёЁ0-9]', x)))
    df['others_l'] = df['name'].apply(lambda x: len(re.findall(r'[^\x00-\x7Fа-яА-ЯёЁ0-9]',x[:int(len(x)/2)])))
    df['others_r'] = df['name'].apply(lambda x: len(re.findall(r'[^\x00-\x7Fа-яА-ЯёЁ0-9]',x[int(len(x)/2):])))
    df['lc'] = df['cyrilic']/df['latin']
    return df

def is_spam(name):
    model = CatBoostClassifier()
    model.load_model('catboost_spam.model')
    validation = data_augmentation(pd.DataFrame([name], columns = ['name']))
    pred = model.predict(validation)
    return pred[0]

print(is_spam('💝💝💝 Г0РЯЧИЕ_ZНAK0МСТBA_TГ 💝💝💝'))
