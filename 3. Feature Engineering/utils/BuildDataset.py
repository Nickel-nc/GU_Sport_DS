import time
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

value_string_template = '\033[91m[[value]]\033[0m'



def prepare_dummies(data):
    
    """
    One-Hot преобразование всех категориальных признаков
    
    Parameters
    ----------
    data: pd.DataFrame
        Датасет, внутри которого будет проводиться преобразование
    """
    
    for cat_colname in data.select_dtypes(include='object').columns[:]:
        data = pd.concat([data, pd.get_dummies(data[cat_colname], prefix=cat_colname)], axis=1)
        data.drop(cat_colname, axis=1, inplace=True)
        
    return data

def encode_labels(data):
    
    """
    Нормализация категориальных признаков на основе Label Encoding
    """
    
    for cat_colname in data.select_dtypes(include='object').columns:
        le = LabelEncoder()
        transformed = le.fit_transform(data[cat_colname].fillna('null'))
        data[cat_colname] = transformed
    return data
    


def build_raw_dataset(raw_data_path='raw_data/',
                      dataset_path='dataset/',
                      id_col = 'APPLICATION_NUMBER',
                      mode='train'):
    
    """
    Сбор датасета из сырых таблиц. Режимы: 'train' | 'test'
    Соединяет признаки по ID полю из левой таблицы 
    Не обрабатывает пропуски, выбросы и несвязанные поля
    
    Parameters
    ----------
    raw_data_path: str
        Путь к каталогу с данными
        
    dataset_path: str
        Путь к каталогу, куда будет сохранен готовый датасет
       
    id_col: str
        Наименование ID-столбца, по которому будут соединяться датасеты
        
    mode: str
        Доступны режимы: 'train' | 'test'
        Сбор датасета в зависимости от его типа

    Returns
    -------
    dataset: pandas.DataFrame
        Готовый train | test датасет
    """
    
    assert mode in ['train', 'test'], 'Error. Bad mode chosen'
    
    mode_formatter = value_string_template.replace('[[value]]', mode)
    print(f"Building dataset in {mode_formatter} mode...", end='')
    
    if mode == 'train':
        data = pd.read_csv(raw_data_path + 'train.csv', encoding='utf-8')
    else:
        data = pd.read_csv(raw_data_path + 'test.csv', encoding='utf-8')

    # Load data
    test = pd.read_csv(raw_data_path + 'test.csv', encoding='utf-8')
    bki = pd.read_csv(raw_data_path + 'bki.csv', encoding='utf-8')
    client_profile = pd.read_csv(raw_data_path + 'client_profile.csv', encoding='utf-8')
    payments = pd.read_csv(raw_data_path + 'payments.csv')
    history = pd.read_csv(raw_data_path + 'applications_history.csv', encoding='utf-8')
    
    # Merge features from each data file
    for df in [bki, client_profile, payments, history]:
        data = pd.merge(data, df, how='left', on=id_col)
    
    print('\033[94mDone\033[0m') # adjust message color
    
    return data

def preprocess_data(mode='train',
                    prepare_catecorical=True,
                    id_col = 'APPLICATION_NUMBER',
                    please_no_dummies=True):
    
    start_t = time.time()   
    data = build_raw_dataset(mode=mode, id_col=id_col)
    print(f"Preprocessing starting...", end='')
    if prepare_catecorical:
        print(f"Preparing categories...", end='')
        if please_no_dummies:
            data = encode_labels(data)
        else:
            data = prepare_dummies(data)
        
    print(f"Grouping features...", end='')
    data = data.groupby(id_col).mean().reset_index()
    print('\033[94mDone\033[0m')
    print(f'Run time: {round(time.time() - start_t, 1)} sec')
    
    return data