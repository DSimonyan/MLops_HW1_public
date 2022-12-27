import pandas as pd
import pickle
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

def read_data(file_name: str, target_name: str, feature_list: list):
    """
    Обработка данных 
    Param:
    file_name: str
    target_name: str 
    feature_list: list
    Return:
    data: pd.DataFrame
    target: pd.Series
    """
    df = pd.read_csv(file_name, sep=',')
    target = df[target_name]
    data = df[feature_list]
    return data, target

def fitting(train_data: pd.DataFrame, train_target: pd.Series, 
            model_name: str, model_id: int, model_params: dict):
    """
    На вход принимает набор признаков, целевую переменную, 
    название, id и параменты модели.
    Param:
    train_data: pd.DataFrame
    train_target: pd.Series
    model_name: str
    id_model: int
    model_params: dict
    """
    
    models = {
                'DecisionTreeClassifier': DecisionTreeClassifier(),
                'LogisticRegression': LogisticRegression()
              }
    
    for param in model_params:
        if param not in models[model_name].get_params().keys():
            return "Error! Parametr missing:  %s."%(param), param
    
    
    model = models[model_name].set_params(**model_params)
    model.fit(train_data, train_target)
    
    with open('savings/saved_models/' + str(model_id) + '.pkl', 'wb') as file:
        pickle.dump(model, file)
    
    return "Model number %s successfully fitted!"%(model_id), 'no_param_missing'


def predicting(model_id: int, test_data: pd.DataFrame):
    """
    На вход принимает id модели и данные для предсказания
    Param:
    test_data: pd.DataFrame
    model_id: int
    
    Return:
    prediction: pd.Series
    """

    with open('savings/saved_models/' + str(model_id) + '.pkl', 'rb') as file:
        model = pickle.load(file)
    prediction = model.predict(test_data)
    return prediction