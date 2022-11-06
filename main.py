# -*- coding: utf-8 -*-

import numpy as np 
import pandas as pd

from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from flask import Flask, jsonify, request
from werkzeug.exceptions import BadRequest
from flask_restx import Api, Resource, reqparse, abort

import os
import joblib

app = Flask(__name__)
api = Api(app)

model_path = 'savings/'

if not os.path.exists(model_path):
    os.mkdir(model_path)


model_params = {
    'CatBoostClassifier': {'learning_rate': 0.5, 'iterations': 200, 
                           'depth': 5},
    'LogisticRegression': {'C': 0.5, 'max_iter': 200, 'penalty':'none'}
    
}

path_to = {
    'CatBoostClassifier': f"{model_path}model1.cbm",
    'LogisticRegression': f"{model_path}model2.joblib",
    1: f"{model_path}model1.cbm",
    2: f"{model_path}model2.joblib"
}

@api.route('/models_list', endpoint='models', methods=['POST'])

class Models_List(Resource):
    def get(self):
        response = {'models':{1:"CatBoostClassifier", 2:"LogisticRegression"}}
        return response
    
api.add_resource(Models_List, '/models_list')

@api.route('/savings', endpoint='savings', methods=['GET'])

class savings(Resource):
    def get(self):
        response = {'savings': os.listdir(model_path)}
        return jsonify(response)
    
api.add_resource(savings, '/savings')

@api.route('/CatBoostClassifier',  endpoint='1', methods=['GET', 'POST'])

class CatBoostClassifier_cc(Resource):
    def get(self):
       
       response = {'model name': 'CatBoostClassifier',
                   'model id': 1,
                   'params description': {'learning_rate': 'positive real, used for reducing the gradient step',
                                          'depth': 'positive integer, max depth of the tree',
                                          'iterations': 'positive integer, num of iterations'},
                   'default params': model_params['CatBoostClassifier'],
                   }
       print(jsonify(response))
       return jsonify(response)
    @api.doc(params={'learning_rate': {'description': 'used for reducing the gradient step',
                                  'type': float, 'default': model_params['CatBoostClassifier']['learning_rate']},
                     'depth': {'description': 'max depth of the tree',
                                  'type': int, 'default': model_params['CatBoostClassifier']['depth']},
                     'iterations': {'description': 'num of iterations for training',
                                  'type': int, 'default': model_params['CatBoostClassifier']['iterations']},
                     },
             responses={200: 'Model params has been successfully set',
                        400: 'Bad request'}
             )
    def post(self):
        params = dict(request.args)
        params_pars = {k: int(v) for k, v in params.items() if k != "learning_rate"}
        if "learning_rate" in params.keys():
            params_pars["learning_rate"] = float(params["learning_rate"])
        model_params['CatBoostClassifier'].update(params_pars)
        return jsonify(model_params['CatBoostClassifier'])
    
    
@api.route('/LogisticRegression',  endpoint='2', methods=['GET', 'POST'])

class LogisticRegression_cc(Resource):
    def get(self):
       
       response = {'model name': 'LogisticRegression',
                   'model id': 2,
                   'params description': {'C': 'positive real, inverse of regularization strength',
                                          'max_iter': 'positive integer, num of iterations',
                                          'penalty': 'string, specify the norm of the penalty'},
                   'default params': model_params['LogisticRegression'],
                   }
       print(jsonify(response))
       return jsonify(response)
    @api.doc(params={'C': {'description': 'inverse of regularization strength',
                                  'type': float, 'default': model_params['LogisticRegression']['C']},
                     'max_iter': {'description': 'num of iterations for training',
                                  'type': int, 'default': model_params['LogisticRegression']['max_iter']},
                     'penalty':{'description': 'specify the norm of the penalty', 
                                'type': str, 'default': model_params['LogisticRegression']['penalty']}
                     },
             responses={200: 'Model params has been successfully set',
                        400: 'Bad request'}
             )
    def post(self):
        params = dict(request.args)
        params_pars_2 = {k: int(v) for k, v in params.items() if k != "C"}
        if "C" in params.keys():
            params_pars_2["C"] = float(params["C"])
        model_params['LogisticRegression'].update(params_pars_2)
        return jsonify(model_params['LogisticRegression'])    


class DF_builder:
    def __init__(self, gender_range, salary_range, 
                 grade_range, over_working_hours_range, employee_assessment_range):
        self.gender_range = gender_range
        self.salary_range = salary_range
        self.grade_range = grade_range
        self.over_working_hours_range = over_working_hours_range
        self.employee_assessment_range = employee_assessment_range
        
        
    def dataset(self, n_samples):
        gender = np.random.randint(self.gender_range[0], self.gender_range[1], n_samples)
        salary = np.random.uniform(*self.salary_range, n_samples)
        grade = np.random.randint(self.grade_range[0], self.grade_range[1], n_samples)
        over_working_hours = np.random.uniform(*self.over_working_hours_range, n_samples)
        employee_assessment = np.random.uniform(*self.employee_assessment_range, n_samples)
        
        data = {'gender':gender, 'salary':salary, 'grade':grade,
                'over_working_hours':over_working_hours, 
                'employee_assessment':employee_assessment}
        X = pd.DataFrame(data = data)
        y_array = gender * 500 + salary *0.3 + grade *17  - over_working_hours *70 + employee_assessment * 100
        y_array = (y_array< np.quantile(y_array, 0.08)).astype(int)
        y =pd.DataFrame(data={'target': y_array})
        return X, y


gen_set_params = {
    'train_size': 700,
    'test_size': 300,
    "gender": (0, 2),
    "salary": (0, 7000),
    "grade": (1, 11),
    "over_working_hours": (0, 100),
    "employee_assessment":(-2.0, 2.0)
}

generator = DF_builder(gen_set_params['gender'], gen_set_params['salary'], gen_set_params['grade'], 
                       gen_set_params['over_working_hours'], gen_set_params['employee_assessment'])





@api.route('/train_params', methods=['GET', 'POST'])

class TrainGen(Resource):
    def get(self):
        """Show current parameters for training set generator"""
        response = {'params generation': gen_set_params}
        return jsonify(response)

    @api.doc(params={'train_size': {'description': f'len of Train sample',
                                    'type': int, 'default': gen_set_params['train_size']},
                     'test_size': {'description': f'len os Test sample',
                                   'type': int, 'default': gen_set_params['test_size']},
                     'gender': {'description': f'duumy for emloyee genser, 0 - female, 1 - male'
                                             f'Integer',
                              'type': str, 'default': "0, 2"},
                     'salary': {'description': f'employee s monthly wage'
                                             f'float',
                              'type': str, 'default': "0.0, 7000.0"},
                     'grade': {'description': f'employee s grade position'
                                              f'Integer',
                               'type': str, 'default': "1, 11"},
                     'over_working_hours': {'description': f'hours over common schedule'
                                             f'float',
                              'type': str, 'default': "0.0, 100.0"},
                     'employee_assessment': {'description': f'employee month assessment'
                                             f'float',
                              'type': str, 'default': "-2.0, 2.0"}})
    @api.doc(responses={200: 'Train params has been successfully set',
                        400: 'Bad request'})
    def post(self):
        global gen_set_params, generator
        response = gen_set_params
        params = dict(request.args)
        for p in params:
            params[p] = list(map(float, params[p].split(', ')))
        response.update(params)

        gen_set_params = response
        generator = DF_builder(gen_set_params['gender'], gen_set_params['salary'], gen_set_params['grade'], 
                               gen_set_params['over_working_hours'], gen_set_params['employee_assessment'])

        return jsonify(response)


@api.route('/model_training', endpoint='train', methods=['POST'])
class ModelTraining(Resource):
    @api.doc(params={'model_id': {'description': ' id is 1 for CatBoostClassifier and 2 for LogisticRegression',
                                  'type': int, 'default': 1}},
             responses={200: 'Model has been successfully trained and saved',
                        400: 'Bad request'}
             )
    def post(self):

        model_id = int(request.args.get('model_id'))
        
        X_train, y_train = generator.dataset(gen_set_params['train_size'])
        X_test, y_test = generator.dataset(gen_set_params['test_size'])
        if model_id == 1:
            params = model_params['CatBoostClassifier']
            model_cbc = CatBoostClassifier(learning_rate = params['learning_rate'],
                                           iterations = params['iterations'],
                                           logging_level = 'Silent',
                                           max_depth = params['depth'])
            model_cbc.fit(X_train, y_train)
            model_cbc.save_model(path_to['CatBoostClassifier'])
            score = f1_score(y_test, model_cbc.predict(X_test), zero_division=1)
            
            response = {"result": "CatBoostClassifier model results:",
                        "score": score,
                        "params": model_params['CatBoostClassifier']}

            

        if model_id == 2:
            params = model_params['LogisticRegression']
            model_lr = LogisticRegression(C = params['C'], max_iter = params['max_iter'], 
                                       penalty = params['penalty'])
            model_lr.fit(X_train, y_train)

            joblib.dump(model_lr, path_to['LogisticRegression'])
            score = f1_score(y_test, model_lr.predict(X_test))
            response = {"result": "LogisticRegression model results:",
                        "score": score,
                        "params": model_params['LogisticRegression']}

        return jsonify(response)
    

class DeleteSavings(Resource):
    @api.doc(params={'model_id': {'description': ' id is 1 for CatBoostClassifier and 2 for LogisticRegression',
                                  'type': int, 'default': 1}},
             responses={200: 'Model has been successfully deleted',
                        400: 'Bad request'}
             )
    def delete(self):
        print(request.args)
        model_id = int(request.args.get('model_id'))
        if model_id == 1:
            os.remove(path_to['CatBoostClassifier'])

        elif model_id == 2:
            os.remove(path_to['LogisticRegression'])
        return jsonify(f"The save of model {model_id} "
                           f"has been deleted!")
    
data_prop = {
    'type': 'object',
    'properties': {
        'gender': {
            'type': 'array',
            'items': {'type': 'number'},
            'minItems': 1,
        },
        'salary': {
            'type': 'array',
            'items': {'type': 'number'},
            'minItems': 1,
        },
        'grade': {
            'type': 'array',
            'items': {'type': 'number'},
            'minItems': 1,
        },
        'over_working_hours': {
            'type': 'array',
            'items': {'type': 'number'},
            'minItems': 1,
        },
        'employee_assessment': {
            'type': 'array',
            'items': {'type': 'number'},
            'minItems': 1,
        },

    },
    'add_Properties': False,
    'required': ['gender','salary','grade', 'over_working_hours','employee_assessment'],
}
request_data = api.schema_model('data', data_prop)
        
@api.route('/predict/<int:model_id>', methods=['POST'])
class Predict(Resource):
    @api.doc(params={'model_id': {'description': 'id is 1 for CatBoostClassifier and 2 for LogisticRegression',
                                  'type': int, 'default': 1}})
    @api.expect(request_data, validate=True)
    def post(self, model_id):
        
        features = ['gender','salary','grade', 'over_working_hours','employee_assessment']
        data =request.json
        dict_making = {f: data[f] for f in features}
        X = pd.DataFrame(data = dict_making)
        
        path = path_to[model_id]
       
        if model_id == 1:
            model_cbc = CatBoostClassifier()
            model_cbc.load_model(path)
            prediction = model_cbc.predict(X).astype(int)
            model_info = {"id": 1,
                            "name": 'CatBoostClassifier',
                            "params": model_params['CatBoostClassifier']}
            
        if model_id == 2:
            model = joblib.load(path_to['LogisticRegression'])
            prediction = model.predict(X).astype(int)
            model_info = {"id": 2,
                            "name": 'LogisticRegression',
                            "params": model_params['LogisticRegression']}
        return jsonify({"model" : model_info,
                            "prediction": list(prediction)})
        
if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5005)