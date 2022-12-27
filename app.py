from flask import Flask
from flask_restx import Api, Resource, reqparse
import json
import os
from modules.model import read_data, fitting, predicting

app = Flask(__name__)
api = Api(app)


params_to_fit = reqparse.RequestParser()
params_to_fit.add_argument('model_id',
                                 help='ID модели в целочисленном формате, пример: 13', 
                                 required=True, location='form')
params_to_fit.add_argument('model_name',
                                 help='Название алгоритмов обучения', 
                                 choices=['LogisticRegression','DecisionTreeClassifier'],
                                 required=True, location='form')
params_to_fit.add_argument('model_params',
                                 help='Параметры для обучения модели, пример {"max_depth": 5}',
                                 location='form')


params_to_predict = reqparse.RequestParser()
params_to_predict.add_argument('model_id', help='Выбор модели для предсказания через ID', 
                               required=True, location='form')


params_to_delete = reqparse.RequestParser()
params_to_delete.add_argument('model_id', help='Выбор модели для удаления через ID', 
                              required=True, location='form')


@api.route('/available_models', methods=['GET'], doc={'description': 'Список алгоритмов доступных для обучения'})
class Available_Alrorithms(Resource):
    @api.response(200, 'OK')
    @api.response(404, 'NOT FOUND')
    def get(self):
        return 'Доступные алгоритмы обучения: LogisticRegression, DecisionTreeClassifier', 200


@api.route('/trained_models', methods=['GET'], doc={'description': 'Список обученных моделей'})
class Trained_Models(Resource):
    @api.response(200, 'OK')
    @api.response(404, 'NOT FOUND')
    def get(self):
        with open("saved_models_params.json", "r") as jsonFile:
            trained_models_params = json.load(jsonFile)
        if trained_models_params == {}:
            return 'Нет доступных обученных моделей', 200
        
        return trained_models_params, 200


@api.route('/fit_model', methods=['POST'], doc={'description': 'Обучить и сохранить модель с выбранными параметрами'})
class Fit_Model(Resource):
    @api.expect(params_to_fit)
    @api.response(200, 'OK')
    @api.response(400, 'BAD REQUEST')
    @api.response(404, 'NOT FOUND')
    def post(self):
        args = params_to_fit.parse_args()
        model_id = args.model_id
        model_name = args.model_name
        model_params = {} if args.model_params is None else json.loads(args.model_params.replace("'", "\""))
        if model_id + '.pkl' in os.listdir('savings/saved_models/'):
            return 'Модель с таким ID уже существует, выберите другое ID', 400
        
        data, target = read_data('savings/datasets/turnover.csv', 
                                 target_name = 'event', 
                                 feature_list =['stag', 'age', 'extraversion',
                                                'independ', 'selfcontrol','anxiety',	
                                                'novator']
                                 )
        status, param = fitting(data, target, model_name, model_id, model_params)
        if param != 'no_param_missing':
            return status, 400

        model_params = 'default'  if model_params ==  {} else model_params
        with open("saved_models_params.json", "r") as jsonFile:
            data = json.load(jsonFile)
                    
        data[model_id] = {'model_name': model_name, 'model_params': model_params}
                
        with open("saved_models_params.json", "w") as jsonFile:
            json.dump(data, jsonFile)
                    
        return 'Модель успешно обучена', 200


@api.route('/predict', methods=['POST'], doc={'description': 'Предсказать значение'})
class Predict(Resource):
    @api.expect(params_to_predict)
    @api.response(200, 'OK')
    @api.response(400, 'BAD REQUEST')
    @api.response(404, 'NOT FOUND')
    def post(self):
        args = params_to_predict.parse_args()
        model_id = args.model_id
        if model_id + '.pkl' not in os.listdir('savings/saved_models/'):
            return f'Обученной модели с ID = {model_id} не существует', 400
        
        data, target = read_data('savings/datasets/turnover.csv', 
                                 target_name = 'event', 
                                 feature_list =['stag', 'age', 'extraversion',
                                                'independ', 'selfcontrol','anxiety',	
                                                'novator']
                                 )
        prediction = predicting(model_id, data)
        
            
        return json.dumps({f'ID = {model_id}': prediction.astype(int).tolist()})


@api.route('/delete_model')
class Delete_Model(Resource):
    @api.expect(params_to_delete)
    @api.response(200, 'OK')
    @api.response(400, 'BAD REQUEST')
    @api.response(404, 'NOT FOUND')
    def delete(self):
        args = params_to_delete.parse_args()
        model_id = args.model_id
        if model_id + '.pkl' not in os.listdir('savings/saved_models/'):
            return f'Обученной модели с ID = {model_id} не существует', 400
        
        with open("saved_models_params.json", "r") as file:
            data = json.load(file)
            data.pop(model_id)
        with open("saved_models_params.json", "w") as jsonFile:
            json.dump(data, jsonFile)
        os.remove('savings/saved_models/' + str(model_id) + '.pkl')
        
        
        return f'Модель с ID = {model_id} удалена', 200


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
