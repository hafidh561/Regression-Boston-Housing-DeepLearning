from flask import Flask
from flask_restful import Api, Resource, reqparse
from flask_cors import CORS
from model import predict_model

app = Flask(__name__)
api = Api(app)
CORS(app)

api_args = reqparse.RequestParser()
api_args.add_argument('crim', type=float,
                      help='per capita crime rate by town', required=True)
api_args.add_argument('zn', type=float,
                      help='proportion of residential land zoned for lots over 25,000 sq.ft', required=True)
api_args.add_argument('indus', type=float,
                      help='proportion of non-retail business acres per town', required=True)
api_args.add_argument('chas', type=float,
                      help='Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)', required=True)
api_args.add_argument('nox', type=float,
                      help='nitrogen oxides concentration (parts per 10 million)', required=True)
api_args.add_argument('rm', type=float,
                      help='average number of rooms per dwelling', required=True)
api_args.add_argument('age', type=float,
                      help='proportion of owner-occupied units built prior to 1940', required=True)
api_args.add_argument('dis', type=float,
                      help='weighted mean of distances to five Boston employment centres', required=True)
api_args.add_argument('rad', type=float,
                      help='index of accessibility to radial highways', required=True)
api_args.add_argument('tax', type=float,
                      help='full-value property-tax rate per \$10,000', required=True)
api_args.add_argument('ptratio', type=float,
                      help='pupil-teacher ratio by town', required=True)
api_args.add_argument('black', type=float,
                      help='1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town', required=True)
api_args.add_argument('lstat', type=float,
                      help='lower status of the population (percent)', required=True)


class Predict(Resource):
    def post(self):
        args = api_args.parse_args()
        predict = predict_model(
            args['crim'],
            args['zn'],
            args['indus'],
            args['chas'],
            args['nox'],
            args['rm'],
            args['age'],
            args['dis'],
            args['rad'],
            args['tax'],
            args['ptratio'],
            args['black'],
            args['lstat'],
        )
        return {'predict': round(float(predict), 2)}, 201


api.add_resource(Predict, '/')

if __name__ == '__main__':
    app.run()
