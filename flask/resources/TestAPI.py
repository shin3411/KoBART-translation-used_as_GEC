from flask import Flask
from flask_restful import Resource, reqparse


class TEST(Resource):
    # POST 요청만 수신
    def post(self):
        try:
            # 1. 인자값 수신
            parser = reqparse.RequestParser()
            parser.add_argument('value1', required=False, type=str, help='value 1')
            parser.add_argument('value2', required=True, type=str, help='value 2')
            # 파서가 정의하지 않은 인수 포함 시, 400 Error 발생
            args = parser.parse_args(strict=True)

            # 결과 반환
            return {'result': args['value1']}
        
        except Exception as e:
            return {'error': str(e)}
