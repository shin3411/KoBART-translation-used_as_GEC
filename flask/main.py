# 참고 출처 : https://techandlife.tistory.com/30?category=927699
import argparse
from flask import Flask
from flask_restful import Api

from resources.MyAPI import MyAPI

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, required=True, help='port number')
    parser.add_argument('--debug', default=False, action='store_true', help='debug mode')
    parser.add_argument('--thread', default=False, action='store_true', help='multi-thread mode')
    parser.add_argument('--processes', type=int, default=1, help='num. of processes')
    opt = parser.parse_args()

    app = Flask(__name__)
    api = Api(app)

    api.add_resource(MyAPI, '/predict', resource_class_kwargs={'model_path': '../../kobart_binary'})

    app.run(port=opt.port, debug=opt.debug, threaded=opt.thread, processes=opt.processes)
