# 참고 출처 : https://techandlife.tistory.com/30?category=927699
import argparse
from flask import Flask
from flask_restx import Api, Resource

from resources import AIpredict
import os


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, required=True, help='port number')
    parser.add_argument('--debug', default=False, action='store_true', help='debug mode')
    parser.add_argument('--thread', default=False, action='store_true', help='multi-thread mode')
    parser.add_argument('--processes', type=int, default=1, help='num. of processes')
    opt = parser.parse_args()

    app = Flask(__name__)
    '''
    - version: API Server의 버전을 명시합니다.
    - title: API Server의 이름을 명시합니다.
    - description: API Server의 설명을 명시합니다.
    - terms_url: API Server의 Base Url을 명시합니다.
    - contact: 제작자 E-Mail 등을 삽입합니다.
    - license: API Server의 라이센스를 명시 합니다.
    - license_url: API Server의 라이센스 링크를 명시 합니다.
    '''
    api = Api(
        app,
        version='0.1',
        title='GEC_model API Server',
        description='API Server gets grammatically errored sentence, and outputs corrected sentence.',
        term_url='/',
        contact='rladudrhs341@gmail.com',
        license='MIT'
        )

    api.add_namespace(ns=AIpredict, path='/predict')
    

    app.run(port=opt.port, debug=opt.debug, threaded=opt.thread, processes=opt.processes, host='0.0.0.0')
