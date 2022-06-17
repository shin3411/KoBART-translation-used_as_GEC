from flask_restx import Resource, Namespace, fields, reqparse
import torch
from kobart import get_kobart_tokenizer
from transformers.models.bart import BartForConditionalGeneration
from resources import WrongMethodError, NoneArgumentError
import os

model_path = os.path.dirname(os.path.realpath(__file__)) + '/../../kobart_binary'
model = BartForConditionalGeneration.from_pretrained(model_path)
tokenizer = get_kobart_tokenizer()

# **특정 endpoint를 포함한 경로**를 만들어내기 위해, 'endpoint명' 추가
# endpoint : http://localhost/Todo 면 localhost는 host 고, Todo는 endpoint 라고 함
AIpredict = Namespace(
    name='AIpredicts',
    description='AI 모델로 GEC한 결과를 내보내기 위해 사용하는 API',
)

"""
Namespace.Model()
- **입력, 출력에 대한 스키마를 나타내는 객체**
- flask_restx 내의 field 클래스를 이용하여 설명, 필수 여부, 예시를 넣을 수 있음
- Namespace.inherit()을 이용하여 Namespace.model() 을 상속 받을 수 있음
"""
sen_fields = AIpredict.model('Sentence', { # Model 객체 생성
    'sentence' : fields.String(description='Incorrect sentence', required=True, example='안돼는건 안니야')
})
message_fields = AIpredict.model('msg', {
    'message' : fields.String(description='해당 endpoint 문구', example='/predict endpoint에 대한 api'),
    'current_path' : fields.String(description='해당 작업 위치', example='/home/elice/')
})

@AIpredict.route('')
class AIapi(Resource):
    def __init__(self):
        # Create a request parser
        parser = reqparse.RequestParser()
        parser.add_argument('TEXT', dest='sentence', type=str, help='input sentence on the model') # 첫번째 인자는 Argument 인스턴스의 이름인듯... 사실 뭔지 잘 모르겠다.
        self.args = parser.parse_args(strict=True) # 정의되지 않은 인수 포함 시 400 Error 발생

    @AIpredict.response(200, 'Success', message_fields)
    @AIpredict.response(500, 'Failed')
    def get(self):
        return { 'message' : '"/predict" endpoint에 대한 api입니다.', 'current_path' : os.getcwd() }, 200
    
    def delete(self):
        raise WrongMethodError
        
    def put(self):
        raise WrongMethodError

    @AIpredict.expect(sen_fields)
    @AIpredict.response(201, 'Success', sen_fields)
    @AIpredict.response(500, 'Failed')
    def post(self):
        ''' prediction '''
        try:
            sen = self.args.get('sentence', None)
            if sen is None:
                raise NoneArgumentError
            

            def inference_one_text(text):
                input_ids = tokenizer.encode(text)
                input_ids = torch.tensor(input_ids)
                input_ids = input_ids.unsqueeze(0)
                output = model.generate(input_ids, eos_token_id=1,max_length=512, num_beams=5)
                output = tokenizer.decode(output[0], skip_special_tokens=True)
            
            return { 'sentence' : inference_one_text(sen) }, 201
        except Exception as e:
            return { 'error' : str(e) }, 500
