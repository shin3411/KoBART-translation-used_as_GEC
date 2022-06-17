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
    
    @AIpredict.response(200, 'Success', message_fields)
    @AIpredict.response(500, 'Failed')
    def get(self):
        """단순 /predict endpoint 접속 확인용입니다."""
        return { 'message' : '/predict endpoint에 대한 api입니다.', 'current_path' : os.getcwd() }, 200
    
    def delete(self):
        """구현 불필요, 사용할 수 없습니다."""
        raise WrongMethodError
        
    def put(self):
        """구현 불필요, 사용할 수 없습니다."""
        raise WrongMethodError

    @AIpredict.expect(sen_fields)
    @AIpredict.response(201, 'Success', sen_fields)
    @AIpredict.response(500, 'Failed')
    def post(self):
        ''' 오문장을 넣으면 GEC모델로 생성한 문장을 출력해 줍니다.'''
        try:
            parser = reqparse.RequestParser()

            # name(첫 인자) – Either a name or a list of option strings, e.g. foo or -f, –foo.
            # dest – The name of the attribute to be added to the object returned by parse_args().    
            parser.add_argument('sentence', dest='sentence', type=str, help='input sentence on the model') 

            # 정의되지 않은 인수 포함 시 400 Error 발생
            self.args = parser.parse_args(strict=True)

            sen = self.args.get('sentence', None)
            if sen is None:
                raise NoneArgumentError
            

            def inference_one_text(text):
                input_ids = tokenizer.encode(text)
                input_ids = torch.tensor(input_ids)
                input_ids = input_ids.unsqueeze(0)
                output = model.generate(input_ids, eos_token_id=1,max_length=512, num_beams=5)
                output = tokenizer.decode(output[0], skip_special_tokens=True)
                return output
            
            return { 'sentence' : inference_one_text(sen) }, 201
        except Exception as e:
            return { 'error' : str(e) }, 500
