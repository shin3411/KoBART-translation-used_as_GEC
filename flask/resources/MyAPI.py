from flask_restful import Resource, reqparse
import torch
from kobart import get_kobart_tokenizer
from transformers.models.bart import BartForConditionalGeneration
from resources.custom_errors import WrongMethodError, NoneArgumentError
import os

class MyAPI(Resource):
    def __init__(self, **kwargs):
        # Create a request parser
        parser = reqparse.RequestParser()
        parser.add_argument('TEXT', dest='sentence', type=str) # 첫번째 인자는 Argument 인스턴스의 이름인듯... 사실 뭔지 잘 모르겠다.
        self.args = parser.parse_args(strict=True)
        self.model_path = kwargs['model_path']

    def get(self):
        return { 'current_path' : os.getcwd() }
    
    def delete(self):
        raise WrongMethodError
        
    def put(self):
        raise WrongMethodError

    def post(self):
        ''' prediction '''
        try:
            sen = self.args.get('sentence', None)
            if sen is None:
                raise NoneArgumentError
            
            model = BartForConditionalGeneration.from_pretrained(self.model_path)
            tokenizer = get_kobart_tokenizer()

            def inference_one_text(text):
                input_ids = tokenizer.encode(text)
                input_ids = torch.tensor(input_ids)
                input_ids = input_ids.unsqueeze(0)
                output = model.generate(input_ids, eos_token_id=1,max_length=512, num_beams=5)
                output = tokenizer.decode(output[0], skip_special_tokens=True)
            
            return inference_one_text(sen)
        except Exception as e:
            return { 'error' : str(e) }
