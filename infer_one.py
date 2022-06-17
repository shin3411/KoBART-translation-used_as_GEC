import torch
from kobart import get_kobart_tokenizer
from transformers.models.bart import BartForConditionalGeneration

def load_model():
    model = BartForConditionalGeneration.from_pretrained('./kobart_binary')
    return model

model = load_model()
tokenizer = get_kobart_tokenizer()

def inference_one_text(text):
    input_ids = tokenizer.encode(text)
    input_ids = torch.tensor(input_ids)
    input_ids = input_ids.unsqueeze(0)
    output = model.generate(input_ids, eos_token_id=1,max_length=512, num_beams=5)
    output = tokenizer.decode(output[0], skip_special_tokens=True)
    return output

from time import time, sleep

org_text = []
org_text.append("내는 한국어 공부 좋아합니다.")
org_text.append("그눔 참 숙제 날래 잘했드래요.")
org_text.append("그 아는 오데를 그리 가려고 하노?")
org_text.append("지그믄 너무 조려서 집에 가고 싶다.")
org_text.append('이게 자 나오까 걱정되고 잘 나오면 좋다.')
org_text.append("안되. 그렇게 돼면 기계가 고장날 수 읻어!.")
org_text.append("BTS에서 지민 너무 살앙하고 한국가 보고 시퍼요")
org_text.append("오늘는 날씨가 너무 졷스니다.")
org_text.append("구멍에 들깨를 너으니 자동으로 기름이 짜여졌다.")

for text in org_text:
    start = time()
    result = inference_one_text(text)
    print(f'{text:<30s} => {result:<30s}')
    end = time()
    print('time elapsed:', end - start)

