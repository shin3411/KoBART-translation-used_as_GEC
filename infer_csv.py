#import torch
import pandas as pd
# from kobart import get_kobart_tokenizer
# from transformers.models.bart import BartForConditionalGeneration
# from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, fbeta_score
# import argparse
from rouge_metric import Rouge

# # 인자값 받을 인스턴스 생성
# parser = argparse.ArgumentParser(
#     description='학습된 모델 파라미터를 불러올 pytorch_model.bin 폴더경로, test csv와 예측한 결과 포함된 csv파일경로 지정')
# # 인자값 등록
# parser.add_argument('--model_path', required=False, default='./kobart_binary', help='pytorch_model.bin 파일이 있는 폴더경로')
# parser.add_argumenr('--test_csv_path', required=True, help='예측할 test csv 파일 경로')
# parser.add_argumenr('--infered_test_csv_path', required=True, help='예측 결과 포함될 test csv 파일 경로')
# # 등록된 인자값을 args에 저장 (type: namespace)
# args = parser.parse_args()


# def load_model():
#     model = BartForConditionalGeneration.from_pretrained(args.model_path)
#     return model


# model = load_model()
# tokenizer = get_kobart_tokenizer()


# def inference_one_text(text):
#     input_ids = tokenizer.encode(text)
#     input_ids = torch.tensor(input_ids)
#     input_ids = input_ids.unsqueeze(0)
#     output = model.generate(input_ids, eos_token_id=1,
#                             max_length=512, num_beams=5)
#     output = tokenizer.decode(output[0], skip_special_tokens=True)
#     return output


# test_df = pd.read_csv(args.test_csv_path, index_col=0)
# test_df['infer'] = test_df['input'].apply(inference_one_text)

# print(test_df.head())
# test_df.to_csv(args.infered_test_csv_path, index=False)

test_df = pd.read_csv('./test-2.csv')

labels = []    # 실제 labels
guesses = []    # 에측된 결과

for index, row in test_df.iterrows():
    labels.append(row['output'])
    guesses.append(row['infer'])

rouge = Rouge(metrics=["rouge-n", "rouge-l", "rouge-w"], max_n=3)
scores = rouge.get_scores(guesses, labels)
print(type(scores))
print(scores)

# acc = accuracy_score(labels, guesses)
# recall = recall_score(labels, guesses, average='micro')
# precision = precision_score(labels, guesses, average='micro')
# f1 = f1_score(labels, guesses, average='micro')
# f0p5 = fbeta_score(labels, guesses, beta=0.5, average='micro')
# print(acc)
# print(recall)    # 0.42
# print(precision)     # 0.5
# print(f1)    # 0.46
# print(f0p5)

# f = open('./score.txt', 'w')
# f.write(f"accuracy_score : {acc:<5.2f},  recall_score : {recall:<5.2f}, precision_score : {precision:<5.2f}, f1_score : {f1:<5.2f}, f0.5_score : {f0p5:<5.2f} \n")
# f.close()
