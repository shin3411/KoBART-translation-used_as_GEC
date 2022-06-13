import torch
import pandas as pd
from kobart import get_kobart_tokenizer
from transformers.models.bart import BartForConditionalGeneration
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, fbeta_score


def load_model():
    model = BartForConditionalGeneration.from_pretrained('./kobart_binary')
    return model


model = load_model()
tokenizer = get_kobart_tokenizer()


def inference_one_text(text):
    input_ids = tokenizer.encode(text)
    input_ids = torch.tensor(input_ids)
    input_ids = input_ids.unsqueeze(0)
    output = model.generate(input_ids, eos_token_id=1,
                            max_length=512, num_beams=5)
    output = tokenizer.decode(output[0], skip_special_tokens=True)
    return output


test_df = pd.read_csv('./test-1.csv', index_col=0)
test_df['infer'] = test_df['input'].apply(inference_one_text)

print(test_df.head())
test_df.to_csv('./test-2.csv')

labels = []    # 실제 labels
guesses = []    # 에측된 결과

for index, row in test_df.iterrows():
    labels.append(row['output'])
    guesses.append(row['infer'])

acc = accuracy_score(labels, guesses)
recall = recall_score(labels, guesses, average='micro')
precision = precision_score(labels, guesses, average='micro')
f1 = f1_score(labels, guesses, average='micro')
f0p5 = fbeta_score(labels, guesses, beta=0.5, average='micro')
print(acc)
print(recall)    # 0.42
print(precision)     # 0.5
print(f1)    # 0.46
print(f0p5)

f = open('./ACC&R&P&F_score.txt', 'w')
f.write(f"accuracy_score : {acc:<5.2f},  recall_score : {recall:<5.2f}, precision_score : {precision:<5.2f}, f1_score : {f1:<5.2f}, f0.5_score : {f0p5:<5.2f} \n")
f.close()
