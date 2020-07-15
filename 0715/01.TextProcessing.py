import numpy as np

samples = ['The cat sat on the mat.', 'The dog ate my homework.']
token_idx = {} # 빈 딕셔너리 생성. ex)token_idx['the'] = 1


for sample in samples:
    for w in sample.split():
        if w not in token_idx:
            token_idx[w] = len(token_idx)+1

# print(token_idx)
# {'The': 1, 'cat': 2, 'sat': 3, 'on': 4, 'the': 5, 'mat.': 6, 'dog': 7, 'ate': 8, 'my': 9, 'homework.': 10}


max_length = 10 # 사용자 지정. 문자 추출할 최대 길이

# 단어 인식 작업
result = np.zeros((len(samples), max_length, max(token_idx.values())+1))
# print(result.shape)     #  (2, 10, 11)

for i, sample in enumerate(samples):
    for j, w in list(enumerate(sample.split()))[:max_length]:
        idx = token_idx.get(w)
        result[i, j, idx] = 1    # samples 배열에서 해당 문자 idx, 문장 내 해당 단어 idx, 사전 해당단어 idx

print(token_idx)
# {'The': 1, 'cat': 2, 'sat': 3, 'on': 4, 'the': 5, 'mat.': 6, 'dog': 7, 'ate': 8, 'my': 9, 'homework.': 10}
