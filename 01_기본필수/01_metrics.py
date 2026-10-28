# """실습 01: 실제 실패 6건인 손계산 예제입니다. 아직 모델을 학습하지 않습니다."""
# from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
# # 행은 실제, 열은 예측입니다. 1은 좋은 결과가 아니라 관심 클래스인 '실패'입니다.
# y = [0]*14 + [1]*6
# pred = [0]*10 + [1]*4 + [0]*4 + [1]*2
# print("혼동행렬(행 실제 / 열 예측, 성공0·실패1)")
# print(confusion_matrix(y, pred, labels=[0, 1]))
# # TODO B01: 네 지표 함수에 실제값 y와 예측값 pred를 순서대로 넣으세요. 
# zero_division=0은 분모가 0일 때 0으로 표시하는 정책입니다.
# acc = precision = recall = f1 = 0.0
# # TODO 끝
# print(f"Accuracy={acc:.3f}, Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}")
# print("실제 실패 6건 중 발견: 2건")



"""실습 01: 실제 실패 6건인 손계산 예제입니다. 아직 모델을 학습하지 않습니다.""" 
# scikit-learn의 metrics 모듈에서 혼동행렬 및 분류 평가지표 함수들을 가져옵니다.
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score 

# [데이터 정의] 행은 실제값, 열은 예측값입니다. 
# 0은 성공(정상 클래스), 1은 우리가 관심을 두고 추적하는 '실패(관심 클래스)'를 의미합니다.
# 실제값(y): 성공(0) 14건 + 실패(1) 6건 = 총 20건
y = [0]*14 + [1]*6 
# 예측값(pred): 성공 예측 10건 + 실패 예측 4건 + 성공 예측 4건 + 실패 예측 2건
pred = [0]*10 + [1]*4 + [0]*4 + [1]*2 

print("혼동행렬(행 실제 / 열 예측, 성공0·실패1)")
# confusion_matrix: 실제 클래스와 예측 클래스의 분포를 표(행렬) 형태로 보여줍니다.
# labels=[0, 1]을 지정하여 0번 클래스(성공)가 1행/1열, 1번 클래(실패)가 2행/2열에 위치하도록 고정합니다.
print(confusion_matrix(y, pred, labels=[0, 1])) 

# TODO B01: 네 지표 함수에 실제값 y와 예측값 pred를 순서대로 넣으세요. 
# zero_division=0은 분모가 0이 되어 수학적 오류가 발생할 때 에러 대신 0으로 출력하라는 설정입니다.
acc = accuracy_score(y, pred)                          # 전체 데이터 중 올바르게 예측한 비율 (정확도)
precision = precision_score(y, pred, zero_division=0)  # 실패라고 예측한 것 중 진짜 실패인 비율 (정밀도)
recall = recall_score(y, pred, zero_division=0)        # 실제 실패한 것 중 모델이 찾아낸 비율 (재현율)
f1 = f1_score(y, pred, zero_division=0)                # 정밀도와 재현율의 균형을 맞춘 조화평균 (F1-스코어)
# TODO 끝 

# 계산된 네 가지 핵심 분류 지표를 소수점 셋째 자리까지 출력합니다.
print(f"Accuracy={acc:.3f}, Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}") 
print("실제 실패 6건 중 발견: 2건")
