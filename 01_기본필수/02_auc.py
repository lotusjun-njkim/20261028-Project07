# """실습 02: AUC에는 0/1 판정이 아니라 실패 클래스의 연속 점수를 넣습니다."""
# from sklearn.metrics import roc_auc_score, roc_curve
# # 정상적인 원리 예제 4건입니다. 수업용 파일의 test 결과가 아닙니다.
# y = [0, 0, 1, 1]
# score = [0.10, 0.40, 0.35, 0.80]
# # TODO B02: roc_auc_score에 실제값과 연속 점수를 넣으세요.
# auc = 0.0
# # TODO 끝
# fpr, tpr, thresholds = roc_curve(y, score)
# print("FPR:", fpr.round(2).tolist())
# print("TPR:", tpr.round(2).tolist())
# print(f"AUC: {auc:.2f}")
# print("양성 점수 0.35·0.80과 음성 점수 0.10·0.40을 네 번 비교하세요.")



"""실습 02: AUC에는 0/1 판정이 아니라 실패 클래스의 연속 점수를 넣습니다."""
# scikit-learn의 metrics 모듈에서 AUC 계산 함수와 ROC 곡선 데이터 생성 함수를 가져옵니다.
from sklearn.metrics import roc_auc_score, roc_curve

# 정상적인 원리 예제 4건입니다. 수업용 파일의 test 결과가 아닙니다.
# y: 실제값 (0 = 정상/음성, 1 = 실패/양성)
y = [0, 0, 1, 1]
# score: 모델이 예측한 실패(1)일 확률 또는 연속 점수
score = [0.10, 0.40, 0.35, 0.80]

# TODO B02: roc_auc_score에 실제값과 연속 점수를 넣으세요.
# 실제값 y와 연속 점수 score를 순서대로 입력하여 AUC(Area Under the ROC Curve)를 계산합니다.
auc = roc_auc_score(y, score)
# TODO 끝

# roc_curve: 임계값(Threshold) 변화에 따른 FPR(거짓 양성 비율)과 TPR(참 양성 비율)을 계산합니다.
fpr, tpr, thresholds = roc_curve(y, score)

print("FPR:", fpr.round(2).tolist())          # 각 임계값에서의 False Positive Rate 출력
print("TPR:", tpr.round(2).tolist())          # 각 임계값에서의 True Positive Rate (재현율) 출력
print(f"AUC: {auc:.2f}")                      # 최종 AUC 점수 출력 (0.5 ~ 1.0 사이의 값)

print("양성 점수 0.35·0.80과 음성 점수 0.10·0.40을 네 번 비교하세요.")
