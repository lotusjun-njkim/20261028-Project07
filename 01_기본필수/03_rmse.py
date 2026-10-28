# """실습 03: 같은 회귀 문제에서도 주 지표에 따라 후보의 순서가 달라집니다."""
# from sklearn.metrics import mean_absolute_error, root_mean_squared_error
# y = [100, 200, 300, 400]  # 실제 실행시간(ms)
# predictions = {"A": [100, 200, 300, 480], "B": [130, 230, 330, 430]}
# for name, pred in predictions.items():
    # mae = mean_absolute_error(y, pred)
    # # TODO B03: 같은 y·pred로 RMSE를 계산하세요. 함수 이름의 root는 제곱근을 뜻합니다.
    # rmse = 0.0
    # # TODO 끝
    # print(f"{name}: MAE={mae:.1f}ms, RMSE={rmse:.1f}ms")
# print("실행시간 단위 ms를 함께 적으세요. 교차검증은 06번에서 이어집니다.")



"""실습 03: 같은 회귀 문제에서도 주 지표에 따라 후보의 순서가 달라집니다."""
# scikit-learn의 metrics 모듈에서 MAE와 RMSE 계산 함수를 가져옵니다.
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

y = [100, 200, 300, 400]  # 실제 실행시간(ms)

# 두 대안 모델(A, B)의 예측값 딕셔너리
predictions = {"A":[100, 200, 300, 480], "B":[130, 230, 330, 430]}

for name, pred in predictions.items():
    # mean_absolute_error: 실제값과 예측값 차이의 절대값을 평균 낸 값 (MAE)
    mae = mean_absolute_error(y, pred)
    # TODO B03: 같은 y·pred로 RMSE를 계산하세요. 함수 이름의 root는 제곱근을 뜻합니다.
    # root_mean_squared_error: 오차를 제곱하여 평균 낸 뒤 제곱근을 씌운 값 (RMSE)
    rmse = root_mean_squared_error(y, pred)
    # TODO 끝    
    # 지표 뒤에 단위(ms)를 붙여 모델별 성능을 출력합니다.
    print(f"{name}: MAE={mae:.1f}ms, RMSE={rmse:.1f}ms")

print("실행시간 단위 ms를 함께 적으세요. 교차검증은 06번에서 이어집니다.")
