# """실습 04: 다른 행의 통계나 정답을 사용하지 않는 파생 열 하나를 만듭니다."""
# from pathlib import Path
# import pandas as pd
# import numpy as np
# raw = pd.read_csv(Path(__file__).resolve().parent / "requests.csv")
# # 원본 파일은 덮어쓰지 않습니다. 글자 수만 복사하고 정답·실행시간은 선택하지 않습니다.
# features = raw[["input_chars"]].copy()
# # TODO B04: np.log1p로 글자 수의 로그 변환 열을 추가하세요. 각 행의 값만 사용합니다.
# features["log_input_chars"] = 0.0
# # TODO 끝
# print(features.head().round(3).to_string(index=False))
# print("행/열:", features.shape)
# print("원본 열:", raw.columns.tolist())



"""실습 04: 다른 행의 통계나 정답을 사용하지 않는 파생 열 하나를 만듭니다."""
from pathlib import Path
import pandas as pd
import numpy as np

# 현재 파이썬 스크립트 파일과 같은 위치에 있는 "requests.csv" 파일을 불러옵니다.
raw = pd.read_csv(Path(__file__).resolve().parent / "requests.csv")

# 원본 파일은 덮어쓰지 않습니다. 안전하게 글자 수("input_chars") 열만 깊은 복사(copy)하여 가져오고, 
# 데이터 누수(Data Leakage)를 방지하기 위해 정답(Target)이나 실행시간 열은 제외합니다.
features = raw[["input_chars"]].copy()

# TODO B04: np.log1p로 글자 수의 로그 변환 열을 추가하세요. 각 행의 값만 사용합니다.
# 다른 행의 평균이나 표준편차를 참고하지 않고, 오직 자기 행의 데이터만 사용하여 1을 더한 후 로그를 취합니다.
features["log_input_chars"] = np.log1p(features["input_chars"])
# TODO 끝

# 상위 5개 행의 데이터를 소수점 셋째 자리까지 반올림하여 인덱스 없이 깔끔하게 출력합니다.
print(features.head().round(3).to_string(index=False))

# 데이터의 최종 형태(행과 열의 개수)를 출력합니다.
print("행/열:", features.shape)

# 원본 데이터에 어떤 열들이 있었는지 확인하여 target 등이 제외되었음을 검증합니다.
print("원본 열:", raw.columns.tolist())
