# """실습 05: train 12건의 글자 수 하나로 두 묶음을 만듭니다."""
# from pathlib import Path
# import pandas as pd
# from sklearn.model_selection import train_test_split
# HERE = Path(__file__).resolve().parent
# raw = pd.read_csv(HERE / "requests.csv")
# # 제공: 앞 수업과 같은 seed·층화·행 순서입니다. test 4건은 마지막 평가 전까지 보관합니다.
# develop, test = train_test_split(raw, test_size=0.2, stratify=raw["failed"], random_state=42)
# train, valid = train_test_split(develop, test_size=0.25, stratify=develop["failed"], random_state=42)

# from sklearn.cluster import KMeans
# # 같은 단위인 한 열만 사용하므로 여기서는 표준화를 생략합니다. failed·elapsed_ms는 넣지 않습니다.
# model = KMeans(n_clusters=2, n_init=10, random_state=42)
# # TODO B05: KMeans를 train의 input_chars 한 열로 학습하세요. 정답 y는 전달하지 않습니다.
# pass
# # TODO 끝
# result = train[["request_id", "input_chars"]].copy()
# result["cluster"] = model.labels_
# print("군집별 개수:", result["cluster"].value_counts().sort_index().to_dict())
# print("중심(글자):", model.cluster_centers_.round(1).ravel().tolist())
# result.to_csv(HERE / "clusters.csv", index=False, encoding="utf-8-sig")
# print("저장: clusters.csv (군집 번호는 성공·실패 라벨이 아닙니다.)")



"""실습 05: train 12건의 글자 수 하나로 두 묶음을 만듭니다."""
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

HERE = Path(__file__).resolve().parent
raw = pd.read_csv(HERE / "requests.csv")

# 제공: 앞 수업과 같은 seed·층화·행 순서입니다. test 4건은 마지막 평가 전까지 보관합니다.
# 전체 20건 기준: develop(16건, 80%), test(4건, 20%) 분할
develop, test = train_test_split(raw, test_size=0.2, stratify=raw["failed"], random_state=42)
# develop 16건 기준: train(12건, 75%), valid(4건, 25%) 분할 (결과적으로 전체의 60%가 train이 됨)
train, valid = train_test_split(develop, test_size=0.25, stratify=develop["failed"], random_state=42)

from sklearn.cluster import KMeans
# 데이터 스케일링을 생략한 채, 글자 수 단일 열을 기준으로 데이터를 2개의 묶음(n_clusters=2)으로 분류합니다.
model = KMeans(n_clusters=2, n_init=10, random_state=42)

# TODO B05: KMeans를 train의 input_chars 한 열로 학습하세요. 정답 y는 전달하지 않습니다.
# 단일 대괄호 train["input_chars"]는 1차원 Series를 반환하여 에러가 나므로,
# 반드시 이중 대괄호 [['input_chars']]를 사용해 2차원 DataFrame 형태로 모델에 전달해야 합니다.
model.fit(train[["input_chars"]])
# TODO 끝

result = train[["request_id", "input_chars"]].copy()
# model.labels_: 학습 데이터의 각 행이 어떤 군집(0 또는 1)으로 할당되었는지 저장된 속성입니다.
result["cluster"] = model.labels_

print("군집별 개수:", result["cluster"].value_counts().sort_index().to_dict())
# model.cluster_centers_: 학습 결과로 결정된 각 군집의 중심점(글자 수 좌표)입니다. 
# ravel()을 통해 1차원으로 펼쳐 리스트로 출력합니다.
print("중심(글자):", model.cluster_centers_.round(1).ravel().tolist())

# 군집화 결과를 csv 파일로 저장하여 이후 작업에서 연계할 수 있도록 합니다.
result.to_csv(HERE / "clusters.csv", index=False, encoding="utf-8-sig")
print("저장: clusters.csv (군집 번호는 성공·실패 라벨이 아닙니다.)")
