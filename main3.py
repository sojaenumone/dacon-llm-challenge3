import os
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
import datetime

# 파일 경로
train_path = "/Users/nextweb/Downloads/open/train.csv"
test_path = "/Users/nextweb/Downloads/open/test.csv"
submission_path = "/Users/nextweb/Downloads/open/sample_submission.csv"

# 데이터 확인
for path in [train_path, test_path, submission_path]:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{path} 파일이 존재하지 않습니다.")

# 데이터 로드
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)
submission = pd.read_csv(submission_path)

# 모델 및 토크나이저 설정 (경량 모델 사용)
MODEL_NAME = "monologg/koelectra-small-v3-discriminator"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

# Dataset 정의
class TextDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels=None):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=256)
        self.labels = labels
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx])
        return item
    def __len__(self):
        return len(self.encodings["input_ids"])

# 학습용 데이터 분할
train_texts, val_texts, train_labels, val_labels = train_test_split(
    train_df["full_text"].tolist(), train_df["generated"].tolist(), test_size=0.1, random_state=42
)

train_dataset = TextDataset(train_texts, train_labels)
val_dataset = TextDataset(val_texts, val_labels)
test_dataset = TextDataset(test_df["paragraph_text"].tolist())

# TrainingArguments 최소화 (오류 방지 목적)
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=2,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    logging_dir="./logs"
)

# Trainer 정의 및 학습
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

trainer.train()

# 예측 수행
predictions = trainer.predict(test_dataset)
probs = torch.nn.functional.softmax(torch.tensor(predictions.predictions), dim=1)[:, 1].numpy()

# 결과 저장
submission["generated"] = probs
now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
output_path = f"/Users/nextweb/Downloads/open/submission_{now}.csv"
submission.to_csv(output_path, index=False)
print(f"✅ 제출용 파일 저장 완료: {output_path}")
