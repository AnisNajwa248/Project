from transformers import RobertaForSequenceClassification, RobertaTokenizerFast
import torch

path = "models/roberta_final"

tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")  # changed: fresh from hub, not local path
model = RobertaForSequenceClassification.from_pretrained(path)
model.eval()

with open("test_article.txt", "r", encoding="utf-8") as f:
    text = f.read()

inputs = tokenizer(text, truncation=True, max_length=128, padding="max_length", return_tensors="pt")
with torch.no_grad():
    logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=1)[0]

print("Probabilities [FAKE, REAL]:", probs)
print(inputs['input_ids'][0][:30])