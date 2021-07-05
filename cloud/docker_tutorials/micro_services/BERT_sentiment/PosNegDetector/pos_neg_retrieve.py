from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline


def load_sent_retriever(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    pre_model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model = pipeline("sentiment-analysis", model=pre_model, tokenizer=tokenizer)
    return model
