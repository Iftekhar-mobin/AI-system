model_name: daigo/bert-base-japanese-sentiment
docker build -t bert_sentiment .
docker run -p 8001:8001 -v /home/iftekhar/bert_sentiment:/home/iftekhar/bert_sentiment bert_sentiment:latest
