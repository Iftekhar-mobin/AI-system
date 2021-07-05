model_name: cl-tohoku/bert-large-japanese
docker build -t bert_masking .
docker run -p 8002:8002 -v /home/iftekhar/BERTS/BERT_masking/:/home/iftekhar/BERTS/BERT_masking/ \
                        -v /home/iftekhar/mywork/amie_datadir/data:/home/iftekhar/mywork/amie_datadir/data \
                        -v /home/iftekhar/mywork/amie_datadir/model:/home/iftekhar/mywork/amie_datadir/model \
                        bert_masking:latest

http://0.0.0.0:8002/mask
{"model_path": "/home/iftekhar/BERTS/BERT_masking/",
"data_dir": "/home/iftekhar/mywork/amie_datadir/data",
"model_dir": "/home/iftekhar/mywork/amie_datadir/model",
"query": "hello how are you",
"agent_id": 1
}