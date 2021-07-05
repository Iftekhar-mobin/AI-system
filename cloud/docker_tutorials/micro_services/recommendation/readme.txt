docker build -t google_assistant .
docker run -p 8004:8004 google_assistant:latest

docker build -t recommendation_model .
docker run -p 8001:8001 -v /home/iftekhar/mywork/amie_datadir/data/:/home/iftekhar/mywork/amie_datadir/data/ /home/iftekhar/mywork/amie_datadir/model/:/home/iftekhar/mywork/amie_datadir/model/  qa_retrieve_model:latest


