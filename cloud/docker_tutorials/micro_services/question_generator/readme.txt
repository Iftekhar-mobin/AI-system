# Micro service Preparation
# Reference https://huggingface.co/valhalla/t5-base-e2e-qg
# Local path of the model
/home/iftekhar/BERTS/QA_model/
docker build -t qa_retrieve_model .
docker run -p 8001:8001 -v /home/iftekhar/BERTS/QA_model:/home/iftekhar/BERTS/QA_model qa_retrieve_model:latest
