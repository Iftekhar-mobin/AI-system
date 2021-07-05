docker build -t google_assistant .
docker run -p 8004:8004 google_assistant:latest

http://0.0.0.0:8004/assistant
{
"query": "hello how are you",
"user_nm": 1,
}