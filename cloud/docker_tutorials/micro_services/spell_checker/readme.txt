docker build -t spell_checker .
docker run -p 8000:8000 -v /home/iftekhar/mywork/amie_datadir/data/:/home/iftekhar/mywork/amie_datadir/data/ spell_checker:latest

