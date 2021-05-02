# base image
FROM postgres:9.6.21-alpine

# run create.sql on init
ADD create.sql /docker-entrypoint-initdb.d
