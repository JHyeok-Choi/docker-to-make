build:
	docker build -t mysql .
run:
	docker run -itd -h mysql-db-00.c9682osmmoa4.ap-northeast-2.rds.amazonaws.com -e MYSQL_ROOT_PASSWORD=asdf1234 -p 3306:3306 -u root --name mysql mysql
exec:
	docker exec -it mysql /bin/bash
logs:
	docker logs mysql
ps:
	docker ps -a
img:
	docker images
rm:
	docker rm -f $$(docker ps -aq)
rmi:
	docker rmi $$(docker images -q)
