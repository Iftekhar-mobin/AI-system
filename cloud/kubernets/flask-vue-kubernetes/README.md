# Running Flask on Kubernetes

## Want to learn how to build this?

Check out the [post](https://testdriven.io/running-flask-on-kubernetes).
https://github.com/testdrivenio/flask-vue-kubernetes

## Want to use this project?

### Docker

Build the images and spin up the containers:

```sh
$ docker-compose up -d --build
```

Run the migrations and seed the database:

```sh
$ docker-compose exec server python manage.py recreate_db
$ docker-compose exec server python manage.py seed_db
```

Test it out at:

1. [http://localhost:8080/](http://localhost:8080/)
1. [http://localhost:5001/books/ping](http://localhost:5001/books/ping)
1. [http://localhost:5001/books](http://localhost:5001/books)

### Kubernetes

#### Minikube

Install and run [Minikube](https://kubernetes.io/docs/setup/minikube/):

1. Install a [Hypervisor](https://kubernetes.io/docs/tasks/tools/install-minikube/#install-a-hypervisor) (like [VirtualBox](https://www.virtualbox.org/wiki/Downloads) or [HyperKit](https://github.com/moby/hyperkit)) to manage virtual machines
1. Install and Set Up [kubectl](https://kubernetes.io/docs/tasks/tools/install-kubectl/) to deploy and manage apps on Kubernetes
1. Install [Minikube](https://github.com/kubernetes/minikube/releases)

Start the cluster:

```sh
$ minikube start --vm-driver=virtualbox
$ minikube dashboard
```

#### Volume

Create the volume:

```sh
$ kubectl apply -f ./kubernetes/persistent-volume.yml
```

Create the volume claim:

```sh
$ kubectl apply -f ./kubernetes/persistent-volume-claim.yml
```

#### Secrets

$ echo -n "pleasechangeme" | base64
cGxlYXNlY2hhbmdlbWU=

$ echo -n "sample" | base64
c2FtcGxl

Create the secret object:

```sh
$ kubectl apply -f ./kubernetes/secret.yml
```

#### Postgres

Create deployment:

```sh
$ kubectl create -f ./kubernetes/postgres-deployment.yml
```

Create the service:

```sh
$ kubectl create -f ./kubernetes/postgres-service.yml
```

Create the database:

```sh
$ kubectl get pods
$ kubectl exec postgres-<POD_IDENTIFIER> --stdin --tty -- createdb -U postgres books
$ kubectl exec postgres-95566f9-xs2cf --stdin --tty -- psql -U sample
/l
cltr+D
```

Get POD names
```
kubectl get pod -l service=postgres -o jsonpath="{.items[0].metadata.name}"
$ POD_NAME=$(kubectl get pod -l service=postgres -o jsonpath="{.items[0].metadata.name}")
$ kubectl exec $POD_NAME --stdin --tty -- createdb -U sample books
```


#### Flask

if don't want to push the image to a Docker registry, after you build the image locally, you can set the image-pull-policy flag to Never to always use the local image.

Build and push the image to Docker Hub:

```sh
$ docker build -t mjhea0/flask-kubernetes ./services/server
$ docker push mjhea0/flask-kubernetes
```

> Make sure to replace `mjhea0` with your Docker Hub namespace in the above commands as well as in *kubernetes/flask-deployment.yml*

Create the deployment:

```sh
$ kubectl create -f ./kubernetes/flask-deployment.yml
```

Create the service:

```sh
$ kubectl create -f ./kubernetes/flask-service.yml
```

Apply the migrations and seed the database:

```sh
$ kubectl get pods
$ kubectl exec flask-<POD_IDENTIFIER> --stdin --tty -- python manage.py recreate_db
$ kubectl exec flask-<POD_IDENTIFIER> --stdin --tty -- python manage.py seed_db
```

``` Varify
$ kubectl exec postgres-95566f9-xs2cf --stdin --tty -- psql -U sample
sample=# \c books
books=# select * from books;
```

	
#### Ingress

Enable and apply:

```sh
$ minikube addons enable ingress
$ kubectl apply -f ./kubernetes/minikube-ingress.yml
```

```Error
Error from server (InternalError): error when creating "minikube-ingress.yml": Internal error occurred: failed calling webhook "validate.nginx.ingress.kubernetes.io": an error on the server ("") has prevented the request from succeeding

$ kubectl delete -A ValidatingWebhookConfiguration ingress-nginx-admission
```


Add entry to */etc/hosts* file:

```
<MINIKUBE_IP> hello.world
$ echo "$(minikube ip) hello.world" | sudo tee -a /etc/hosts
```

Try it out:

1. [http://hello.world/books/ping](http://hello.world/books/ping)
1. [http://hello.world/books](http://hello.world/books)


#### Vue

Build and push the image to Docker Hub:

```sh
$ docker build -t mjhea0/vue-kubernetes ./services/client \
    -f ./services/client/Dockerfile-minikube

$ docker push mjhea0/vue-kubernetes
```

> Again, replace `mjhea0` with your Docker Hub namespace in the above commands as well as in *kubernetes/vue-deployment.yml*

Create the deployment:

```sh
$ kubectl create -f ./kubernetes/vue-deployment.yml
```

```check
kubectl get deploy vue
kubectl get pods
```


Create the service:

```sh
$ kubectl create -f ./kubernetes/vue-service.yml
```

Try it out at [http://hello.world/](http://hello.world/).

```scaling
$ kubectl scale deployment flask --replicas=2
kubectl get deployments flask
kubectl get pods -o wide
for ((i=1;i<=10;i++)); do curl http://hello.world/books/ping; done
```