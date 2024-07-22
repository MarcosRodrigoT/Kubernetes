kubectl create namespace jupyterhub
kubectl apply -f local_storage.yaml -n jupyterhub
helm upgrade --install jupyterhub jupyterhub/jupyterhub --namespace jupyterhub --values config.yaml
