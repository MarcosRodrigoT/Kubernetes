helm delete jupyterhub --namespace jupyterhub
kubectl delete sc local-storage
kubectl delete namespace jupyterhub
