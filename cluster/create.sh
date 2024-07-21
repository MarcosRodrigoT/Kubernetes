# Create cluster
sudo kubeadm init --kubernetes-version=v1.30.2 --cri-socket unix:///var/run/cri-dockerd.sock --pod-network-cidr=192.168.0.0/16

# Prepare permissions for kubectl
mkdir -p $HOME/.kube
sudo cp -i /etc/kubernetes/admin.conf $HOME/.kube/config
sudo chown $(id -u):$(id -g) $HOME/.kube/config

# Apply Calico networking manifest
kubectl apply -f calico.yaml

# Apply NVIDIA manifest to allow GPU usage
kubectl create -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.11.0/nvidia-device-plugin.yml

# Print command to let other nodes joining
join_command=$(sudo kubeadm token create --print-join-command)
full_join_command="sudo $join_command --cri-socket unix:///var/run/cri-dockerd.sock"
echo
echo
echo
echo "Cluster created successfully! You can join with other nodes by running this command:"
echo
echo "$full_join_command"
echo
