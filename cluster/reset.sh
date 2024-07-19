sudo kubeadm reset --cri-socket unix:///var/run/cri-dockerd.sock
sudo rm -f $HOME/.kube/config
sudo rm -rf /etc/cni/net.d
sudo rm -rf /var/lib/kubelet/*
sudo rm -rf /etc/kubernetes/
