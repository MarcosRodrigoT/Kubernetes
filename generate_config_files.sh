#!/bin/bash

# Function to print usage
usage() {
    echo "Usage: $0 --framework <tensorflow|pytorch> --nodes <number_of_nodes> --gpus <number_of_gpus>"
    exit 1
}

# Check if the number of arguments is correct
if [ "$#" -ne 6 ]; then
    usage
fi

# Parse arguments
while [ "$#" -gt 0 ]; do
    case "$1" in
        --framework)
            FRAMEWORK="$2"
            shift 2
            ;;
        --nodes)
            NODES="$2"
            shift 2
            ;;
        --gpus)
            GPUS="$2"
            shift 2
            ;;
        *)
            usage
            ;;
    esac
done

# Validate arguments
if [[ "$FRAMEWORK" != "tensorflow" && "$FRAMEWORK" != "pytorch" ]]; then
    echo "Error: --framework must be either 'tensorflow' or 'pytorch'"
    usage
fi

if ! [[ "$NODES" =~ ^[0-9]+$ ]] || [ "$NODES" -lt 1 ]; then
    echo "Error: --nodes must be an integer greater than or equal to 1"
    usage
fi

if ! [[ "$GPUS" =~ ^[0-9]+$ ]] || [ "$GPUS" -lt 0 ]; then
    echo "Error: --gpus must be an integer greater than or equal to 0"
    usage
fi

# Generate the configuration for TensorFlow
if [ "$FRAMEWORK" == "tensorflow" ]; then
    CONFIG="apiVersion: v1
kind: Service
metadata:
  name: tensorflow-master-service
spec:
  selector:
    app: tensorflow-master
  ports:
    - protocol: TCP
      port: 12345
      targetPort: 12345
---
"

    for ((i=1; i<NODES; i++)); do
        CONFIG+="apiVersion: v1
kind: Service
metadata:
  name: tensorflow-worker-${i}-service
spec:
  selector:
    app: tensorflow-worker-${i}
  ports:
    - protocol: TCP
      port: 12345
      targetPort: 12345
---
"
    done

    CONFIG+="apiVersion: apps/v1
kind: Deployment
metadata:
  name: tensorflow-master
  labels:
    app: tensorflow-master
spec:
  replicas: 1
  selector:
    matchLabels:
      app: tensorflow-master
  template:
    metadata:
      labels:
        app: tensorflow-master
    spec:
      containers:
      - name: tensorflow
        image: tensorflow/tensorflow:latest-gpu
        command: [\"/bin/bash\", \"-c\", \"--\"]
        args: [\"while true; do sleep 10; done;\"]
        env:
        - name: TF_CONFIG
          value: '{
            \"cluster\": {
              \"worker\": [
                \"\$(TENSORFLOW_MASTER_SERVICE_SERVICE_HOST):\$(TENSORFLOW_MASTER_SERVICE_PORT_12345_TCP_PORT)\",
"

    for ((i=1; i<NODES; i++)); do
        CONFIG+="                \"\$(TENSORFLOW_WORKER_${i}_SERVICE_SERVICE_HOST):\$(TENSORFLOW_WORKER_${i}_SERVICE_PORT_12345_TCP_PORT)\",
"
    done

    CONFIG+="              ]
            },
            \"task\": {\"type\": \"worker\", \"index\": 0}
          }'
        resources:
          limits:
            nvidia.com/gpu: $GPUS
        volumeMounts:
        - mountPath: /workspace
          name: data
      volumes:
      - name: data
        hostPath:
          path: /workspace
          type: Directory
---
"

    for ((i=1; i<NODES; i++)); do
        CONFIG+="apiVersion: apps/v1
kind: Deployment
metadata:
  name: tensorflow-worker-${i}
  labels:
    app: tensorflow-worker-${i}
spec:
  replicas: 1
  selector:
    matchLabels:
      app: tensorflow-worker-${i}
  template:
    metadata:
      labels:
        app: tensorflow-worker-${i}
    spec:
      containers:
      - name: tensorflow
        image: tensorflow/tensorflow:latest-gpu
        command: [\"/bin/bash\", \"-c\", \"--\"]
        args: [\"while true; do sleep 10; done;\"]
        env:
        - name: TF_CONFIG
          value: '{
            \"cluster\": {
              \"worker\": [
                \"\$(TENSORFLOW_MASTER_SERVICE_SERVICE_HOST):\$(TENSORFLOW_MASTER_SERVICE_PORT_12345_TCP_PORT)\",
"

        for ((j=1; j<NODES; j++)); do
            CONFIG+="                \"\$(TENSORFLOW_WORKER_${j}_SERVICE_SERVICE_HOST):\$(TENSORFLOW_WORKER_${j}_SERVICE_PORT_12345_TCP_PORT)\",
"
        done

        CONFIG+="              ]
            },
            \"task\": {\"type\": \"worker\", \"index\": $i}
          }'
        resources:
          limits:
            nvidia.com/gpu: $GPUS
        volumeMounts:
        - mountPath: /workspace
          name: data
      volumes:
      - name: data
        hostPath:
          path: /workspace
          type: Directory
---
"
    done

    echo "$CONFIG" > tensorflow_deployment.yaml
    echo "Generated tensorflow_deployment.yaml"
fi

# Generate the configuration for PyTorch
if [ "$FRAMEWORK" == "pytorch" ]; then
    CONFIG="apiVersion: v1
kind: Service
metadata:
  name: pytorch-service
spec:
  clusterIP: None
  selector:
    app: pytorch-master
  ports:
    - protocol: TCP
      port: 29400
      targetPort: 29400
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: pytorch-master
  labels:
    app: pytorch-master
spec:
  replicas: 1
  selector:
    matchLabels:
      app: pytorch-master
  template:
    metadata:
      labels:
        app: pytorch-master
    spec:
      containers:
      - name: pytorch
        image: pytorch/pytorch:latest
        command: [\"/bin/bash\", \"-c\", \"--\"]
        args: [\"while true; do sleep 10; done;\"]
        env:
        - name: NODE_RANK
          value: \"0\"
        - name: MASTER_ADDR
          value: \"localhost\"
        - name: MASTER_PORT
          value: \"29400\"
        ports:
        - containerPort: 29400
        resources:
          limits:
            nvidia.com/gpu: $GPUS
        volumeMounts:
        - mountPath: /workspace
          name: data
      volumes:
      - name: data
        hostPath:
          path: /workspace
          type: Directory
---
"

    for ((i=1; i<NODES; i++)); do
        CONFIG+="apiVersion: apps/v1
kind: Deployment
metadata:
  name: pytorch-worker-${i}
  labels:
    app: pytorch-worker
spec:
  replicas: 1
  selector:
    matchLabels:
      app: pytorch-worker
  template:
    metadata:
      labels:
        app: pytorch-worker
    spec:
      containers:
      - name: pytorch
        image: pytorch/pytorch:latest
        command: [\"/bin/bash\", \"-c\", \"--\"]
        args: [\"while true; do sleep 10; done;\"]
        env:
        - name: NODE_RANK
          value: \"${i}\"
        - name: MASTER_ADDR
	  value: \"\$(PYTORCH_SERVICE_SERVICE_HOST):\$(PYTORCH_SERVICE_PORT_29400_TCP_PORT)\"
        ports:
        - containerPort: 29400
        resources:
          limits:
            nvidia.com/gpu: $GPUS
        volumeMounts:
        - mountPath: /workspace
          name: data
      volumes:
      - name: data
        hostPath:
          path: /workspace
          type: Directory
---
"
    done

    echo "$CONFIG" > pytorch_deployment.yaml
    echo "Generated pytorch_deployment.yaml"
fi
