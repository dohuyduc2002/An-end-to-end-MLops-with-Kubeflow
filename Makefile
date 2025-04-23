.PHONY: create-network, connect-network, down-network

# Create Docker network named mlops if it does not already exist
create-network:
	docker network inspect mlops >/dev/null 2>&1 || docker network create mlops

# Connect Kind container to the mlops network
connect-network:
	docker network connect mlops kubeflow-control-plane

# Disconnect Kind from mlops and remove the mlops network
down-network:
	- docker network disconnect mlops kubeflow-control-plane || true
	- docker network rm mlops || true
