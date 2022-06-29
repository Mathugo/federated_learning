# openfl_semantic_seg
Federated Learning with Openfl and Pytorch 
## Goal of this project 
This project aims to perform a federated and distributed learning approch to semantic segmentation with Pytorch using the director pattern from Openfl

[![openfl][openfl]](https://openfl.readthedocs.io/en/latest/_images/director_workflow.svg)


## How to use 
### envoy 
Simply start the envoys using their associated bash files : 
```
    chmod +x start_envoy.sh 
    ./start_envoy.sh $ENVOY_NAME $DIRECTOR_ADDR
```

with parameters $ENVOY_NAME (name of the envoy, ex : env_one) $DIRECTOR_ADDR (ip address of the director, ex: localhost)

### director 
Start the director with
```sh
    chmod +x start_director.sh
    ./start_director.sh 
```

### Experiment
You can then try to run the notebooks in the [research](https://github.com/Mathugo/federated_learning/tree/main/research) folder 

## Run commands
### Create director workspace for research
fx director create-workspace -p path/to/director_workspace_dir

### Create Envoy workspace for collaborator manager 
fx envoy create-workspace -p path/to/envoy_workspace_dir

### Start director and disable mTLS protection 
fx director start --disable-tls -c director.yaml

### Start envoy and disable mTLS protection
ENVOY_NAME=orange_client_1
fx envoy start \
    -n "$ENVOY_NAME" \
    --disable-tls \
    --envoy-config-path envoy_config.yaml \
    -dh director_fqdn \
    -dp port

### Start director with federation PKI Certificates 

fx director start -c director.yaml \
     -rc cert/root_ca.crt \
     -pk cert/priv.key \
     -oc cert/open.crt

### Start envoy with federation PKI Certificates

ENVOY_NAME=orange_client1

fx envoy start \
    -n "$ENVOY_NAME" \
    --envoy-config-path envoy_config.yaml \
    -dh director_fqdn \
    -dp port \
    -rc cert/root_ca.crt \
    -pk cert/"$ENVOY_NAME".key \
    -oc cert/"$ENVOY_NAME".crt

