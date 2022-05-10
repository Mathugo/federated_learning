# federated_learning
Federated Learning with Openfl

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

