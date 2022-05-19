#!/bin/bash
set -e


ENVOY_NAME=$1
DIRECTOR_FQDN=$2

fx envoy start -n "$ENVOY_NAME" --disable-tls --envoy-config-path envoy_config_no_gpu.yaml -dh "$DIRECTOR_FQDN" -dp 50055
