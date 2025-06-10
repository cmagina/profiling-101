#! /bin/bash

set -o errexit  # abort on nonzero exitstatus
set -o nounset  # abort on unbound variable
set -o pipefail # don't hide errors within pipes
# set -o xtrace   # display expanded commands and arguments

pushd /workspace 1>/dev/null || return
jupyter notebook --ip=0.0.0.0 --port="$NOTEBOOK_PORT" --no-browser --allow-root
popd 1>/dev/null || return
