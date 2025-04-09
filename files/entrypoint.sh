#! /bin/bash

set -o errexit  # abort on nonzero exitstatus
set -o nounset  # abort on unbound variable
set -o pipefail # don't hide errors within pipes
# set -o xtrace   # display expanded commands and arguments

exec jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root
