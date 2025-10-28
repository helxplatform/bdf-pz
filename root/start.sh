#!/bin/bash

set -eoux pipefail

export USER=${USER-"helx"}
export USER_UID=${USER_UID-"1000"}
export USER_GID=${USER_GID-"0"}
export DEFAULT_USER="helx"
# WARNING: Setting RSTUDIO_SERVER_BASE_PATH to anything else other than "/"
export USER_IDENTITY=${USER_IDENTITY-"local"}

declare -i CURRENT_UID=`id -u`
if [ $CURRENT_UID -ne 0 ]
then
  export HOME="/home/$USER"
else
  export HOME="/root"
fi

export NB_ROOT_DIR=${NB_ROOT_DIR-$HOME}
export JUPYTER_BASE_URL=$NB_PREFIX
export JUPYTER_WS_URL="wss://$HOST$NB_PREFIX/"
export JUPYTER_SERVER=$NB_PREFIX

# Change to the root directory to mitigate problems if the current working
# directory is deleted.
cd /

# Add other init scripts in $HELX_SCRIPTS_DIR with ".sh" as their extension.
# To run in a certain order, name them appropriately.
HELX_SCRIPT_DIR=/helx
INIT_SCRIPTS_TO_RUN=$(ls -1 $HELX_SCRIPT_DIR/*.sh) || true
for INIT_SCRIPT in $INIT_SCRIPTS_TO_RUN
do
  echo "Running $INIT_SCRIPT"
  $INIT_SCRIPT
done

# Change CWD to /home/$USER so it is the starting point for shells.
cd $HOME


# Where user-specific non-essential (cached) data should be written (analogous to /var/cache).
# Should default to $HOME/.cache.
# https://wiki.archlinux.org/title/XDG_Base_Directory
export XDG_CACHE_HOME=$HOME/.cache

# Fix static URLs in beaker asset bundle.
/fix-ui-bundle.sh

python -m bdf_pz.server \
    --IdentityProvider.token='' \
    --ServerApp.ip='0.0.0.0' \
    --ServerApp.base_url=${NB_PREFIX} \
    --ServerApp.allow_origin="*" \
    --ServerApp.root_dir=${NB_ROOT_DIR} \
    --ServerApp.default_url=${NB_PREFIX} \
    --ContentsManager.allow_hidden=True