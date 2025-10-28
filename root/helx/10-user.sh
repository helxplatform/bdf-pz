#!/bin/bash

set -eoux pipefail

# Set DELETE_DEFAULT_USER_HOME_IF_UNUSED to anything other than "yes" to disable.
DELETE_DEFAULT_USER_HOME_IF_UNUSED=${DELETE_DEFAULT_USER_HOME_IF_UNUSED-"yes"}
# Set CHOWN_USER_HOME_DIR to anything other than "no" to enable.
CHOWN_USER_HOME_DIR=${CHOWN_USER_HOME_DIR="no"}
declare -i DEFAULT_UID=1000
declare -i DEFAULT_GID=0
declare -i CURRENT_UID=`id -u`
declare -i CURRENT_GID=`id -g`

make_home() {
  HOME_DIR=$1
  HOME_UID=$2
  HOME_GID=$3
  # allow errors for this section
  set +e
  mkdir -p $HOME_DIR
  chown $HOME_UID:$HOME_GID $HOME_DIR
  # Copy default environment setup files if they don't already exist.
  if [ ! -f $HOME_DIR/.bashrc ]; then
    if [ -f /etc/skel/.bashrc ]; then
      cp /etc/skel/.bashrc $HOME_DIR/.bashrc
      chown $HOME_UID:$HOME_GID $HOME_DIR/.bashrc
    fi
  fi
  if [ ! -f $HOME_DIR/.bash_logout ]; then
    if [ -f /etc/skel/.bash_logout ]; then
      cp /etc/skel/.bash_logout $HOME_DIR/.bash_logout
      chown $HOME_UID:$HOME_GID $HOME_DIR/.bash_logout
    fi
  fi
  if [ ! -f $HOME_DIR/.profile ]; then
    if [ -f /etc/skel/.profile ]; then
      cp /etc/skel/.profile $HOME_DIR/.profile
      chown $HOME_UID:$HOME_GID $HOME_DIR/.profile
    fi
  fi
  set -e
}

echo "running as UID=$CURRENT_UID GID=$CURRENT_GID"

if [ $CURRENT_UID -ne 0 ]; then
  echo "not running as root"
  if [[ $CURRENT_UID -eq $DEFAULT_UID && "$USER" == "$DEFAULT_USER" ]]; then
    echo "image is running as the default user created in Dockerfile"
  else
    echo "not running as uid=$DEFAULT_UID or USER!=\"$DEFAULT_USER\""
    if [[ $USER_IDENTITY == "local" ]]; then
      # Modify user entry in /etc/passwd.
      cp /etc/passwd /tmp/passwd
      sed -i -e "s/^$DEFAULT_USER\:x\:$DEFAULT_UID\:$DEFAULT_GID\:\:\/home\/$DEFAULT_USER/$USER\:x\:$CURRENT_UID\:$CURRENT_GID\:\:\/home\/$USER/" /tmp/passwd
      cp /tmp/passwd /etc/passwd
      rm /tmp/passwd
    fi
    make_home /home/$USER $CURRENT_UID $CURRENT_GID
  fi
else
  echo "running as root"
  if [[ "$USER" != "$DEFAULT_USER" || \
        "$USER_UID" != "$DEFAULT_UID" || \
        "$USER_GID" != "$DEFAULT_GID" ]]; then
    if [[ $USER_IDENTITY == "local" ]]; then
      # running as root, but will modify the default user entry in /etc/passwd
      # so it can be switched to from root.
      echo "renaming $DEFAULT_USER username to $USER"

      # Modify user entry in /etc/passwd.
      cp /etc/passwd /tmp/passwd
      sed -i -e "s/^$DEFAULT_USER\:x\:$DEFAULT_UID\:$DEFAULT_GID\:\:\/home\/$DEFAULT_USER/$USER\:x\:$USER_UID\:$USER_GID\:\:\/home\/$USER/" /tmp/passwd
      cp /tmp/passwd /etc/passwd
      rm /tmp/passwd

      # modify entry in /etc/shadow - needed to su to user
      cp /etc/shadow /tmp/shadow
      sed -i -e "s/^$DEFAULT_USER\:!\:/$USER\:!\:/" /tmp/shadow
      cp /tmp/shadow /etc/shadow
      rm /tmp/shadow
    fi

    make_home /home/$USER $USER_UID $USER_GID
    make_home $HOME 0 0
    if [[ $CHOWN_USER_HOME_DIR != "no" ]]; then
      # This can fail on some filesystems (NFS with root squash).  Might also
      # cause problems if there are existing files that have a different UID/GID.
      # Can also take a long time if lots of files.
      chown -R $USER_UID:$USER_GID /home/$USER
    fi
  fi
fi

if [[ ${USER_IDENTITY:-} != "ldap" ]]; then
  echo "USER_IDENTITY is either unset or not 'ldap'"
  if [[ "$USER" != "$DEFAULT_USER" ]]; then
    if [[ $DELETE_DEFAULT_USER_HOME_IF_UNUSED == "yes" ]]; then
      if [ -d /home/$DEFAULT_USER ]; then
        echo "deleting /home/$DEFAULT_USER"
        rm -rf /home/$DEFAULT_USER || echo "Warning: failed to delete /home/$DEFAULT_USER"
      fi
    fi
  fi
fi