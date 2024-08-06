#!/bin/bash

# This script is used to build the CUDA modules for the environments in this repository.
# This is a great way to switch between different versions of CUDA on the same machine.

# install modules
sudo apt-get update
sudo apt-get install environment-modules -y

# create the modulefiles directory
sudo mkdir -p /usr/share/modules/modulefiles/cuda


####################################################################################################
# Install and configure CUDA 12.1 for Modulefiles
####################################################################################################
{
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.1.0/local_installers/cuda-repo-ubuntu2204-12-1-local_12.1.0-530.30.02-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2204-12-1-local_12.1.0-530.30.02-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2204-12-1-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda-12-1

wget https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/linux-x86_64/cudnn-linux-x86_64-9.3.0.75_cuda12-archive.tar.xz
tar -xvf cudnn-linux-x86_64-9.3.0.75_cuda12-archive.tar.xz

cd cudnn-linux-x86_64-9.3.0.75_cuda12-archive/
sudo cp include/cudnn*.h /usr/local/cuda-12.1/include
sudo cp lib/libcudnn* /usr/local/cuda-12.1/lib64
cd ../

sudo chmod a+r /usr/local/cuda-12.1/include/cudnn*.h /usr/local/cuda-12.1/lib64/libcudnn*

sudo touch /usr/share/modules/modulefiles/cuda/12.1
sudo tee /usr/share/modules/modulefiles/cuda/12.1 > /dev/null <<EOF
#%Module1.0
##
## cuda 12.1 modulefile
##

proc ModulesHelp { } {
    global version

    puts stderr "\tSets up environment for CUDA \$version\n"
}

module-whatis "sets up environment for CUDA 12.1"

set version 12.1.0
set root /usr/local/cuda-12.1
setenv CUDA_HOME \$root

prepend-path PATH \$root/bin
prepend-path LD_LIBRARY_PATH \$root/extras/CUPTI/lib64
prepend-path LD_LIBRARY_PATH \$root/lib64
conflict cuda
EOF

} || {
    echo "CUDA 12.1 installation failed."
}

####################################################################################################
# End of CUDA 12.1 installation
####################################################################################################

# but let us make a cuda.version file suc hthat 12.1 is the default version

sudo touch /usr/share/modules/modulefiles/cuda.version
sudo cat /usr/share/modules/modulefiles/cuda.version <<EOF
#%Module
set ModulesVersion 12.1
EOF


####################################################################################################
# Install and configure CUDA 11.8 for Modulefiles
####################################################################################################
{
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda-repo-ubuntu2204-11-8-local_11.8.0-520.61.05-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2204-11-8-local_11.8.0-520.61.05-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2204-11-8-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda-11-8

wget https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/linux-x86_64/cudnn-linux-x86_64-9.3.0.75_cuda11-archive.tar.xz
tar -xvf cudnn-linux-x86_64-9.3.0.75_cuda11-archive.tar.xz

cd cudnn-linux-x86_64-9.3.0.75_cuda11-archive/
sudo cp include/cudnn*.h /usr/local/cuda-11.8/include
sudo cp lib/libcudnn* /usr/local/cuda-11.8/lib64
cd ../

sudo chmod a+r /usr/local/cuda-11.8/include/cudnn*.h /usr/local/cuda-11.8/lib64/libcudnn*

sudo touch /usr/share/modules/modulefiles/cuda/11.8
sudo tee /usr/share/modules/modulefiles/cuda/11.8 > /dev/null <<EOF
#%Module1.0
##
## cuda 11.8 modulefile
##

proc ModulesHelp { } {
    global version

    puts stderr "\tSets up environment for CUDA \$version\n"
}

module-whatis "sets up environment for CUDA 11.8"

set version 11.8.0
set root /usr/local/cuda-11.8
setenv CUDA_HOME \$root

prepend-path PATH \$root/bin
prepend-path LD_LIBRARY_PATH \$root/extras/CUPTI/lib64
prepend-path LD_LIBRARY_PATH \$root/lib64
conflict cuda
EOF

} || {
    echo "CUDA 11.8 installation failed."
}
####################################################################################################
# End of CUDA 11.8 installation
####################################################################################################


# Let us clean up the installation files and directories and make our repository clean
# rm any .deb files
rm *.deb*

# rm any tar files
rm *.tar*

# rm any cuda extracted directories
rm -r *cudnn*
