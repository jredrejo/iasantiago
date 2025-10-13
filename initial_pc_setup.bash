# Prerrequisitos del kernel y compilación
sudo apt update

sudo apt install -y build-essential dkms linux-headers-$(uname -r) curl ca-certificates gnupg

# instalación de python
sudo apt install -y python-dev-is-python3 python3-pip

# instalación de docker
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg

echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt update
sudo apt install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
sudo usermod -aG docker $USER
sudo systemctl enable --now docker

# Blacklist the nouveau driver
sudo cp blacklist-nvidia-nouveau.conf /etc/modprobe.d/
# Update the initial ramdisk to apply the blacklist
sudo update-initramfs -u

# Repositorio oficial de CUDA (Ubuntu 24.04 = noble)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb

echo "deb [signed-by=/usr/share/keyrings/cuda-archive-keyring.gpg] https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/ /" \
    | tee /etc/apt/sources.list.d/cuda-ubuntu2404.list
sudo apt-get update

# Instala toolkit CUDA 13.0 (debe ser open para el driver serie 580 compatible con Blackwell)
sudo apt-get install -y nvidia-driver-580-open nvidia-dkms-580-open cuda-drivers-580

sudo apt -y install nvidia-cuda-toolkit cuda-toolkit-13-0


# Install the toolkit
sudo apt-get install -y nvidia-container-toolkit

# Configure Docker to use NVIDIA runtime
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker





