#!/usr/bin/env bash
set -e

echo "🚀 Setting up Docker dev environment..."

# ---------------------------
# 1. System packages
# ---------------------------
echo "📦 Installing system packages..."
apt update && apt install -y \
    curl wget git build-essential \
    tmux htop vim nano \
    unzip zip \
    ca-certificates \
    software-properties-common \
    libgl1-mesa-glx libglfw3 libosmesa6-dev patchelf \
    zsh

# ---------------------------
# 2. Install Miniconda
# ---------------------------
echo "🐍 Installing Miniconda..."
CONDA_DIR=/opt/conda
if [ ! -d "$CONDA_DIR" ]; then
  wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
  bash /tmp/miniconda.sh -b -p $CONDA_DIR
  rm /tmp/miniconda.sh
fi

# Init conda for bash + zsh
$CONDA_DIR/bin/conda init bash
$CONDA_DIR/bin/conda init zsh

# ---------------------------
# 3. Install Oh-My-Zsh
# ---------------------------
echo "✨ Installing Oh My Zsh..."
export RUNZSH=no
export CHSH=no
sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"

# ---------------------------
# 4. Zsh plugins
# ---------------------------
echo "🔌 Installing Zsh plugins..."
ZSH_CUSTOM=${ZSH_CUSTOM:-$HOME/.oh-my-zsh/custom}

git clone https://github.com/zsh-users/zsh-autosuggestions $ZSH_CUSTOM/plugins/zsh-autosuggestions || true
git clone https://github.com/zsh-users/zsh-syntax-highlighting $ZSH_CUSTOM/plugins/zsh-syntax-highlighting || true
git clone https://github.com/agkozak/zsh-z.git ~/.zsh/zsh-z $ZSH_CUSTOM/plugins/z || true

# ---------------------------
# 5. Enable plugins
# ---------------------------
echo "⚙️ Configuring ~/.zshrc..."
sed -i 's/^plugins=.*/plugins=(git conda zsh-autosuggestions zsh-syntax-highlighting)/' ~/.zshrc

# ---------------------------
# 6. Make zsh default shell
# ---------------------------
chsh -s /usr/bin/zsh root

echo "✅ Setup complete!"
echo "➡️  Restart the shell: exec zsh"
