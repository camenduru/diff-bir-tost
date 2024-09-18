FROM runpod/pytorch:2.2.1-py3.10-cuda12.1.1-devel-ubuntu22.04
WORKDIR /content
ENV PATH="/home/camenduru/.local/bin:${PATH}"

RUN adduser --disabled-password --gecos '' camenduru && \
    adduser camenduru sudo && \
    echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers && \
    chown -R camenduru:camenduru /content && \
    chmod -R 777 /content && \
    chown -R camenduru:camenduru /home && \
    chmod -R 777 /home && \
    apt update -y && add-apt-repository -y ppa:git-core/ppa && apt update -y && apt install -y aria2 git git-lfs unzip ffmpeg

USER camenduru

RUN pip install -q opencv-python imageio imageio-ffmpeg ffmpeg-python av runpod \
    einops==0.7.0 pytorch-lightning==2.2.1 omegaconf==2.3.0 lpips==0.1.4 ftfy==6.2.0 regex==2023.12.25 timm==0.9.16 facexlib==0.3.0 accelerate==0.28.0 && \
    git clone -b v2.0 https://github.com/camenduru/DiffBIR /content/DiffBIR && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/DiffBIR/resolve/main/face_full_v1.ckpt -d /content/DiffBIR/weights -o face_full_v1.ckpt && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/DiffBIR/resolve/main/general_full_v1.ckpt -d /content/DiffBIR/weights -o general_full_v1.ckpt && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/DiffBIR/resolve/main/face_swinir_v1.ckpt -d /content/DiffBIR/weights -o face_swinir_v1.ckpt && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/DiffBIR/resolve/main/general_swinir_v1.ckpt -d /content/DiffBIR/weights -o general_swinir_v1.ckpt && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/DiffBIR/resolve/main/BSRNet.pth -d /content/DiffBIR/weights -o BSRNet.pth && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/DiffBIR/resolve/main/scunet_color_real_psnr.pth -d /content/DiffBIR/weights -o scunet_color_real_psnr.pth && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/DiffBIR-v2/resolve/main/v1_face.pth -d /content/DiffBIR/weights -o v1_face.pth && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/DiffBIR-v2/resolve/main/v1_general.pth -d /content/DiffBIR/weights -o v1_general.pth && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/DiffBIR-v2/resolve/main/v2.pth -d /content/DiffBIR/weights -o v2.pth && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/stabilityai/stable-diffusion-2-1-base/resolve/main/v2-1_512-ema-pruned.ckpt -d /content/DiffBIR/weights -o v2-1_512-ema-pruned.ckpt

COPY ./worker_runpod.py /content/DiffBIR/worker_runpod.py
WORKDIR /content/DiffBIR
CMD python worker_runpod.py