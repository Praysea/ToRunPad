#!/usr/bin/env bash

# Start SSH server if PUBLIC_KEY is set (enables remote access and dev-sync.sh)
if [ -n "$PUBLIC_KEY" ]; then
    mkdir -p ~/.ssh
    echo "$PUBLIC_KEY" > ~/.ssh/authorized_keys
    chmod 700 ~/.ssh
    chmod 600 ~/.ssh/authorized_keys

    # Generate host keys if they don't exist (removed during image build for security)
    for key_type in rsa ecdsa ed25519; do
        key_file="/etc/ssh/ssh_host_${key_type}_key"
        if [ ! -f "$key_file" ]; then
            ssh-keygen -t "$key_type" -f "$key_file" -q -N ''
        fi
    done

    service ssh start && echo "worker-comfyui: SSH server started" || echo "worker-comfyui: SSH server could not be started" >&2
fi

# Use libtcmalloc for better memory management
TCMALLOC="$(ldconfig -p | grep -Po "libtcmalloc.so.\d" | head -n 1)"
export LD_PRELOAD="${TCMALLOC}"

# ---------------------------------------------------------------------------
# GPU pre-flight check
# Verify that the GPU is accessible before starting ComfyUI. If PyTorch
# cannot initialize CUDA the worker will never be able to process jobs,
# so we fail fast with an actionable error message.
# ---------------------------------------------------------------------------
echo "worker-comfyui: Checking GPU availability..."
if ! GPU_CHECK=$(python3 -c "
import torch
try:
    torch.cuda.init()
    name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    # Launch a real kernel. The driver-only calls above succeed even when this
    # PyTorch build has no compiled kernels for the GPU architecture (e.g. an
    # older torch on a newer GPU). Without this, the worker boots, ComfyUI dies
    # on the first GPU op, and it surfaces as the misleading 'server not
    # reachable' error instead of a clear cause here.
    _ = (torch.zeros(8, device='cuda') + 1).sum().item()
    torch.cuda.synchronize()
    print(f'OK: {name} (sm_{cap[0]}{cap[1]}), torch {torch.__version__}, cuda {torch.version.cuda}')
except Exception as e:
    print(f'FAIL: {e}')
    exit(1)
" 2>&1); then
    echo "worker-comfyui: GPU is not available or incompatible with this PyTorch build:"
    echo "worker-comfyui: $GPU_CHECK"
    echo "worker-comfyui: A 'no kernel image is available' error means this torch build"
    echo "worker-comfyui: lacks kernels for this GPU. Otherwise the GPU may not be"
    echo "worker-comfyui: properly initialized — please contact RunPod support."
    exit 1
fi
echo "worker-comfyui: GPU available — $GPU_CHECK"

# Ensure ComfyUI-Manager runs in offline network mode inside the container
comfy-manager-set-mode offline || echo "worker-comfyui - Could not set ComfyUI-Manager network_mode" >&2

# ---------------------------------------------------------------------------
# Flatten models that arrived through RunPod "Cached models"
#
# Cached models are delivered as a plain HuggingFace hub cache whose layout is
#   <hub>/models--<org>--<repo>/snapshots/<commit>/<file>
# ComfyUI resolves a ckpt_name by joining a registered folder with the name we
# send in the workflow, so using that path would mean hard-coding a commit hash
# for every model revision. Symlink the weights into one flat directory instead
# and let src/extra_model_paths.yaml register that directory.
# ---------------------------------------------------------------------------
LINK_ROOT="/runpod-modellinks"
mkdir -p "$LINK_ROOT/checkpoints" "$LINK_ROOT/loras"

MODEL_COUNT=0
for HUB in "${HF_HOME:-/runpod-volume/huggingface-cache}/hub" \
           "/runpod-volume/huggingface-cache/hub" \
           "/root/.cache/huggingface/hub"; do
    [ -d "$HUB" ] || continue
    # Snapshots are usually symlinks into blobs/, so -type l has to be included.
    while IFS= read -r -d '' f; do
        case "$f" in
            *.safetensors|*.ckpt|*.pt|*.pth|*.bin) ;;
            *) continue ;;
        esac
        # Weights and LoRAs are both .safetensors with nothing in the name to tell
        # them apart, so link each file into both folders and let either loader
        # find it.
        ln -sf "$f" "$LINK_ROOT/checkpoints/" 2>/dev/null
        ln -sf "$f" "$LINK_ROOT/loras/" 2>/dev/null
        MODEL_COUNT=$((MODEL_COUNT + 1))
    done < <(find "$HUB" -maxdepth 5 -path '*/snapshots/*/*' \( -type f -o -type l \) -print0 2>/dev/null)
done

if [ "$MODEL_COUNT" -gt 0 ]; then
    echo "worker-comfyui: Linked $MODEL_COUNT cached model file(s) into $LINK_ROOT:"
    ls -1 "$LINK_ROOT/checkpoints"
else
    echo "worker-comfyui: No cached models found under /runpod-volume/huggingface-cache/hub" >&2
    echo "worker-comfyui: Add a model to the endpoint's Cached models, or mount a network volume." >&2
fi

echo "worker-comfyui: Starting ComfyUI"

# Allow operators to tweak verbosity; default is DEBUG.
: "${COMFY_LOG_LEVEL:=DEBUG}"

# PID file used by the handler to detect if ComfyUI is still running
COMFY_PID_FILE="/tmp/comfyui.pid"

# Serve the API and don't shutdown the container
if [ "$SERVE_API_LOCALLY" == "true" ]; then
    python -u /comfyui/main.py --disable-auto-launch --disable-metadata --listen --verbose "${COMFY_LOG_LEVEL}" --log-stdout &
    echo $! > "$COMFY_PID_FILE"

    echo "worker-comfyui: Starting RunPod Handler"
    python -u /handler.py --rp_serve_api --rp_api_host=0.0.0.0
else
    python -u /comfyui/main.py --disable-auto-launch --disable-metadata --verbose "${COMFY_LOG_LEVEL}" --log-stdout &
    echo $! > "$COMFY_PID_FILE"

    echo "worker-comfyui: Starting RunPod Handler"
    python -u /handler.py
fi