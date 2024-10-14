import kagglehub

# Download latest version
path = kagglehub.model_download("metaresearch/segment-anything/pyTorch/vit-l")

print("Path to model files:", path)