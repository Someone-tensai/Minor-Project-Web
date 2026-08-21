import torch.nn.functional as F
import numpy as np
import cv2


class GradCam:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor, class_idx=None):
        self.model.eval()

        # Ensure gradients can flow to the target layer even if the
        # backbone was frozen during training (requires_grad=False on params).
        input_tensor = input_tensor.clone().requires_grad_(True)

        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        self.model.zero_grad()

        loss = output[0, class_idx]
        loss.backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        cam = cam.squeeze().detach().cpu().numpy()
        cam = cv2.resize(cam, (input_tensor.shape[3], input_tensor.shape[2]))
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        return cam, class_idx, output


def get_target_layer(model, arch):
    if arch in ("resnet18", "resnet50"):
        return model.layer4[-1]
    if arch == "vgg16":
        return model.features[-1]
    raise ValueError(f"No target layer defined for {arch}")


def make_overlay(rgb_image: np.ndarray, cam: np.ndarray) -> np.ndarray:
    """
    rgb_image: HxWx3 uint8 RGB array (the original uploaded image, resized
               to match the model input size used for the cam)
    cam:       HxW float array in [0, 1] from GradCam.generate()
    returns:   HxWx3 uint8 RGB overlay
    """
    heatmap_color = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(rgb_image, 0.6, heatmap_color, 0.4, 0)
    return overlay
