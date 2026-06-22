import torch

from xctom.model_unet import UNet2D


def test_unet_forward_shape():
    model = UNet2D(
        input_channels=1,
        output_channels=1,
        base_channels=16,
        depth=4,
        dropout=0.1,
    )

    x = torch.randn(2, 1, 256, 256)

    with torch.no_grad():
        y = model(x)

    assert y.shape == (2, 1, 256, 256)
    assert torch.isfinite(y).all()
    assert float(y.min()) >= 0.0
    assert float(y.max()) <= 1.0