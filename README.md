# XCT-to-OM U-Net

Supervised baseline framework for translating 2D X-ray computed tomography (XCT) slices into corresponding optical microscopy (OM)-like images using a U-Net architecture.

This repository is part of a broader 2D multimodal data-fusion workflow, where OM acts as an intermediate bridge between XCT and chemical/compound mapping modalities such as Raman microscopy.

## Project Goal

The goal is to learn a paired image-to-image mapping:

```text
XCT slice → predicted OM image
```

The model is trained using spatially aligned XCT–OM image pairs. During training, the XCT image is used as input and the corresponding real OM image is used as the supervised target. After training, the model can infer an OM-like image from a new XCT slice.

This repository currently implements the first baseline step:

```text
XCT → U-Net → OM-like prediction
```

Adversarial models such as pix2pix may be added later after the U-Net baseline has been validated.

## Scientific Context

XCT and OM provide complementary contrast mechanisms. XCT is mainly sensitive to X-ray attenuation, density, and internal structural features, while OM captures surface optical contrast, color, texture, and sample preparation effects.

Therefore, this model should not be interpreted as a physical reconstruction of OM from XCT. Instead, it learns a data-driven, supervised XCT-conditioned prediction of an OM-like representation.

The validity of this approach depends strongly on:

* accurate XCT–OM registration,
* consistent preprocessing and normalization,
* careful train/validation/test splitting,
* evaluation beyond visual appearance,
* assessment of whether predicted OM preserves scientifically relevant structures.

## Current Status

Implemented:

* 2D U-Net model for XCT-to-OM prediction
* PyTorch-based project structure
* Pixi environment setup
* smoke test for model import and forward pass
* GitHub-ready repository structure

Planned:

* paired XCT–OM dataset loader
* training loop
* validation metrics
* prediction export
* checkpointing
* loss monitoring
* comparison against pix2pix or other image-to-image models


The `data/` and `outputs/` folders are intentionally excluded from Git tracking.

## Installation

This project uses Pixi for environment management.

From the project root:

```powershell
pixi install
```

To enter the environment:

```powershell
pixi shell
```

## Running Tests

Run the smoke test:

```powershell
pixi run test
```

Expected output:

```text
1 passed
```

The current smoke test verifies that the U-Net can be imported, receives an input tensor, and returns an output tensor with the expected shape.

## Model

The implemented model is a 2D U-Net.

The U-Net consists of:

* an encoder path that extracts increasingly abstract image features,
* a bottleneck representation,
* a decoder path that reconstructs the output image,
* skip connections that transfer spatial information from encoder to decoder.

This architecture is appropriate for paired image-to-image translation tasks where input and output images share spatial structure but differ in contrast or appearance.

## Author

**Arvin (Fazel) Mirzaei**
Postdoctoral Researcher
Paul Scherrer Institute, Switzerland

## License

License to be added.
