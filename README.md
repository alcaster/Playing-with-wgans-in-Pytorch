# Playing with Wgans and Gans + Flexible Checkpoint Saver for Pytorch
## Dataset - [Color FERET](https://www.nist.gov/itl/iad/image-group/color-feret-database)
FERET database of facial images collected under the FERET program, sponsored by the DOD Counterdrug Technology
Development Program Office
## Main Objectives
- [x] Understand gans and wgans.
- [x] Obtain and create Dataloader for images of faces.
- [x] Create wgan for creating faces.
- [x] Create flexible [Checkpoint Saver](utils.py) for pytorch.
- [x] Compare dense layers with deconvolution layers.
- [ ] Create wgan for text.
- [ ] Visualize training f.ex loss, compare with other types of models.
- [ ] Compare Conditional improved WGAN which replace cliping gradient with penalty.
### Requirements
- Python >= 3.6
- Pytorch >= 0.4
- Color Feret Dataset
