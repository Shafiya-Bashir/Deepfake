# FaceForensics++ (FF++) Dataset Specification

## Overview
- **Paper**: FaceForensics++: Learning to Detect Manipulated Facial Images (Rossler et al., ICCV 2019)
- **Source**: 1,000 pristine YouTube video clips (H.264).
- **Manipulations**:
  1. Deepfakes (autoencoder swap)
  2. Face2Face (expression reenactment)
  3. FaceSwap (3D facial landmark mesh rendering)
  4. NeuralTextures (neural texture rendering on mouth area)
  5. FaceShifter (occlusion-aware two-stage face swapping)

## Compressions
- **Raw**: Uncompressed ProRes 422
- **c23**: High quality (visually lossless)
- **c40**: Heavy compression (realistic social media transmission)

## Directory Target

