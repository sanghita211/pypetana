# Petridish Image Extraction Tool and Dynamic Feature Analysis in Biological Systems

## Introduction
This repository focuses on the extraction of growth images of biological organisms on petri dishes and the calculation of dynamic features such as _area_, _perimeter_, _circularity_ and _fractal dimension_.

## System 

### Linux/macOS Installation
Install pixi:

```console
curl -fsSL https://pixi.sh/install.sh | bash
```

Clone the repo and path to it:
```console
git clone https://github.com/bhimberg/pypetana.git
cd pypetana
```

Setup the pixi environment, launch a pixi shell and run pypetana:
```console
pixi shell
pypetana
```

## Methods

* Sample isolation through use of cropping and other tuning parameters
* Interpolation of tuning parameters across multiple images or video frames
* Extraction of area, perimeter of growth contour
* Extraction of circularity based on area and perimeter of growth contour
* Extraction of mass and boundary Fractal dimension of growth contour

## Acknowledgments

This research was undertaken in part thanks to funding from the Alfred P. Sloan Foundation
(Matter-to-Life program). I thank Nirosha J. Murugan and Philip Kurian for helpful discussions
with regards to Fricker Software and fractal dimensions, respectively.

## Contributing

* New features, bug fixes, documentation, tutorial examples, code testing is welcome in the developer community!
