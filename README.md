[![DOI](https://zenodo.org/badge/663112132.svg)](https://zenodo.org/doi/10.5281/zenodo.10645343)

# PadAnalyser
Repository with code for segmenting and plotting single-cell microscopy data from agar pads. 

## Getting started
- The package relies on Rust. Install using the command in [https://rustup.rs/](https://rustup.rs/) before running `pip install`.
- If you just want to use the package, you can download it using pip: `pip install -U "git+https://github.com/Cicuta-Group/PadAnalysis.git"`
- If you want to edit the source code, you can clone the repository and install it in editable mode: `pip install -e .`. The package can then be accessed from any directory.

## Usage
1. Make your FrameSet - this is an object that contains all information required to process one field of view on a pad, including time-series images, timestamps, and a dictionary with metadata. Labs store theur data in different ways, so you can subclass the FrameSet abstract class to support your particular format. If your images are Tiffs, we have included a TiffFrameSet subclass you can use. 
2. Make your OutputConfig - this is an object that specifies where output files should be printed, what debug information should be generated and where.
