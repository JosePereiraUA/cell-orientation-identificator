# Cell Orientation Identificator
This simple script allows the user to obtain both a .png file showing the identified cells and their oval orientations, as well as a graph with the polar histogram of the orientations of all cells in the provided input file.

### Instructions

Simply provide an input picture of some cells or blobs (for example, from a microscope).
On a command line, run:
```console
python identify_cells.py -i input_file.tiff -oh histogram.dat -of results.svg
```

Open the script and change the input variables (lines 14-25) to change the overall aesthetic look of your results.

### Google Colab

In order to bypass installing all necessary pre-requesite packages, you can use the Cell Orientation Identificator directly on [Google Colab](https://colab.research.google.com/drive/1Vrskhx9Ig5FBTMSBu5oJVrhoPrr7pVaA#scrollTo=hZVFcaNN4URt)!

## Contact
For more information or help, please contact jose.manuel.pereira@ua.pt
