# Cell Orientation Identificator
This simple script allows the user to obtain both a .png file showing the identified cells and their oval orientations, as well as a graph with the polar histogram of the orientations of all cells in the provided input file.

How to:
Simply provide an input picture of some cells or blobs (for example, from a microscope).
On a command line, run:
```console
python identify_cells.py -i input_file.tiff -oh histogram.dat -of results.svg
```

Open the script and change the input variables (lines 14-25) to change the overall aesthetic look of your results.

## Contact
for more information, please contact jose.manuel.pereira@ua.pt
