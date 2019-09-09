# coding=utf-8
from __future__ import print_function
from __future__ import division
import cv2 as cv
import numpy as np
import argparse
from math import atan2, cos, sin, sqrt, pi
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
import math

"""
This script was developed by José Pereira, Universidade de Aveiro, 2019.
For more information and licensing issues, please contact: jose.manuel.pereira@ua.pt
"""

class IdentifyCells:
    def __init__(self, input_filename):
        self.draw_contour           = False         # Draw an outline surrounding the identified cells;
        self.contour_colour         = (7, 138, 245) # Colour of the contour;
        self.contour_thickness      = 1             # Thickness of the contour;
        self.draw_arrow             = True          # Draw an arrow depicting the oval orientation of the identified cells;
        self.arrow_scale            = 100           # Length of the arrow;
        self.arrow_colour           = (0, 210, 205) # Colour of the arrow;
        self.arrow_thickness        = 1             # Thickness of the arrow;
        self.max_area               = 180           # Max area allowed when identifying cells: if too big, 2 or more overlapping cells are counted as 1;
        self.min_area               = 0             # Minimum area allowed when identifying cells: useful to filter out artifacts;
        self.angle_hist_n_bins      = 50            # Number of bins for the angle distribution histogram;
        self.angle_hist_n_ticks     = 5             # Number of ticks for the Y axis of the distribution histogram;
        self.transparent_background = True          # Use transparent background when exporting results to a file.

        self.fig       = None
        self.hist_data = []
        self.read_image(input_filename)


    def read_image(self, input_filename):
        self.img = cv.imread(input_filename)
        # Convert image to grayscale
        gray = cv.cvtColor(self.img, cv.COLOR_BGR2GRAY)
        # Convert image to binary (BLACK and WHITE)
        _, bw = cv.threshold(gray, 1, 255, cv.THRESH_BINARY)
        # Apply median blur
        bw = cv.medianBlur(bw, 5)

        # Identify contours of cells
        contours, hier = cv.findContours(bw, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)

        self.angles = []
        self.cell_count = 0
        for i, c in enumerate(contours):
            
            # Calculate the area of each contour
            area = cv.contourArea(c);

            # Ignore contours that are too small or too large
            if area > self.max_area or area < self.min_area:
                continue

            if self.draw_contour:
                # Draw each contour only for visualisation purposes
                cv.drawContours(self.img, contours, i, self.contour_colour, self.contour_thickness);

            # Find and save the orientation of each shape
            self.getOrientation(c)
            self.cell_count += 1

        self.angles = filter(lambda x: x != 0.0, self.angles)
        print("Cell count: %d" % (self.cell_count))

    
    def getOrientation(self, pts):
        sz = len(pts)
        data_pts = np.empty((sz, 2), dtype=np.float64)
        for i in range(data_pts.shape[0]):
            data_pts[i,0] = pts[i,0,0]
            data_pts[i,1] = pts[i,0,1]
        # Perform PCA analysis
        mean = np.empty((0))
        mean, eigenvectors, eigenvalues = cv.PCACompute2(data_pts, mean)
        # Store the center of the object
        cntr = (int(mean[0,0]), int(mean[0,1]))
        p1 = (cntr[0] + - 0.02 * eigenvectors[0,0] * eigenvalues[0,0], cntr[1] + - 0.02 * eigenvectors[0,1] * eigenvalues[0,0])
        angle = atan2(eigenvectors[0,1], eigenvectors[0,0]) # orientation in radians
        self.angles.append(angle)
        
        if self.draw_arrow:
            self.drawAxis(cntr, p1)


    def drawAxis(self, p_, q_):
        p = list(p_)
        q = list(q_)
        
        angle = atan2(p[1] - q[1], p[0] - q[0]) # angle in radians
        hypotenuse = sqrt((p[1] - q[1]) * (p[1] - q[1]) + (p[0] - q[0]) * (p[0] - q[0]))
        # Here we lengthen the arrow by a factor of scale
        q[0] = p[0] - self.arrow_scale * hypotenuse * cos(angle)
        q[1] = p[1] - self.arrow_scale * hypotenuse * sin(angle)
        cv.line(self.img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), self.arrow_colour, self.arrow_thickness, cv.LINE_AA)
        # Create the arrow hooks
        p[0] = q[0] + 9 * cos(angle + pi / 4)
        p[1] = q[1] + 9 * sin(angle + pi / 4)
        cv.line(self.img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), self.arrow_colour, self.arrow_thickness, cv.LINE_AA)
        p[0] = q[0] + 9 * cos(angle - pi / 4)
        p[1] = q[1] + 9 * sin(angle - pi / 4)
        cv.line(self.img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), self.arrow_colour, self.arrow_thickness, cv.LINE_AA)


    def show(self, render = True):
        if self.fig == None:
            self.fig = plt.figure()
            self.plot_src()
            self.plot_angles()
        if render:
            plt.tight_layout()
            plt.show()


    def plot_src(self):
        ax = self.fig.add_subplot(121)
        ax.imshow(self.img)
        ax.yaxis.set_major_locator(ticker.NullLocator())
        ax.xaxis.set_major_locator(ticker.NullLocator())


    def calc_angles_hist(self):
        self.hist_data, self.bins = np.histogram(self.angles, bins=self.angle_hist_n_bins)
        self.centers = (self.bins[:-1] + self.bins[1:]) / 2


    def plot_angles(self):
        ax     = self.fig.add_subplot(122, polar=True)
        if len(self.hist_data) == 0:
            self.calc_angles_hist()
        width  = (self.bins[1] - self.bins[0])
        bars   = ax.bar(self.centers, self.hist_data, align = 'center', width = width)
        for r, bar in zip(self.hist_data, bars):
            bar.set_facecolor(plt.cm.jet(r/10.0))
            bar.set_alpha(0.8)

        n_ticks   = 5
        max_hist  = int(max(self.hist_data))
        tick_step = int(round(max_hist / self.angle_hist_n_ticks))
        ax.set_ylim(top = max_hist + tick_step)
        ax.yaxis.set_major_locator(ticker.MaxNLocator(n_ticks))
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.set_thetamin(-45)
        ax.set_thetamax(135)
        ax.set_xticks(np.array([-45, 0, 45, 90, 135]) / 180 * pi)
        ax.set_theta_direction(-1)
        ax.set_theta_zero_location("W")


    def export_hist(self, output_filename_hist):
        if len(self.hist_data) == 0:
            self.calc_angles_hist()
        with open(output_filename_hist, "w") as file_out:
            for value, center in zip(self.hist_data, self.centers):
                file_out.write("%7.3f %7d\n" % (np.rad2deg(center), value))

    
    def export_fig(self, output_filename_fig):
        if self.fig == None:
            self.show(render = False)
        self.fig.savefig("%s" % (output_filename_fig), dpi=1200, transparent = self.transparent_background)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Identify cells from microscopy images. For help or more information, contact: jose.manuel.pereira@ua.pt')
    parser.add_argument('-i', '--input', type = str, help = "Input image file", required = True)
    parser.add_argument('-oh', '--out_hist', type = str, help = "Output file name for histogram data", default = None)
    parser.add_argument('-of', '--out_fig', type = str, help = "Output file name for figure", default = None)

    args = parser.parse_args()
    data = IdentifyCells(args.input)
    if args.out_hist != None:
        data.export_hist(args.out_hist)
    if args.out_fig != None:
        data.export_fig(args.out_fig)
    data.show()