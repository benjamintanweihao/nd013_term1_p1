import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import sys
from scipy import stats


class LaneLines:
    """ Finding lane lines on the road"""

    # TODO: change `image_path` to *NOT* hardcode
    def __init__(self, image_path="test_images/solidWhiteCurve.jpg"):
        self.image_path = image_path

    def process(self):
        """Reads an image from `image_path` and outputs another image
        with the lane markings."""

        # 1. Read the image
        image = mpimg.imread(self.image_path)

        # 2. Convert image to grayscale
        gray = self.grayscale(image)

        # 3. Apply Gaussian smoothing
        blur_gray = self.gaussian_blur(gray, 5)

        # 4. Apply Canny edge detection
        edges = self.canny(blur_gray, 50, 100)

        # 5. Define a four side polygon to mask
        vertices = np.array([[(449, 290), (503, 295), (890, 537), (84, 537)]], dtype=np.int32)
        masked_edges = self.region_of_interest(edges, vertices)

        # 6. Apply Hough on edge detected image
        rho = 1
        theta = np.pi / 180
        threshold = 1
        min_line_len = 15
        max_line_gap = 10

        lines_image = self.hough_lines(masked_edges, rho, theta, threshold, min_line_len, max_line_gap)

        # Combine the lines image with the lines image
        lines_edges = cv2.addWeighted(image, 0.6, lines_image, 1, 0)

        # plt.imshow(masked_edges)
        plt.imshow(lines_edges)
        plt.show()

    # Helper functions

    def grayscale(self, img):
        """Applies the Grayscale transform
        This will return an image with only one color channel
        but NOTE: to see the returned image as grayscale
        (assuming your grayscaled image is called 'gray')
        you should call plt.imshow(gray, cmap='gray')"""
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # Or use BGR2GRAY if you read an image with cv2.imread()
        # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    def canny(self, img, low_threshold, high_threshold):
        """Applies the Canny transform"""
        return cv2.Canny(img, low_threshold, high_threshold)

    def gaussian_blur(self, img, kernel_size):
        """Applies a Gaussian Noise kernel"""
        return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

    def region_of_interest(self, img, vertices):
        """
        Applies an image mask.

        Only keeps the region of the image defined by the polygon
        formed from `vertices`. The rest of the image is set to black.
        """
        # defining a blank mask to start with
        mask = np.zeros_like(img)

        # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
        if len(img.shape) > 2:
            channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
            ignore_mask_color = (255,) * channel_count
        else:
            ignore_mask_color = 255

        # filling pixels inside the polygon defined by "vertices" with the fill color
        cv2.fillPoly(mask, vertices, ignore_mask_color)

        # returning the image only where mask pixels are nonzero
        masked_image = cv2.bitwise_and(img, mask)
        return masked_image

    def draw_lines(self, img, lines, color=[255, 0, 0], thickness=6):
        """
        NOTE: this is the function you might want to use as a starting point once you want to
        average/extrapolate the line segments you detect to map out the full
        extent of the lane (going from the result shown in raw-lines-example.mp4
        to that shown in P1_example.mp4).

        Think about things like separating line segments by their
        slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
        line vs. the right line.  Then, you can average the position of each of
        the lines and extrapolate to the top and bottom of the lane.

        This function draws `lines` with `color` and `thickness`.
        Lines are drawn on the image inplace (mutates the image).
        If you want to make the lines semi-transparent, think about combining
        this function with the weighted_img() function below
        """
        left_lines = []
        right_lines = []

        # NOTE: Most likely we need to passes at the slope!
        lower_slope_threshold = 0.50
        upper_slope_threshold = 0.80

        epsilon = 10 ** -7

        slopes = []

        for line in lines:
            for x1, y1, x2, y2 in line:
                slope = ((y2 - y1) / (x2 - x1 + epsilon))
                slopes.append(slope)

                if lower_slope_threshold <= slope <= upper_slope_threshold:
                    right_lines.append(line)

                if -upper_slope_threshold <= slope <= -lower_slope_threshold:
                    left_lines.append(line)

        print("slopes =", sorted(slopes))
        print(stats.mode(slopes))

        self.polyfit_line(left_lines, img, color, thickness)
        self.polyfit_line(right_lines, img, color, thickness)

    def polyfit_line(self, lines, img, color, thickness):
        xs = []
        ys = []

        start_x = sys.maxsize
        start_y = sys.maxsize

        for line in lines:
            for x1, y1, x2, y2 in line:
                # collect all the x and y
                xs.append(x1)
                ys.append(y1)

                xs.append(x2)
                ys.append(y2)

                if y1 < y2 and y1 < start_y:
                    start_x = x1
                    start_y = y1

                if y2 < y1 < start_y:
                    start_x = x2
                    start_y = y2

        # find the best fit line
        [m, c] = np.polyfit(np.array(xs, dtype="float"),
                            np.array(ys, dtype="float"), deg=1)

        y_size = img.shape[0]

        end_x = int((y_size - c) / m)
        end_y = y_size

        cv2.line(img, (start_x, start_y), (end_x, end_y), color, thickness)

    def hough_lines(self, img, rho, theta, threshold, min_line_len, max_line_gap):
        """
        `img` should be the output of a Canny transform.

        Returns an image with hough lines drawn.
        """
        lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]),
                                minLineLength=min_line_len,
                                maxLineGap=max_line_gap)
        line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

        # TODO: will need to change this in the juypter notebook
        self.draw_lines(line_img, lines)
        return line_img

    # Python 3 has support for cool math symbols.

    def weighted_img(self, img, initial_img, α=0.8, β=1., λ=0.):
        """
        `img` is the output of the hough_lines(), An image with lines drawn on it.
        Should be a blank image (all black) with lines drawn on it.

        `initial_img` should be the image before any processing.

        The result image is computed as follows:

        initial_img * α + img * β + λ
        NOTE: initial_img and img must be the same shape!
        """
        return cv2.addWeighted(initial_img, α, img, β, λ)


ll = LaneLines()
ll.process()
