"""
This script calculates 'blob' sizes of irregular shapes.
It is intended to be applied to "Splat Test" images of Ice Recrystallisation Inhibition (IRI) images.
The script was put together for use within the Gibson Group. Check us out at:

 - http://www2.warwick.ac.uk/fac/sci/chemistry/research/gibson/gibsongroup/

Info about the author:

 - http://www2.warwick.ac.uk/fac/sci/moac/people/students/2013/joseph_healey

Here's an ASCII art representation of one! Notice the blobs and voids. Tricky images to work with.
The statistic of interest is the MLGS (mean largest grain size) when considering ice crystals.

The script uses computer vision techniques to identify structures within the image, and then simple
bounding box arithmetic to calculate sizes.

+--------------------------------------------------------------------------------------------------------+
|+------------------------------------------------------------------------------------------------------+|
||                                                                                                      ||
|| :+ooo+`     ``````````   ````..` ```````````  ````````    ``  ```                        `           ||
||  `./+.  ``  `````....`    ```   ```....```..  ``.....``` .::.-. -``                                  ||
||      ``-://-` `...----`        `...........-`  .`---.````/++/.                                       ||
||      `/+ooooo/.```..`  ```  ` `..........--:`  .-:::`...-oo/` `-:.              `.``..               ||
||      `//oyyyssso/-   `..`` ```.--.------::.``...-:::`...::-.-/oyys+:-.        .-/+oosy+``.`          ||
||      `+oossyyyyyso. .--...`.:..------:::-`.-/+++/../`` `-/osyyyhhhhys+/-` `.-.`:syyo/+s: `` ``       ||
||   ``./syyyyyhhyy+/`.....--.-:`.--:::::.`::+oooooo+-  .oyyhhhhhhhhhhhhyso/-/yhs+/oso+.``        ````  ||
|| .`   .osyyyyysyy++-`..-------:.`-:///:..++ooosssssso-.:yhhhhhhhhhhhhhhhyyyhyshhs/```..`.....`    ``` ||
|| -:-` `/shhhh+/+/`-..-----::::/:`:+/+/`-++oossssssso+.`:yhdddhhhhhhhhhhhhhyyhs./y:.-/--.-----.`   ..` ||
|| ::::-``.//:-:/+-  `----::/:-..` ..`.` ./+ossssssso+o-`+yhddddddddddhhhhhhhhhyo:`:++:--::------`  `.. ||
|| :////:     `--.`` `-:/::..-/oyyo/ `+so/+oosyyyyysoos:.oyhhdddddddddddddhhhhhyso-++:--::::-----.`  .. ||
|| //++/.     ````.-. ..``/shddddmdh:oddddhysosyyyoosso/-osyyhdddddddddddddddddhys-//:///::::::----`  . ||
|| ++++.       `` `.--::`/ddmmmmmmmdoddmmmmdddhs/-`:oo/`.+syohddddddddddddddddddhs`:+/////:::::::::.  ` ||
|| /++-`         `-`.:/:./dmmmmmmmmdyddmmmmmmmmddy//:.` `.:-.+syhhhdddddddddddddy+.:+o++//////////::- ` ||
|| +/:.`         `. .///:./ydmmmmmmhhddmmmmmmmmmmmhys:`:++::/:--ohdhhhhdddddddho-/+:/oo+++//+++///:-.   ||
|| +/.          ```./++++//+shdmmmmyhmmmmmmmmmmmmmds:`:/+syhhhyyyooyhhhhhddddy/  `:``/ssso+//:-..--.`   ||
|| ``         ``.-`:+++++++oos/sso+--ymmmmmmmmmdhs/.``-.sddddddhhhyssosy///:-..:-+yyo///+:--....```     ||
||            ` ..:oooooosoooo..`..``-+ydmmmdhs+-..```:osddddddddhyyy+``       .+dddmd+:-..`            ||
||        `````  .+oooosssoo/.`      `..-/++:-.````````/ohdmmdddddhhy/`        `/sdddhhy+.              ||
||       ````..``:oooossoo/-.`          ` `````````````-/sddmmdddddhs+:        `ohdmdhy+.               ||
||    ```````..`.+ooosoo/-.``          `  ``..`````````.-+hddddddddyoo-`        /hhmhh+.                ||
||  `````````..`.+osso+:-.``              ```..`````````-/hdddddddhss/.```````` -:+s//.`                ||
|| ``````````.-..+oo+:--.              `:+++-.`.......``-+hdmmmmddyy+-``````````.-ss.``                 ||
|| ``````````.-.`//:::.`              .odddmdy+-.`.......:shdddhys//-.````````.+shhhs/.`              ` ||
|| `````````..:.``.--`               -ymmmmmmmmhs+-.``... `-+++/:.``````````.:sddddddho-`            `: ||
|| ```````.----`. `.`               `ymmmmmmmmmmmmdy/```   `````````  `` `-/sddmmmmmddho-`           .+ ||
|| ``````.--..:/+`.:.`     ``.```--.-ydmmmmmmmmmmmmmy-  ``````````````   .shdmmmmmmdddhy``  ```     `-+ ||
|| //:.`.-...-+oss-//:.``.-://///++/oyhmmmmmmmmmmmmmh/`  ````````````````-hddmmmmmddhyo:         `` -o/ ||
|| sss+...---://++--:-.://++++++o+++/yhhmmmmmmmmmmmdo- ````````````````` .yddmmmddhs/.`         `/:..o+ ||
|| syysso/::-..-::oss+-/+++++oooooo+::syhmmmmmmmmdy:` ```````````````````..+hdddho:`            ./:. :o ||
|| //yhhhyo.`:/osssso//-:++++oooo+++:-/ysdmmmmmh+:--..`  ```````````````..``-:/:.`              `...``` ||
|| /syhhhhho/ooooss+::--.+++++++++++//+s+ommdho-:+ooo+/-.` `````````` ````.``                 `.:ossoo+ ||
|| +yhhhhhho+ooooss+///:`-+oo+++oooooo+sssdh.-:ossssssooo/:.   ``..-....`                    `/shhhhhyy ||
|| :oyhhhds-+ossoss+//+/  -/oooosssoosooyhh+`/ossssssssssss+. `/+ossooooo+/-                 /hhhhhhhhy ||
|| ::+osys.`/ossoooo/::..-..-/ooso++++//-:. -osssssssssyssss/`/yyyyyyyyyyyys/`              .yhhhhhhhhh ||
|| ////:/:` :+ooo+o+:.`.://++/:++osyyyys+:`./ossssssssyyysss+`:yhhhhhhhhhhhyy+`        `-/::syhhhhhhhhh ||
|| ////::/.`:+://:--:----:oo+yy++yyyyyhyyo:-/osssssssssssssoo-`shhhhhhhhhhyyys.`..  `/shddhddhyhhhhhhhh ||
|| -:::::-.:..../osssso/::-.-//--ossyyyyo/-./osssssssssssssso+.:yyysooo+++/::-`/yy+-odddddddddyyhhhhhhh ||
|| ://++/../`.-+syyyyyyso+:.-.`..`-/+syyo+.`/ossssssssssysssso/`.-/:::::::-:--.shyo+yddddddddd/:oyhhhhh ||
|| hyyyoo:/+`.:+ssyyyyyyys/``/+/+:-::/+oo/- ./osssssssssssssso+``.:/+oosssssso-+ss/:ohddddddyo--:/syyyy ||
|| dmmmdhhs+.-:/ossyyyyyso.`:+oosssssyh+:+s+`.+ssssssssssssso+--:/syyyyyyyyyss- .:..-ohddhyyhy+:/osysso ||
|| dmmmmmmmds-::+osyyyss+-`+syyyyysyyyhhoyyys.:ossssssooossoo:`-:oyyyyhhhhyyss:.+sssyhhs/oyso+++++++++/ ||
||                                                                                                      ||
|+------------------------------------------------------------------------------------------------------+|
+--------------------------------------------------------------------------------------------------------+

### CHANGELOG: ###

03-09-2017 | Version 1.0 | Joe Healey <j.r.j.healey@warwick.ac.uk>
    * Initial commit with basic functionality

#TODO:
 - Report length sizes, rather than pixel sizes OR include a calculator?
  * Require user to specify pixel dimensions
  * Figure it out from a scale bar?

"""




__author__ = "J. R. J. Healey"
__version__ = "1.0"
__title__ = "PrecICE.py"
__author_email__ = "J.R.J.Healey@warwick.ac.uk"
__license__ = "GPLv3"


# This program is free software: you can redistribute it and/or
# modify it under the terms of the GNU General Public License as
# published by the Free Software Foundation, either version 3 of
# the License, or (at your option) any later version.
# If it is used or modified for any work, please give credit.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.


# Program begins...

# Module imports (many are non-standard library and will need to be installed).

import imutils

from imutils import contours
from imutils import perspective
from scipy.spatial import distance as dist
import scipy.misc as sm
import matplotlib as mpl
mpl.use("TkAgg")
from matplotlib import pyplot as plt
import numpy as np
import cv2
import argparse
import traceback
import sys
import logging
custom_log = logging.getLogger().setLevel(logging.INFO)



def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

def skeletonise(image):
    """Skeletonise an image using OpenCV operations"""

    size = np.size(image)
    skel = np.zeros(image.shape, np.uint8)

    ret, image = cv2.threshold(image, 127, 255, 0)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    done = False

    while (not done):
        skerode = cv2.erode(image, element)
        temp = cv2.dilate(skerode, element)
        temp = cv2.subtract(image, temp)
        skel = cv2.bitwise_or(skel, temp)
        image = skerode.copy()

        zeros = size - np.count_nonzero(image)
        if zeros == size:
            done = True
            return skel


try:
    parser = argparse.ArgumentParser(
            description='Computer vision program for feature size determination. '
                        'Utilises computer vision and image analysis to return '
                        'approximated feature dimensions.',
            usage='python PrecICE.py [-h|--help] [options] IMAGEFILE')

    parser.add_argument('image',
                        action='store',
                        metavar='IMAGEFILE',
                        help='The image file to be analysed.')
    parser.add_argument('--gkernel',
                        action='store',
                        type=int,
                        default=5,
                        metavar='N',
                        help='Kernel size for Gaussian Blurring. [Def 5]')
    parser.add_argument('--gsigma',
                        action='store',
                        type=int,
                        default=0,
                        metavar='N',
                        help='Standard deviation of Gaussian Blur kernel. [Def 0]')
    parser.add_argument('--ignore_intermediates',
                        action='store_true',
                        help='Output intermediate images from processing. [Def off]')
    parser.add_argument('--skip_denoise',
                        action='store_true',
                        help='Skip the Gaussian blur de-noising step.'
                             'If the image has well defined edges it may not be necessary '
                             'or even offer and improvement over the with-blur processing.')

except NameError:
    print "An exception occured with argument parsing. Check your provided options."
    traceback.print_exc()
    sys.exit(1)


args = parser.parse_args()

print('"PrecICE" Image analysis tool, version ' + __version__ + " by " + __author__ + ' <' + __author_email__ + '>\n')

print('Analysing image: %s \n' %args.image)

# List of intermediate images to store for later
intermediate_images = []
intermediate_imagenames = []

# IMAGE PREPROCESSING:
logging.info(' (Step 1/) Loading image: %s' %args.image)
# 1. Read image in to numerical array
image = cv2.imread(args.image)
dims = np.size(image)

intermediate_images.append(image)
intermediate_imagenames.append('Original Image')
logging.info(' (Step 2/) Original image stored...')


# IMAGE PREPROCESSING:
print('\nPreprocessing beginning...\n')
logging.info(' (Step 3/) Image converted to 8-bit greyscale...')
greyed = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
intermediate_images.append(greyed)
intermediate_imagenames.append('Greyed Image')
logging.info(' (Step 4/) Greyed image stored...')


if args.skip_denoise is False:
    logging.info(' (Step 5/) Gaussian Filtering the image with a %d x %d kernel (standard deviation %d)' %(args.gkernel, args.gkernel, args.gsigma))
    # Gaussian blur de-noises the image. Parameters are kernel size (5,5) and st. dev (0)
    blurred = cv2.GaussianBlur(greyed, (args.gkernel, args.gkernel), args.gsigma)
    intermediate_images.append(blurred)
    intermediate_imagenames.append('Gaussian Filtered Image')
    logging.info(' (Step 6/) Gaussian filtered image stored...')
else:
    logging.info(' (Step 6/) --- SKIPPING STEP 6 DE-NOISING ---')


#### NOT SURE IF TO BINARISE OR EDGE DETECT???

logging.info(' (Step 7/) Binary thresholding the image (lower 127, upper 255)...')
retval, binary = cv2.threshold(blurred,30 ,200, cv2.THRESH_BINARY)
intermediate_images.append(binary)
intermediate_imagenames.append('Binarized Image')
logging.info(' (Step 7/) Binary threshold image stored...')


# Canny edge detector
logging.info(' (Step 8/) Detecting edges with Canny edge detector...')
edged = cv2.Canny(blurred, 30, 200)
intermediate_images.append(edged)
intermediate_imagenames.append('Edged Image')
logging.info(' (Step 9/) Edged image stored...')

logging.info(' (Step 10/) Beginning ...')
dilated = cv2.dilate(edged, None, iterations=1)
intermediate_images.append(dilated)
intermediate_imagenames.append('Dilated Image')

eroded = cv2.erode(dilated, None, iterations=1)
intermediate_images.append(eroded)
intermediate_imagenames.append('Eroded Image')



# Next steps:
# - complete the contour of the blobs (if erode-dilate isn't sufficient)
# - fill the contour areas
# - find pixel values > 100 etc (just blobs)
# -


# Output

# ## Images
# if args.ignore_intermediates is False:
#     for name, imageobj in zip(intermediate_imagenames, intermediate_images):
#         cv2.imshow(name, imageobj)
#
#     cv2.waitKey(0)

# Panel images:

# OpenCV's BGR encoding needs to be converted to RGB for matplotlib to display correctly:
mpl_converted_intermediates = [sm.toimage(image) for image in intermediate_images]
import math
for i in xrange(len(zip(intermediate_imagenames, mpl_converted_intermediates))):
    rows = math.ceil(float(len(mpl_converted_intermediates))/2)
    plt.subplot(rows,2,i+1)
    plt.imshow(mpl_converted_intermediates[i])
    plt.title(intermediate_imagenames[i])
    plt.xticks([]),plt.yticks([])
plt.show()


## Numerics










#
# # Contour finding
# cnts = cv2.findContours(eroded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#
# cnts = cnts[0] if imutils.is_cv2() else cnts[1]
#
# (cnts, _) = contours.sort_contours(cnts)
#
#
# # loop over the contours individually
# for c in cnts:
#     # if the contour is not sufficiently large, ignore it
#     if cv2.contourArea(c) < 10:
#         continue
#
#     # compute the rotated bounding box of the contour
#     orig = image.copy()
#     box = cv2.minAreaRect(c)
#     box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
#     box = np.array(box, dtype="int")
#
#     # order the points in the contour such that they appear
#     # in top-left, top-right, bottom-right, and bottom-left
#     # order, then draw the outline of the rotated bounding
#     # box
#     box = perspective.order_points(box)
#     cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)
#
#     # loop over the original points and draw them
#     for (x, y) in box:
#         cv2.circle(orig, (int(x), int(y)), 5, (0, 0, 255), -1)
#
#     # unpack the ordered bounding box, then compute the midpoint
#     # between the top-left and top-right coordinates, followed by
#     # the midpoint between bottom-left and bottom-right coordinates
#     (tl, tr, br, bl) = box
#     (tltrX, tltrY) = midpoint(tl, tr)
#     (blbrX, blbrY) = midpoint(bl, br)
#
#     # compute the midpoint between the top-left and top-right points,
#     # followed by the midpoint between the top-righ and bottom-right
#     (tlblX, tlblY) = midpoint(tl, bl)
#     (trbrX, trbrY) = midpoint(tr, br)
#
#     # draw the midpoints on the image
#     cv2.circle(orig, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
#     cv2.circle(orig, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
#     cv2.circle(orig, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
#     cv2.circle(orig, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)
#
#     # draw lines between the midpoints
#     cv2.line(orig, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),
#              (255, 0, 255), 2)
#     cv2.line(orig, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),
#              (255, 0, 255), 2)
#
#
#
#     # compute the Euclidean distance between the midpoints
#     dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
#     dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))  # compute the size of the object
#
#
# # draw the object sizes on the image
#     cv2.putText(orig, "{:.1f}in".format(dA),
#         (int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX,
#         0.65, (255, 255, 255), 2)
#     cv2.putText(orig, "{:.1f}in".format(dB),
#         (int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX,
#         0.65, (255, 255, 255), 2)
#
# for i in intermediate_images:
#     cv2.imshow(str(i), i)
#
# # show the output image
# cv2.imshow("Final", orig)
# cv2.waitKey(0)
