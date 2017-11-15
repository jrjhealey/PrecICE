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

"""
"""
TODO
 - Report length sizes, rather than pixel sizes OR include a calculator?
  * Require user to specify pixel dimensions
  * Figure it out from a scale bar?
  
 - Switch Gaussian to bilateral filtering.
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


# These imports are needed if trying to render a panel of images via Matplotlib
#import imutils
#from imutils import contours
#from imutils import perspective
#from scipy.spatial import distance as dist
#import scipy.misc as sm
#import matplotlib as mpl
#mpl.use("TkAgg")
#from matplotlib import pyplot as plt

# These are required imports for basic code functionality
import numpy as np
import cv2
import argparse, traceback, os, sys, logging
custom_log = logging.getLogger().setLevel(logging.INFO)


# Collect commandline arguments
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
    parser.add_argument('-o',
                        '--out',
                        action='store',
                        default=None,
                        help='Output filename stem for image saving.')
    parser.add_argument('--kernel_size',
                        action='store',
                        type=int,
                        default=7,
                        metavar='N',
                        help='Kernel size passed to relevant functions (one kernel for all functions for now). [Def 7]')
    parser.add_argument('--gsigma',
                        action='store',
                        type=int,
                        default=2,
                        metavar='N',
                        help='Standard deviation of Gaussian Blur kernel. [Def 0]')
# Custom boundary definition to be implemented later
#    parser.add_argument('--thresh_bounds',
#                        action='store',
#                        nargs=2,
#                        metavar=('lower', 'upper'),
#                        default=[0,15],
#                        help='Lower and upper bound of pixel intensity for thresholding. [Def 0,15]')
    parser.add_argument('--ignore_intermediates',
                        action='store_true',
                        help='Don\'t output intermediate images from processing. [Def off]')
    parser.add_argument('--scaling',
                        action='store',
                        default=0.75,
                        help='OpenCV displays images fullsize, this scales them down to this fraction of screen size.')

except NameError:
    # This is a general error handle. Not very useful to debug, but catches malformed arguments etc
    print "An exception occured with argument parsing. Check your provided options."
    traceback.print_exc()
    sys.exit(1)


args = parser.parse_args()

# Welcome to the program start...
print('"PrecICE" Image analysis tool, version %s by %s <%s>' %(__version__, __author__, __author_email__))


### SET UP ###

# List of intermediate images to store for later
# (at present, the code collects all the intermediate images it creates
# to assist with debugging and assessing the quality of the analysis. This
# does mean the program runs slower and uses more memory than necessary for
# actual analysis.


intermediate_images = []
intermediate_imagenames = []

# Read image in to numerical array
image = cv2.imread(args.image)
if os.path.isfile(args.image) is False:
    print("ERROR: Could not find the file, check the filepath. Aborted.")
    sys.exit(1)

print('Analysing image: %s \n' % args.image)

logging.info(' (Step 1/) Loaded image: %s' %args.image)

# Get the file extension, and produce images in the same format
ext = os.path.splitext(args.image)[1]

# Store the unaltered input image
intermediate_images.append(image)
intermediate_imagenames.append('Original Image')
logging.info('Original image stored...')


### PREPROCESSING IMAGE ####

# Gaussian filtering seems to help slightly denoise the edges and improve the morphological
# filtering steps to come

logging.info(' (Step 2/) Gaussian Filtering image...')

# Kernel sizes and sigma values may be able to be played with for more optimal results.
blur = cv2.GaussianBlur(image, (args.kernel_size, args.kernel_size), args.gsigma)
h, w = image.shape[:2] # Get image size

# Store the blurred/Gaussian filtered image.
intermediate_images.append(blur)
intermediate_imagenames.append('Gaussian Filtered Image...')
logging.info('Gaussian Filtered image stored...')


# Apply morphological filtering to try and close more boundaries.
logging.info(' (Step 3/) Applying morphological gradient...')
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (args.kernel_size, args.kernel_size))
gradient = cv2.morphologyEx(blur, cv2.MORPH_GRADIENT, kernel)


### BINARISING GRADIENT ###

logging.info(' (Step 4/) Binarising the image...')
#lowerb = np.array([args.thresh_bounds[0] for i in xrange(2)])
#upperb = np.array([args.thresh_bounds[1] for i in xrange(2)])

# Bounds are currently arbitrarily chosen and hard-coded. This is another
# area where some optimisation may be possible. Ideally the bounds could be
# heuristically chosen from the input image (maybe some percentile of average
# gray intensity or similar).
lowerb = np.array([0, 0, 0])
upperb = np.array([15, 15, 15])
# Binarise the image
binary = cv2.inRange(gradient, lowerb, upperb)
binary_store = cv2.inRange(gradient, lowerb, upperb)
# Store the Binarised image.
intermediate_images.append(binary_store)
intermediate_imagenames.append('Binarised Image')
logging.info('Binarised image stored...')

### CORRECTING EDGES ###

# Now we need to exclude blobs on the edge of the image from subsequent
# analysis by watershedding, so loop around the perimeter of the image, and if
# the pixel is white (==255), it's a blob and we can't know it's actual dimensions
# as it will be extending outside the of image frame, so it's floodfilled black to make it background.

logging.info(' (Step 5/) Floodfilling perimeter targets...')
for row in xrange(h):
    if binary[row, 0] == 255:
        cv2.floodFill(binary, None, (0, row), 0)
    if binary[row, w-1] == 255:
        cv2.floodFill(binary, None, (w-1, row), 0)

for col in xrange(w):
    if binary[0, col] == 255:
        cv2.floodFill(binary, None, (col, 0), 0)
    if binary[h-1, col] == 255:
        cv2.floodFill(binary, None, (col, h-1), 0)

intermediate_images.append(binary)
intermediate_imagenames.append('Floodfilled Image')
logging.info('Floodfilled image stored...')


### FOREGROUNDING AND CREATING UNKNOWN MASK ###
logging.info(' (Step 6/) Running cleanup mask (identifying back/foreground and unknown space)...')
foreground = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
foreground = cv2.morphologyEx(foreground, cv2.MORPH_CLOSE, kernel)
intermediate_images.append(foreground)
intermediate_imagenames.append('Foreground Image Image')
logging.info('Foreground regions image stored...')

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))

background = cv2.dilate(foreground, kernel, iterations=3)
intermediate_images.append(background)
intermediate_imagenames.append('Background Regions Image')
logging.info('Background regions image stored...')

unknown = cv2.subtract(background, foreground)
intermediate_images.append(foreground)
intermediate_imagenames.append('Unknown Regions Image')
logging.info('Unknown regions image stored...')

### WATERSHEDDING IMAGE ###
logging.info(' (Step 7/) Watershedding the image...')
markers = cv2.connectedComponents(foreground)[1]
# Set background intensity to 1
markers += 1
# Set unknown region intensity to 0
markers[unknown == 255] = 0
markers = cv2.watershed(image, markers)

# Something to try later
# Maybe try watershedding a blurred image to smooth the blob boundaries
# median = cv2.medianBlur(img, 5)
# markers2 = cv2.watershed(blur, markers)

### VISUALISE WATERSHED ###
hue_markers = np.uint8(179*np.float32(markers)/np.max(markers))
blank_channel = 255*np.ones((h, w), dtype=np.uint8)
marker_image = cv2.merge([hue_markers, blank_channel, blank_channel])
marker_image = cv2.cvtColor(marker_image, cv2.COLOR_HSV2BGR)

intermediate_images.append(marker_image)
intermediate_imagenames.append('Marker Image')
logging.info('Marker image stored...')

### Create overlaid image ###
labelled_image = image.copy()
labelled_image[markers>1] = marker_image[markers>1]  # 1 is background color
labelled_image = cv2.addWeighted(image, 0.5, labelled_image, 0.5, 0)

intermediate_images.append(labelled_image)
intermediate_imagenames.append('Annotated Image')
logging.info('Annotated image stored...')

## Display/Save Images

if args.ignore_intermediates is False:
    import Tkinter
    from math import ceil
    root = Tkinter.Tk()  # Need screen dimensions to deal with large images else they don't fit on screen.
    s_width = root.winfo_screenwidth()
    s_height = root.winfo_screenheight()
    # User can specify how much they want to downscale the image themself.
    scaling = args.scaling
    print("Screen resolution: %s x %s" %(s_width,s_height))

    for name, imageobj in zip(intermediate_imagenames, intermediate_images): # iterate list of names and images
        print("Rendering: " + name)
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        # Resize images that are too large to display
        cv2.resizeWindow(name, int(ceil(s_width*scaling)), int(ceil(s_height*scaling)))
        cv2.imshow(name, imageobj)

        # Write output files of intermediate images (default names used but with _ subbed in for whitespace
        if args.out is None:
            name = name.replace(" ", "_") + ext
        else:
            name = "%s%s%s%s" %(args.out, "_", name.replace(" ", "_"), ext)

        cv2.imwrite(name, imageobj)

    cv2.waitKey(0)


# This is commented out until I can be bothered to work out an easy way to switch BGR to RGB for the relevant images.
# The images will display in a panel based window, rather than in separate ones but the colours will be off.

# Panel images:

#OpenCV's BGR encoding needs to be converted to RGB for matplotlib to display correctly:
#mpl_converted_intermediates = [sm.toimage(image) for image in intermediate_images]
# from math import ceil
# import matplotlib as mpl
# mpl.use("TkAgg")
# from matplotlib import pyplot as plt
# for i in xrange(len(zip(intermediate_imagenames, intermediate_images))):
#     rows = ceil(float(len(intermediate_images))/2)
#     plt.subplot(rows,2,i+1)
#     plt.imshow(intermediate_images[i])
#     plt.title(intermediate_imagenames[i])
#     plt.xticks([]),plt.yticks([])
# plt.show()


#TODO
# Write the code that will gather 'blob' contours
# Write code to calculate a bounding box and measure the crystals
# Enumerate the number of crystals and output some simple statistics (mean size/number of crystals detected etc)
# Write code which will back-calculate pixel dimensions to actual size measurements.

# Eventually refactor the code to be tidier, more function based, faster/more efficient.
