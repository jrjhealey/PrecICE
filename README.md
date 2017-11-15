# PrecICE

This is a small peice of software that attempts to use computer vision to analyse images, look for features and extract
measurements of those features in an automatic fashion. It has been developed specifically for use with so-called
'Splat Test' images that are created for the study of Ice Recrystallisation Inhibition experiments conducted within the
Gibson laboratory at the University of Warwick.

# Installation
This program relies (unfortunately) on many large 3rd party modules, though ones that are in common use. Cheif among these
are the `python` implementation of `OpenCV` (`opencv-python` in `pip`), and the scientific computing libraries `scipy` and 
`numpy` with an emphasis on their image analysis elements.

#### Basic Installation
  1. Clone the repository somewhere you're happy to have the folder:
    
    $ cd my/directory/of/choice
    $ git clone https://github.com/jrjhealey/PrecICE

  2. Install the dependencies:
  
    $ cd path/to/PrecICE
    $ python -m pip install -r requirements.txt
  
I would advise downloading an `Anaconda` or `Miniconda` distribution so that you don't encounter problems with installing
modules in to your system's python install. Follow the instructions at https://conda.io/miniconda.html to install a Conda
scientific computing environment. Allow the installer to edit your `.bashrc` file if it asks.

You will need to ensure the result of the command:

    which python
    
shows you something like:

    /home/username/miniconda2/python2.7


# Basic Usage

    $ python PrecICE.py [-h|--help] [options] IMAGEFILE

# Advanced Usage
        
    usage: $ python PrecICE.py [-h|--help] [options] IMAGEFILE
    
    
    Computer vision program for feature size determination. Utilises computer
    vision and image analysis to return approximated feature dimensions.
    
    positional arguments:
      IMAGEFILE             The image file to be analysed.
    
    optional arguments:
      -h, --help            show this help message and exit
      -o OUT, --out OUT     Output filename stem for image saving.
      --kernel_size N       Kernel size passed to relevant functions (one kernel
                            for all functions for now). [Def 7]
      --gsigma N            Standard deviation of Gaussian Blur kernel. [Def 0]
      --ignore_intermediates
                            Don't output intermediate images from processing. [Def
                            off]
      --scaling SCALING     OpenCV displays images fullsize, this scales them down
                            to this fraction of screen size.


# Disclaimers
This software is not well tested, and the many dependencies may make it difficult to install for the uninitiated. I also
do not claim credit for all of the code, some of which is borrowed heavily from resources on the web including, but not limited 
to:

 * http://www.pyimagesearch.com/2016/03/28/measuring-size-of-objects-in-an-image-with-opencv/
 * http://opencvpython.blogspot.co.uk/2012/05/skeletonization-using-opencv-python.html    

and this interesting thread at StackOverflow:

 * https://stackoverflow.com/questions/38598690/how-to-find-the-diameter-of-objects-using-image-processing-in-python
