# DIY Video Filters

This repo is an example and demonstration of how one can create video filters on top of video live feed.
It also aims to show how one could combine Python with C++ in Windows with an ability of mixed-mode debugging in Visual Studio.
Mixed-mode debuging allows you to debug Python and C++ code at the same time.

The basic idea of video filters is simple:

* create a virtual video camera in the system
* start capturing frames from a real camera
* do image processing on every frame
* write processed frame to the virtual camera.

Virtual camera then can be used in video conferencing apps like Skype, Teams, Zoom, e.t.c.

Tested in Python 3.9 under Windows 11.

## Requirements

You would need to install the following:
1. Visual Studio 2022 with Python and C++ workloads.
2. Install a virtual web camera: https://github.com/schellingb/UnityCapture

## How to run

1. Install Python 3.9 either independently of as part of Visual Studio 2022.
2. Create a virtual environment and activate it. Create virtual environment with symlinks enabled by running `python -m venv --symlinks venv`.
3. Activate virtual environment by running `venv\Scripts\activate.bat`.
4. Install dependencies by running `pip install -r requirements.txt`.
5. **Optional**. One video filter is based on dlib library. It is distributed in source and hence not part of requirements.txt.
  You may use pre-compiled version from the internet. Here is one: https://github.com/sachadee/Dlib . You will also need shape prediction model file (shape_predictor_68_face_landmarks.dat) from [here](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2). Extract it to the same directory as `app.py` file.
5. In order to build C++ library, we need to point compiler to header and library files.
  I tried my best to make it automatic via the script `start_vs.cmd`.
  Use this script to start Visual Studio 2022. Adjust if necessary.
6. Set `Python Environments` to your environment for `diy_pyfilters` project.
7. Run or debug `diy_pyfilters` project.