# 2228 CougarTech Pi Vision Project

## Overview
This project contains the wpilibip vision code for the 2021 FRC season

## Important Components

- `./src/` Java source code for the vision application
- `./neuralNetwork/` A self-contained Jupyter notebook for training and saving the convolutional neural network (CNN) used in the galactic search challange
- `./opencv-4.5.1.tgz` A full distribution of OpenCV 4.5.1 compiled to execute on the Raspberry Pi ARM platform

## Building and Deploying
The Raspberry Pi wpilibpi vision application web page must be set to "custom" instead of "uploaded Java jar" or any other option. The actual copying of the vision code, and the required libraries and neural network model will be performed by gradle.

Note: The stock wpilibpi image only supports OpenCV 3.4.7, so a custom buld of 4.5.1 was made, and will be deployed to the Pi automatically

To deploy all required components to the Pi, and automatically restart the vision service, run the following commands from VSCode:
- `./gradlew build` -- Build the actual Java application
- `./gradlew deploy` -- Deploy everything to the Pi

This will do the following:
- Check for the presence of `/home/pi/opencv-4.5.1`. If this does not exist, it will copy over the opencv-4.5.1 distribution and extract it to disk
- Check for the presence of the CNN .pb file, if not found, it will copy it to `/home/pi/`
- Copy the latest compiled .jar file, and the runCamera command to `/home/pi/`
- Restart the camera service, causing the updated code to start running
