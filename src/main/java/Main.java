
/*----------------------------------------------------------------------------*/
/* Copyright (c) 2018 FIRST. All Rights Reserved.                             */
/* Open Source Software - may be modified and shared by FRC teams. The code   */
/* must be accompanied by the FIRST BSD license file in the root directory of */
/* the project.                                                               */
/*----------------------------------------------------------------------------*/

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;

import org.opencv.core.KeyPoint;
import org.opencv.core.Mat;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.RotatedRect;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;

import edu.wpi.cscore.MjpegServer;
import edu.wpi.cscore.UsbCamera;
import edu.wpi.cscore.VideoCamera;
import edu.wpi.cscore.VideoSource;
import edu.wpi.cscore.CvSource;
import edu.wpi.cscore.CvSink;
import edu.wpi.cscore.VideoMode.PixelFormat;
import edu.wpi.first.cameraserver.CameraServer;
import edu.wpi.first.networktables.NetworkTableInstance;
import edu.wpi.first.networktables.NetworkTable;
import edu.wpi.first.networktables.NetworkTableEntry;
import edu.wpi.first.vision.VisionThread;
import edu.wpi.first.wpilibj.shuffleboard.Shuffleboard;
import edu.wpi.first.wpilibj.shuffleboard.ShuffleboardTab;
import vision.GripPipeline;



/*
   JSON format:
   {
       "team": <team number>,
       "ntmode": <"client" or "server", "client" if unspecified>
       "cameras": [
           {
               "name": <camera name>
               "path": <path, e.g. "/dev/video0">
               "pixel format": <"MJPEG", "YUYV", etc>   // optional
               "width": <video mode width>              // optional
               "height": <video mode height>            // optional
               "fps": <video mode fps>                  // optional
               "brightness": <percentage brightness>    // optional
               "white balance": <"auto", "hold", value> // optional
               "exposure": <"auto", "hold", value>      // optional
               "properties": [                          // optional
                   {
                       "name": <property name>
                       "value": <property value>
                   }
               ],
               "stream": {                              // optional
                   "properties": [
                       {
                           "name": <stream property name>
                           "value": <stream property value>
                       }
                   ]
               }
           }
       ]
   }
 */

// **************************************************************************
// * 
// * Main Class
// *
// **************************************************************************
public final class Main {
  public static final int MJPEG_OPENCV_SERVER_PORT = 1183;
  public static final double IMAGE_WIDTH_PIXELS = 640.0;
  public static final double IMAGE_HEIGHT_PIXELS = 480.0;
  public static final int DEFAULT_FRAME_RATE = 30;
  public static final double HALF_IMAGE_WIDTH_IN_PIXELS = IMAGE_WIDTH_PIXELS / 2.0;

  public static final int TARGETING_STATE_SEARCHING = 0;
  public static final int TARGETING_STATE_ACQUIRING = 1;
  public static final int TARGETING_STATE_LOCKED = 2;

  public static final double TARGET_HEIGHT_INCHES = 5.5;
  public static final double TARGET_WIDTH_INCHES = 2.0;

  public static final double TARGET_ASPECT_RATIO_TOLERANCE = .20;
  public static final double TARGET_ASPECT_RATIO_FOR_LOW_ANGLE = TARGET_HEIGHT_INCHES / TARGET_WIDTH_INCHES;
  public static final double TARGET_ASPECT_RATIO_MIN_THRESHOLD_FOR_LOW_ANGLE = TARGET_ASPECT_RATIO_FOR_LOW_ANGLE
      - (TARGET_ASPECT_RATIO_FOR_LOW_ANGLE * TARGET_ASPECT_RATIO_TOLERANCE);
  public static final double TARGET_ASPECT_RATIO_MAX_THRESHOLD_FOR_LOW_ANGLE = TARGET_ASPECT_RATIO_FOR_LOW_ANGLE
      + (TARGET_ASPECT_RATIO_FOR_LOW_ANGLE * TARGET_ASPECT_RATIO_TOLERANCE);

  // For the high angle (i.e., the ~-75 degree vision strip), we need to swap the
  // height and width in the aspect ratio calculation. TODO - not sure why this is
  // necessary
  public static final double TARGET_ASPECT_RATIO_FOR_HIGH_ANGLE = TARGET_WIDTH_INCHES / TARGET_HEIGHT_INCHES;
  public static final double TARGET_ASPECT_RATIO_MIN_THRESHOLD_FOR_HIGH_ANGLE = TARGET_ASPECT_RATIO_FOR_HIGH_ANGLE
      - (TARGET_ASPECT_RATIO_FOR_HIGH_ANGLE * TARGET_ASPECT_RATIO_TOLERANCE);
  public static final double TARGET_ASPECT_RATIO_MAX_THRESHOLD_FOR_HIGH_ANGLE = TARGET_ASPECT_RATIO_FOR_HIGH_ANGLE
      + (TARGET_ASPECT_RATIO_FOR_HIGH_ANGLE * TARGET_ASPECT_RATIO_TOLERANCE);

  public static final double TARGET_LOW_ANGLE = -15.0;
  public static final double TARGET_HIGH_ANGLE = -75.0;
  public static final double TARGET_ANGLE_TOLERANCE_IN_DEGREES = 10;
  public static final double TARGET_LOW_ANGLE_MIN_THRESHOLD = TARGET_LOW_ANGLE - TARGET_ANGLE_TOLERANCE_IN_DEGREES;
  public static final double TARGET_LOW_ANGLE_MAX_THRESHOLD = TARGET_LOW_ANGLE + TARGET_ANGLE_TOLERANCE_IN_DEGREES;
  public static final double TARGET_HIGH_ANGLE_MIN_THRESHOLD = TARGET_HIGH_ANGLE - TARGET_ANGLE_TOLERANCE_IN_DEGREES;
  public static final double TARGET_HIGH_ANGLE_MAX_THRESHOLD = TARGET_HIGH_ANGLE + TARGET_ANGLE_TOLERANCE_IN_DEGREES;

  public static final int MIN_HASH_MAP_DISTANCE = 18;
  public static final int MAX_HASH_MAP_DISTANCE = 48;

  public static final double MINIMUM_HORIZONTAL_OFFSET_REQ_IN_PIXELS = 200.0;

  public static final double GS_X_OFFSET = 0;
  public static final double GS_SIZE_OFFSET = 0;

  public static final int PT_CAMERA_EXPOSURE = 8;
  public static final int GS_CAMERA_EXPOSURE = -1; //-1 means auto

  // When we were empirically collecting data for the distance calculation hash
  // map,
  // we observed that the actual measured distance between the front of the camera
  // and the target was roughly 2 inches less than what the distance calculation
  // in
  // the code was telling us.
  public static final double DISTANCE_CORRECTION_OFFSET = 2.0;

  // !!! Very small changes in this constant dramatically affects distance calc
  // accuracy !!!
  public static final double CAMERA_FOV_ANGLE = 60.010; // FOV Angle determined empirically
  public static final double CAMERA_FOV_ANGLE_CALC = Math.tan(CAMERA_FOV_ANGLE);

  private static String configFile = "/boot/frc.json";

  public static class CameraConfig {
    public String name;
    public String path;
    public JsonObject config;
    public JsonElement streamConfig;
  }

  public static int team;
  public static boolean server;
  public static List<CameraConfig> cameraConfigs = new ArrayList<>();
  public static int targetingState = TARGETING_STATE_SEARCHING;

  // This will be the list of targets that we'll use to determine whether or not
  // we're locked on the two angle vision tape strips.
  public static List<Rect> targets = new ArrayList<>();
  public static List<RotatedRect> targetRects = new ArrayList<>();

  static List<VideoCamera> cameras = new ArrayList<>();

  private Main() {
  }

  // **************************************************************************
  // *
  // * Report parse error.
  // *
  // **************************************************************************
  public static void parseError(String str) {
    System.err.println("config error in '" + configFile + "': " + str);
  }

  // **************************************************************************
  // *
  // * Read single camera configuration
  // *
  // **************************************************************************
  public static boolean readCameraConfig(JsonObject config) {
    CameraConfig cam = new CameraConfig();

    // name
    JsonElement nameElement = config.get("name");
    if (nameElement == null) {
      parseError("could not read camera name");
      return false;
    }
    cam.name = nameElement.getAsString();

    // path
    JsonElement pathElement = config.get("path");
    if (pathElement == null) {
      parseError("camera '" + cam.name + "': could not read path");
      return false;
    }
    cam.path = pathElement.getAsString();

    // stream properties
    cam.streamConfig = config.get("stream");

    cam.config = config;

    cameraConfigs.add(cam);
    return true;
  }

  // **************************************************************************
  // *
  // * Read configuration file.
  // *
  // **************************************************************************
  public static boolean readConfig() {
    // parse file
    JsonElement top;

    try {
      top = new JsonParser().parse(Files.newBufferedReader(Paths.get(configFile)));
    } catch (IOException ex) {
      System.err.println("could not open '" + configFile + "': " + ex);
      return false;
    }

    // top level must be an object
    if (!top.isJsonObject()) {
      parseError("must be JSON object");
      return false;
    }

    JsonObject obj = top.getAsJsonObject();

    // team number
    JsonElement teamElement = obj.get("team");

    if (teamElement == null) {
      parseError("could not read team number");
      return false;
    }

    team = teamElement.getAsInt();

    // ntmode (optional)
    if (obj.has("ntmode")) {

      String str = obj.get("ntmode").getAsString();

      if ("client".equalsIgnoreCase(str)) {
        server = false;
      } else if ("server".equalsIgnoreCase(str)) {
        server = true;
      } else {
        parseError("could not understand ntmode value '" + str + "'");
      }
    }

    JsonElement camerasElement = obj.get("cameras");

    if (camerasElement == null) {
      parseError("could not read cameras");
      return false;
    }

    JsonArray cameras = camerasElement.getAsJsonArray();

    for (JsonElement camera : cameras) {
      if (!readCameraConfig(camera.getAsJsonObject())) {
        return false;
      }
    }

    return true;
  }

  // **************************************************************************
  // *
  // * Start running the camera
  // *
  // **************************************************************************
  public static VideoCamera startCamera(CameraConfig config) {
    System.out.println("Starting camera '" + config.name + "' on " + config.path);
    UsbCamera camera = new UsbCamera(config.name, config.path);
    MjpegServer mjpegServer = CameraServer.getInstance().startAutomaticCapture(camera);

    Gson gson = new GsonBuilder().create();

    camera.setConfigJson(gson.toJson(config.config));
    camera.setConnectionStrategy(VideoSource.ConnectionStrategy.kKeepOpen);

    if (config.streamConfig != null) {
      mjpegServer.setConfigJson(gson.toJson(config.streamConfig));
    }

    return camera;
  }

  public static NetworkTableEntry hubTopLeftX;
  public static NetworkTableEntry hubTopLeftY;
  public static NetworkTableEntry hubBottomRightX;
  public static NetworkTableEntry hubBottomRightY;
  public static NetworkTableEntry shapesMinX;
  public static NetworkTableEntry shapesMinY;
  public static NetworkTableEntry shapesMaxX;
  public static NetworkTableEntry shapesMaxY;
  public static NetworkTableEntry shapesAreas;
  public static NetworkTableEntry boundsX;
  public static NetworkTableEntry boundsY;
  public static NetworkTableEntry center;
  public static NetworkTableEntry distanceFt;
  public static NetworkTableEntry quadraticABC;
  public static NetworkTableEntry width;
  public static NetworkTableEntry deviationFromCenter;
  public static NetworkTableEntry activeCamera;
  public static NetworkTableEntry hslThresholdHueNT;
  public static NetworkTableEntry hslThresholdSaturationNT;
  public static NetworkTableEntry hslThresholdLuminanceNT;
  public static NetworkTableEntry shooterCameraExposureNT;

  public static VideoSource shooterCamera;
  public static CvSource outputStream;

  private static double[] hslThresholdHue = {44, 91};
	private	static double[] hslThresholdSaturation = {204, 255.0};
	private	static double[] hslThresholdLuminance = {28, 193};

  private static int shooterCameraExposure;

  public static void main(String... args) {
    if (args.length > 0) {
      configFile = args[0];
    }

    // Read configuration
    if (!readConfig()) {
      return;
    }

    // Start NetworkTables
    NetworkTableInstance ntinst = NetworkTableInstance.getDefault();

    if (server) {
      System.out.println("Setting up NetworkTables server");
      ntinst.startServer();
    } else {
      System.out.println("Setting up NetworkTables client for team " + team);
      ntinst.startClientTeam(team);
    }

    
    // ShuffleboardTab tab = Shuffleboard.getTab("PowerTower");
    // xOffset = tab.add("PT Offset", 0).getEntry();

    NetworkTable hubsettingstable =  ntinst.getTable("Hub Settings");

    hslThresholdHueNT = hubsettingstable.getEntry("Hue");
    hslThresholdSaturationNT = hubsettingstable.getEntry("Saturation");
    hslThresholdLuminanceNT = hubsettingstable.getEntry("Luminance");

    if(hslThresholdHue.length == 0){
      hslThresholdHue = new double[] {44, 91};
      System.out.println("hue default");
    }
    if(hslThresholdSaturation.length == 0){
      hslThresholdSaturation = new double[] {204, 255.0};
      System.out.println("Saturation default");
    }
    if(hslThresholdLuminance.length == 0){
      hslThresholdLuminance = new double[] {28, 193};
      System.out.println("Luminance default");
    }
    


    hslThresholdHueNT.getDoubleArray(hslThresholdHue);
    hslThresholdSaturationNT.getDoubleArray(hslThresholdSaturation);
    hslThresholdLuminanceNT.getDoubleArray(hslThresholdLuminance);

    NetworkTable table = ntinst.getTable("Hub");

    hubTopLeftX = table.getEntry("hubTopLeftX");
    hubTopLeftY = table.getEntry("hubTopLeftY");
    hubBottomRightX = table.getEntry("hubBottomRightX");
    hubBottomRightY = table.getEntry("hubBottomRightY");

    shapesMinX = table.getEntry("shapesMinX");
    shapesMinY = table.getEntry("shapesMinY");
    shapesMaxX = table.getEntry("shapesMaxX");
    shapesMaxY = table.getEntry("shapesMaxY");
    shapesAreas = table.getEntry("shapesAreas");

    boundsX = table.getEntry("totalWidth");
    boundsY = table.getEntry("totalHeight");
    center = table.getEntry("center");
    distanceFt = table.getEntry("distanceFeet");
    quadraticABC = table.getEntry("quadraticABC");
    width = table.getEntry("width");
    deviationFromCenter = table.getEntry("deviationFromCenter");
    activeCamera = table.getEntry("currentCamera");
    shooterCameraExposureNT = hubsettingstable.getEntry("shooterCamerExposure");

    shooterCameraExposure = shooterCameraExposureNT.getNumber(100).intValue();
  
    // Start cameras
    for (CameraConfig cameraConfig : cameraConfigs) {      
      cameras.add(startCamera(cameraConfig));
    }

    CvSink cvSink = new CvSink("openCV Camera");

    // Mat openCVOverlay = new Mat();

    // Start image processing on camera 0 if present
    if (cameras.size() >= 1) {

      // For OpenCV processing, you need a "source" which will be our camera and
      // a "sink" or "destination" which will be an ouputStream that is fed into
      // an MJPEG Server.
      // TODO - this will always get the first camera detected and that may be the
      // back camera which is no bueno
      for(VideoCamera camera : cameras){
        if(camera.getName().equals("Shooter")){
          shooterCamera = camera;
          camera.setExposureManual(shooterCameraExposure);
        }
      }
      //shooterCamera = cameras.get(0);

      cvSink.setSource(shooterCamera);
      outputStream = new CvSource("2228_OpenCV", PixelFormat.kMJPEG, (int) IMAGE_WIDTH_PIXELS,
          (int) IMAGE_HEIGHT_PIXELS, DEFAULT_FRAME_RATE);

      // This is MJPEG server used to create an overlaid image of what the OpenCV
      // processing is
      // coming up with on top of the live streamed image from the robot's front
      // camera.
      MjpegServer mjpegServer2 = new MjpegServer("serve_openCV", MJPEG_OPENCV_SERVER_PORT);
      mjpegServer2.setSource(outputStream);

    } else {
      System.out.println("No cameras found");
    }
    // **************************************************************************
    // *
    // * Main "Forever" Loop
    // *
    // **************************************************************************
    String previousSelected = null;
    Thread currentVisionThread = null;
    String visionMode = "Hub";
    for (;;) {

      try {
        Thread.sleep(300);
      } catch (InterruptedException ex) {
        return;
      }

      if (!visionMode.equals(previousSelected)) {

            //setCameraExposure(PT_CAMERA_EXPOSURE);
            currentVisionThread = makePowerTower();
            currentVisionThread.start();
            System.out.println("Starting Power Tower");

        previousSelected = visionMode;
      }
    }
  }

  static ArrayList<Double> distances = new ArrayList<Double>();
  private static VisionThread makePowerTower() {
    return new VisionThread(shooterCamera, new GripPipeline(hslThresholdHue, hslThresholdSaturation, hslThresholdLuminance), pipeline -> {
      // This grabs a snapshot of the live image currently being streamed
      // cvSink.grabFrame(openCVOverlay);
      ArrayList<MatOfPoint> filterContoursOutput = pipeline.filterContoursOutput();

      double minx = 99999;
      double miny = 99999;
      double maxx = 0;
      double maxy = 0;
      // double width;
      // double height;
      double[] minX = new double[filterContoursOutput.size()];
      double[] minY = new double[filterContoursOutput.size()];
      double[] maxX = new double[filterContoursOutput.size()];
      double[] maxY = new double[filterContoursOutput.size()];
      double[] areas = new double[filterContoursOutput.size()];
      double[] centerX = new double[filterContoursOutput.size()];
      double[] centerY = new double[filterContoursOutput.size()];
      int count = 0;
      
      
      for (MatOfPoint points : filterContoursOutput) {
        areas[count] = Imgproc.contourArea(points);
        double current_min_x = minx;
        double current_min_y = miny;
        double current_max_x = maxx;
        double current_max_y = maxy; 
        double shape_min_x = 99999;
        double shape_min_y = 99999;
        double shape_max_x = 0;
        double shape_max_y = 0;
        boolean isValid = true;
        for(Point point : points.toArray()) {

          if (point.x < minx) {
            minx = point.x;
          }
          if (point.x > maxx) {
            maxx = point.x;
          }
          if (point.y < miny) {
            miny = point.y;
          }
          if (point.y > maxy) {
            maxy = point.y;
          }

          if (point.x < shape_min_x) {
            shape_min_x = point.x;
          }
          if (point.x > shape_max_x) {
            shape_max_x = point.x;
          }
          if (point.y < shape_min_y) {
            shape_min_y = point.y;
          }
          if (point.y > shape_max_y) {
            shape_max_y = point.y;
          }
        }
        minX[count] = shape_min_x;
        minY[count] = shape_min_y;
        maxX[count] = shape_max_x;
        maxY[count] = shape_max_y;
        centerX[count] = shape_max_x - shape_min_x;
        centerY[count] = shape_max_y - shape_min_y;
        count++;

      }
      
      // width = maxx - minx;
      // height = maxy - miny;

      // System.out.println("X: " + minx);
      double yAve = 0;
      double xAve = 0;
      for(int i = 0; i < centerX.length; i++){
        xAve += centerX[i];
        yAve += centerY[i];
      }
      xAve /= centerX.length;
      yAve /= centerY.length;
      double dist = -1.2033 * yAve + 23.176;
      if(distances.size() < 10){
        distances.add(dist);
      }
      else{
        distances.remove(0);
        distances.add(dist);
      }
      dist = 0;
      for(double distance : distances){
        dist += distance;
      }
      dist /= distances.size();

      shapesMinX.setDoubleArray(minX);
      shapesMinY.setDoubleArray(minY);
      shapesMaxX.setDoubleArray(maxX);
      shapesMaxY.setDoubleArray(maxY);
      shapesAreas.setDoubleArray(areas);
      boundsX.setDouble(xAve);
      boundsY.setDouble(yAve);
      
      width.setDouble(dist);

      hubTopLeftX.setNumber(minx);
      hubTopLeftY.setNumber(miny);
      hubBottomRightX.setNumber(maxx);
      hubBottomRightY.setNumber(maxy);      
      deviationFromCenter.setNumber(xAve - IMAGE_WIDTH_PIXELS / 2);

      //boundsX.setNumber(maxx - minx); 
      //boundsY.setNumber(maxy - miny);
      //center.setNumber(minx + (maxx - minx) / 2);
      //double width = maxx - minx;
      //distanceFt.setDouble(-.1757 * width + 34.209);
      //distanceFt.setDouble(width);
      // double x1 = minX[0];
      // double y1 = minY[0];
      // double x2 = minX[1];
      // double y2 = minY[1];
      // double x3 = minX[2];
      // double y3 = minY[2];

      //double b = (y1 * (x1 * x1 - x3 * x3) - y2 * (x1 * x1 - x3 * x3) - y1 * (x1 * x1 - x2 * x2) - y3 * (x1 * x1 - x2 * x2))
      /// ((x1 - x2) * (x1 * x1 - x3 * x3) - (x1 - x3) * (x1 * x1 - x2 * x2));
      //double a = (y2 - y3 - b * (x2 - x3)) / (x2 * x2 - x3 * x3);
      //double c = y2 - a * x2 * x2 - b * x2;
      //double[] vals = {a, b, c};
      //quadraticABC.setDoubleArray(vals);

      // double a = x1 * x1;
      // double b = x1;
      // double c = 1;
      // double d = y1;
      // double e = x2 * x2;
      // double f = x2;
      // double g = 1;
      // double h = y2;
      // double i = x3 * x3;
      // double j = x3;
      // double k = 1;
      // double l = y3;

      // double delta = (a * f * k) + (b * g * i) + (c * e * j) - (c * f * i) - (a * g * j) - (b * e * k);
      // double aNum = (d * f * k) + (b * g * l) + (c * h * j) - (c * f * l) - (d * g * j) - (b * h * k);
      // double bNum = (a * h * k) + (d * g * i) + (c * e * l) - (c * h * i) - (a * g * l) - (d * e * k);
      // double cNum = (a * f * l) + (b * h * i) + (d * e * j) - (d * f * i) - (a * h * j) - (b * e * l);

      // double[] vals = {aNum / delta, bNum/delta, cNum/delta};
      // quadraticABC.setDoubleArray(vals);
    });
  }


  // public static void setCameraExposure(int value){
  //   for(VideoCamera camera : cameras){
  //     if (value >= 0){
  //       camera.setExposureManual(value);
  //     }
  //     else{
  //       camera.setExposureAuto();
  //     }
  //   }
  // }
}

