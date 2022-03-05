
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
import java.util.List;
import java.util.function.Consumer;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;

import org.opencv.core.Core;
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
import edu.wpi.first.networktables.ConnectionNotification;
import edu.wpi.first.networktables.EntryNotification;
import edu.wpi.first.networktables.NetworkTable;
import edu.wpi.first.networktables.NetworkTableEntry;
import edu.wpi.first.vision.VisionThread;
import vision.GripPipeline;

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

  // This will be the list of targets that we'll use to determine whether or not
  // we're locked on the two angle vision tape strips.
  public static List<Rect> targets = new ArrayList<>();
  public static List<RotatedRect> targetRects = new ArrayList<>();

  static List<VideoCamera> cameras = new ArrayList<>();

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
  public static NetworkTableEntry averageHeight;
  public static NetworkTableEntry deviationFromCenter;
  public static NetworkTableEntry activeCameraNT;
  public static NetworkTableEntry hslThresholdHueNT;
  public static NetworkTableEntry hslThresholdSaturationNT;
  public static NetworkTableEntry hslThresholdLuminanceNT;
  public static NetworkTableEntry shooterCameraExposureNT;

  static MjpegServer mjpegServer = null;
  // static VideoCamera acquirerCamera = null;
  static VideoCamera shooterCamera = null;
  static int shooterCameraExposure;

  private static double[] hslThresholdHue = {44, 91};
	private	static double[] hslThresholdSaturation = {204, 255.0};
	private	static double[] hslThresholdLuminance = {28, 193};
  private static boolean ntReady = false;

  public static CvSource cvOutputStream;
  public static Scalar greenColor;
  public static Scalar redColor;

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

    if (mjpegServer == null) {
      mjpegServer = CameraServer.getInstance().startAutomaticCapture(camera);
      mjpegServer.setCompression(30);
    }

    Gson gson = new GsonBuilder().create();

    camera.setConfigJson(gson.toJson(config.config));
    camera.setConnectionStrategy(VideoSource.ConnectionStrategy.kKeepOpen);
    return camera;
  }

  private static void initCamera() {
    System.out.println("initCamera()");
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
      for(VideoCamera camera : cameras){
        if(camera.getName().equals("Shooter")){
          shooterCamera = camera;
          setShooterCameraExposure(shooterCameraExposure);
        }
        // else if (camera.getName().equals("Acquirer")) {
        //   acquirerCamera = camera;
        // }
      }

      cvSink.setSource(shooterCamera);
      cvOutputStream = new CvSource("2228_OpenCV", PixelFormat.kMJPEG, (int) IMAGE_WIDTH_PIXELS,
          (int) IMAGE_HEIGHT_PIXELS, DEFAULT_FRAME_RATE);

      // This is MJPEG server used to create an overlaid image of what the OpenCV
      // processing is
      // coming up with on top of the live streamed image from the robot's front
      // camera.
      // MjpegServer mjpegServer2 = new MjpegServer("serve_openCV", MJPEG_OPENCV_SERVER_PORT);
      // mjpegServer2.setCompression(40);
      // mjpegServer2.setSource(outputStream);

    } else {
      System.out.println("No cameras found");
    }
  }
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

    NetworkTable hubsettingstable =  ntinst.getTable("Hub Settings");

    hslThresholdHueNT = hubsettingstable.getEntry("Hue");
    hslThresholdSaturationNT = hubsettingstable.getEntry("Saturation");
    hslThresholdLuminanceNT = hubsettingstable.getEntry("Luminance");
    shooterCameraExposureNT = hubsettingstable.getEntry("Exposure");

    greenColor = new Scalar(0.0, 255.0, 0.0);
    redColor = new Scalar(0.0, 0.0, 255.0);

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
    activeCameraNT = table.getEntry("currentCamera");
    averageHeight = table.getEntry("averageHeight");

    ntinst.addConnectionListener(new Consumer<ConnectionNotification>() {

      @Override
      public void accept(ConnectionNotification event) {
        if (event.connected) {
          System.out.println("NT Connected!!!");
          if(!shooterCameraExposureNT.exists()){
            shooterCameraExposure = 20;
            System.out.println("exposure default");
            shooterCameraExposureNT.setNumber(shooterCameraExposure);
          }
          else{
            shooterCameraExposure = shooterCameraExposureNT.getNumber(-1).intValue();
            System.out.println("read from NT: " + shooterCameraExposure);
          }
          shooterCameraExposureNT.addListener(new Consumer<EntryNotification>() {

            @Override
            public void accept(EntryNotification event) {
              System.out.println("shooterCameraExposureNT on Change -- Name: " + event.name + ", Value: " + event.value.getDouble());
              setShooterCameraExposure((int)event.value.getDouble());
            }
          }, 0xFFFF);

          if(!hslThresholdHueNT.exists()){
            hslThresholdHue = new double[] {44, 91};
            System.out.println("hue default");
            hslThresholdHueNT.setDoubleArray(hslThresholdHue);
          }
          else{
            hslThresholdHue = hslThresholdHueNT.getDoubleArray(new double[] {44, 91});
          }

          if(!hslThresholdSaturationNT.exists()){
            hslThresholdSaturation = new double[] {204, 255.0};
            System.out.println("Saturation default");
            hslThresholdSaturationNT.setDoubleArray(hslThresholdSaturation);
          }
          else{
            hslThresholdSaturationNT.getDoubleArray(hslThresholdSaturation);
          }

          if(!hslThresholdLuminanceNT.exists()){
            hslThresholdLuminance = new double[] {28, 193};
            System.out.println("Luminance default");
            hslThresholdLuminanceNT.setDoubleArray(hslThresholdLuminance);
          }
          else{
            hslThresholdLuminanceNT.getDoubleArray(hslThresholdLuminance);
          }

          hslThresholdHueNT.getDoubleArray(hslThresholdHue);
          hslThresholdSaturationNT.getDoubleArray(hslThresholdSaturation);
          hslThresholdLuminanceNT.getDoubleArray(hslThresholdLuminance);

          // activeCameraNT.addListener(new Consumer<EntryNotification>() {
          //   @Override
          //   public void accept(EntryNotification event) {
          //     String activeCamera = activeCameraNT.getString("");
          //     System.out.println("Active Camera Switched to " + activeCamera);
          //     if (mjpegServer != null) {
          //       if (activeCamera.equals("Ball")) {
          //         mjpegServer.setSource(acquirerCamera);
          //       } else if (activeCamera.equals("Target")) {
          //         mjpegServer.setSource(cvOutputStream);
          //       } else if (activeCamera.equals("Settings")){
          //         mjpegServer.setSource(shooterCamera);
          //       }
          //     }
          //   }
          // }, 0xfff);

          ntReady = true;
        }
      }
    }, true);

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

      if (ntReady && !visionMode.equals(previousSelected)) {
        initCamera();

        mjpegServer.setSource(cvOutputStream);
        //setCameraExposure(PT_CAMERA_EXPOSURE);
        currentVisionThread = makeVisionThread();
        currentVisionThread.start();
        System.out.println("Starting Vision Thread");
        previousSelected = visionMode;
      }
    }
  }

  static ArrayList<Double> distances = new ArrayList<Double>();
  private static VisionThread makeVisionThread() {
    return new VisionThread(shooterCamera, new GripPipeline(hslThresholdHue, hslThresholdSaturation, hslThresholdLuminance), pipeline -> {
      // This grabs a snapshot of the live image currently being streamed
      //cvSink.grabFrame(openCVOverlay);
      Mat openCVOverlay = pipeline.cvFlipOutput();
      
      double xOff = deviationFromCenter.getDouble(0.0);

      Core.rotate(openCVOverlay, openCVOverlay, Core.ROTATE_90_CLOCKWISE);

      Imgproc.line(openCVOverlay, new Point((IMAGE_WIDTH_PIXELS / 2), IMAGE_HEIGHT_PIXELS),
          new Point((IMAGE_WIDTH_PIXELS / 2), 0), greenColor, 3, 4);
      // double greenX = (IMAGE_HEIGHT_PIXELS / 2);
      
      Imgproc.line(openCVOverlay, new Point((xOff + IMAGE_WIDTH_PIXELS / 2), IMAGE_HEIGHT_PIXELS),
      new Point((xOff + IMAGE_WIDTH_PIXELS / 2), 0), redColor, 3, 4);
      //double greenX = (IMAGE_HEIGHT_PIXELS / 2);

      // Imgproc.line(openCVOverlay, new Point((IMAGE_HEIGHT_PIXELS / 2) + xOff, 25),
      //     new Point((IMAGE_HEIGHT_PIXELS / 2) + xOff, IMAGE_WIDTH_PIXELS - 10), greenColor, 3, 4);
      // double greenX = (IMAGE_HEIGHT_PIXELS / 2) + xOff;

      cvOutputStream.putFrame(openCVOverlay);

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
        // double current_min_x = minx;
        // double current_min_y = miny;
        // double current_max_x = maxx;
        // double current_max_y = maxy; 
        double shape_min_x = 99999;
        double shape_min_y = 99999;
        double shape_max_x = 0;
        double shape_max_y = 0;
        // boolean isValid = true;
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
        centerX[count] = shape_min_x + (shape_max_x - shape_min_x) / 2;
        centerY[count] = shape_min_y + (shape_max_y - shape_min_y) / 2;
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
      averageHeight.setDouble(yAve);
      hubTopLeftX.setNumber(minx);
      hubTopLeftY.setNumber(miny);
      hubBottomRightX.setNumber(maxx);
      hubBottomRightY.setNumber(maxy);      
      deviationFromCenter.setNumber(xAve - IMAGE_WIDTH_PIXELS / 2);
    });
  }

  private static void setShooterCameraExposure(int value) {
    shooterCameraExposure = value;
    if(shooterCamera != null) {
      System.out.println("setting shooter cam exposure to " + shooterCameraExposure);
      shooterCamera.setExposureManual(shooterCameraExposure);
    }
  }
}

