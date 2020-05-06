package com.example.opencvtest;

import android.graphics.Bitmap;
import android.os.Environment;
import android.util.Log;

import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.Utils;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.utils.Converters;

import java.io.File;
import java.io.FileOutputStream;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class OpenCvHelper {

    private Mat untouched;
    private Point p1, p2,p3,p4;

    public Mat getRectangle(CameraBridgeViewBase.CvCameraViewFrame inputFrame){
        Mat originalFrame = new Mat();
        inputFrame.rgba().copyTo(originalFrame);
        untouched = originalFrame.clone();
        Mat frame = inputFrame.rgba();

        //GrayScale
        Imgproc.cvtColor(frame, frame, Imgproc.COLOR_BGR2GRAY);
        //Blur
        Imgproc.GaussianBlur(frame, frame, new Size(5, 5), 10);
        //Edge Detection
        Imgproc.Canny(frame, frame, 75, 150);
        Imgproc.threshold(frame, frame, 60,255, Imgproc.THRESH_BINARY);
        //Dilate&Erode
        final Size kernelSize = new Size(5, 5);
        final Point anchor = new Point(-1, -1);
        Mat kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, kernelSize);
        Imgproc.dilate(frame, frame, kernel, anchor,2);
        Imgproc.erode(frame, frame, kernel, anchor, 1);
        //Find Contours in the edged frame, save the largest ones
        List<MatOfPoint> contours = new ArrayList<MatOfPoint>();
        Imgproc.findContours(frame, contours, new Mat(), Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);
        MatOfPoint2f a = new MatOfPoint2f();

        MatOfPoint2f approxCurve = new MatOfPoint2f();
        //Write the largest contour
        double maxVal = 100000;
        int maxValIdx = -1;
        for (int contourIdx = 0; contourIdx < contours.size(); contourIdx++)
        {
            MatOfPoint2f temp = new MatOfPoint2f(contours.get(contourIdx).toArray());
            approxCurve = new MatOfPoint2f();
            Imgproc.approxPolyDP(temp, approxCurve,
                    Imgproc.arcLength(temp, true) * 0.003, true);
            if(approxCurve.total() == 4 ){
                double contourArea = Imgproc.contourArea(contours.get(contourIdx));
                if (maxVal < contourArea)
                {
                    maxVal = contourArea;
                    maxValIdx = contourIdx;
                    a = approxCurve;
                }
            }
        }

        if(-1 != maxValIdx){
            double temp_double[] = a.get(0, 0);
            Point p1 = new Point(temp_double[0], temp_double[1]);
            Imgproc.circle(originalFrame, new Point(p1.x, p1.y), 10, new Scalar(17, 103, 53), 5);
            this.p1 = p1;
            temp_double = a.get(1, 0);
            Point p2 = new Point(temp_double[0], temp_double[1]);
            Imgproc.circle(originalFrame, new Point(p2.x, p2.y), 10, new Scalar(17, 103, 53), 5);
            this.p2 = p2;

            temp_double = a.get(2, 0);
            Point p3 = new Point(temp_double[0], temp_double[1]);
            Imgproc.circle(originalFrame, new Point(p3.x, p3.y), 10, new Scalar(17, 103, 53), 5);
            this.p3 = p3;

            temp_double = a.get(3, 0);
            Point p4 = new Point(temp_double[0], temp_double[1]);
            Imgproc.circle(originalFrame, new Point(p4.x, p4.y), 10, new Scalar(17, 103, 53), 5);
            this.p4 = p4;
            Imgproc.drawContours(originalFrame, contours, maxValIdx, new Scalar(17, 103, 53),3);
        }
        return originalFrame;
    }

    public Bitmap warp(Bitmap image, Point p1, Point p2, Point p3, Point p4){
        int resultWidth = 500;
        int resultHeight = 700;

        Mat inputMat = new Mat(image.getHeight(), image.getHeight(), CvType.CV_8UC4);
        Utils.bitmapToMat(image, inputMat);
        Mat outputMat = new Mat(resultWidth, resultHeight, CvType.CV_8UC4);

        Point ocvPIn1 = new Point(p2.x , p4.y );
        Point ocvPIn2 = new Point(p1.x , p3.y);
        Point ocvPIn3 = new Point(p4.x , p2.y);
        Point ocvPIn4 = new Point(p3.x , p1.y);
        List<Point> source = new ArrayList<Point>();
        source.add(ocvPIn1);
        source.add(ocvPIn2);
        source.add(ocvPIn3);
        source.add(ocvPIn4);
        Mat startM = Converters.vector_Point2f_to_Mat(source);

        Point ocvPOut1 = new Point(0, 0);
        Point ocvPOut2 = new Point(0, resultHeight);
        Point ocvPOut3 = new Point(resultWidth, resultHeight);
        Point ocvPOut4 = new Point(resultWidth, 0);
        List<Point> dest = new ArrayList<Point>();
        dest.add(ocvPOut1);
        dest.add(ocvPOut2);
        dest.add(ocvPOut3);
        dest.add(ocvPOut4);
        Mat endM = Converters.vector_Point2f_to_Mat(dest);

        Mat perspectiveTransform = Imgproc.getPerspectiveTransform(startM, endM);

        Imgproc.warpPerspective(inputMat,
                outputMat,
                perspectiveTransform,
                new Size(resultWidth, resultHeight),
                Imgproc.INTER_CUBIC);
        Imgproc.cvtColor(outputMat,outputMat, Imgproc.COLOR_BGR2GRAY);

        Imgproc.threshold(outputMat, outputMat,   40,125, Imgproc.THRESH_TOZERO);

        Bitmap output = Bitmap.createBitmap(resultWidth, resultHeight, Bitmap.Config.RGB_565);
        Utils.matToBitmap(outputMat, output);
        return output;
    }

    public void saveImage(){
        Bitmap myBitmap = Bitmap.createBitmap(untouched.cols(), untouched.rows(),
                Bitmap.Config.ARGB_8888);;
        Utils.matToBitmap(untouched,myBitmap);
        myBitmap = warp(myBitmap, p1,p2,p3,p4);
        String root = Environment.getExternalStorageDirectory().toString();
        File myDir = new File(root + "/req_images");
        myDir.mkdirs();
        Random generator = new Random();
        int n = 10000;
        n = generator.nextInt(n);
        String fname = "Image-" + n + ".jpg";
        File file = new File(myDir, fname);
        if (file.exists())
            file.delete();
        try {
            FileOutputStream out = new FileOutputStream(file);
            myBitmap.compress(Bitmap.CompressFormat.JPEG, 100, out);
            out.flush();
            out.close();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
