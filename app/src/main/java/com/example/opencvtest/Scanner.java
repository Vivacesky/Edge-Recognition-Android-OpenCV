package com.example.opencvtest;

import org.opencv.android.CameraBridgeViewBase;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.List;

public class Scanner implements CameraBridgeViewBase.CvCameraViewListener2 {

    @Override
    public void onCameraViewStarted(int width, int height) {
    }

    @Override
    public void onCameraViewStopped() {
    }

    @Override
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
        Mat originalFrame = new Mat();

        inputFrame.rgba().copyTo(originalFrame);
        Mat frame = inputFrame.rgba();

        //GrayScale
        Imgproc.cvtColor(frame, frame, Imgproc.COLOR_BGR2GRAY);
        //Blur
        Imgproc.GaussianBlur(frame, frame, new Size(5, 5), 5);
        //Edge Detection
        Imgproc.Canny(frame, frame, 75, 150);
//        Imgproc.threshold(frame, frame, 60,255, Imgproc.THRESH_BINARY);
        //Dilate&Erode
        final Size kernelSize = new Size(5, 5);
        final Point anchor = new Point(-1, -1);
        Mat kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, kernelSize);
        Imgproc.dilate(frame, frame, kernel, anchor,2);
        Imgproc.erode(frame, frame, kernel, anchor, 1);
        //Find Contours in the edged frame, save the largest ones
        List<MatOfPoint> contours = new ArrayList<MatOfPoint>();
        Imgproc.findContours(frame, contours, new Mat(), Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);
        MatOfPoint2f approxCurve = null;
        //Write the largest contour
        double maxVal = 0;
        int maxValIdx = 0;
        for (int contourIdx = 0; contourIdx < contours.size(); contourIdx++)
        {
            MatOfPoint2f temp = new MatOfPoint2f(contours.get(contourIdx).toArray());
            approxCurve = new MatOfPoint2f();
            Imgproc.approxPolyDP(temp, approxCurve,
                    Imgproc.arcLength(temp, true) * 0.04, true);
            if(approxCurve.total() == 4 ){
                double contourArea = Imgproc.contourArea(contours.get(contourIdx));
                if (maxVal < contourArea)
                {
                    maxVal = contourArea;
                    maxValIdx = contourIdx;
                }
            }
        }
        if(contours.size() > maxValIdx){
            Imgproc.drawContours(originalFrame, contours, maxValIdx, new Scalar(17, 103, 53), 10);
        }
        return originalFrame;
    }


}
