package com.example.opencvtest;

import androidx.appcompat.app.AppCompatActivity;

import android.graphics.Bitmap;
import android.media.Image;
import android.os.Bundle;
import android.os.Environment;
import android.util.Log;
import android.view.SurfaceView;
import android.view.View;
import android.widget.Button;
import android.widget.Toast;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.JavaCameraView;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvException;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.RotatedRect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Date;
import java.util.Iterator;
import java.util.List;

public class MainActivity extends AppCompatActivity implements CameraBridgeViewBase.CvCameraViewListener2, View.OnClickListener {

    private static String TAG= "MainActivity";
    CameraBridgeViewBase cameraBridgeViewBase;
    BaseLoaderCallback baseLoaderCallback;
    Button takePicture;
    Mat originalFrame;
    MatOfPoint2f approx;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        takePicture = (Button) findViewById(R.id.my_btn_take_picture);
        cameraBridgeViewBase = (JavaCameraView) findViewById(R.id.my_camera_view);

        takePicture.setOnClickListener(MainActivity.this);
        cameraBridgeViewBase.setVisibility(SurfaceView.VISIBLE);
        cameraBridgeViewBase.setCvCameraViewListener(MainActivity.this);

        //baseloader
        baseLoaderCallback = new BaseLoaderCallback(this) {
            @Override
            public void onManagerConnected(int status) {
                super.onManagerConnected(status);

                switch (status){
                    case BaseLoaderCallback.SUCCESS:
                        cameraBridgeViewBase.enableView();
                        break;
                    default:
                        super.onManagerConnected(status);
                        break;
                }
            }
        };
    }

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
            approx = approxCurve;
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

        this.originalFrame = originalFrame;
        return originalFrame;
    }



    @Override
    protected void onResume() {
        super.onResume();
        if (!OpenCVLoader.initDebug()){
            Toast.makeText(getApplicationContext(),"There's a problem, yo!", Toast.LENGTH_SHORT).show();
        }
        else
        {
            baseLoaderCallback.onManagerConnected(baseLoaderCallback.SUCCESS);
        }
    }

    @Override
    protected void onPause() {
        super.onPause();
        if(cameraBridgeViewBase!=null){

            cameraBridgeViewBase.disableView();
        }

    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (cameraBridgeViewBase!=null){
            cameraBridgeViewBase.disableView();
        }
    }

    @Override
    public void onClick(View view) {
        //Reordering points

        Mat destImage = new Mat(400, 400, CvType.CV_8UC1);

        Mat dst = new MatOfPoint2f(new Point(0, 0),
                new Point(approx.width() - 1, 0),
                new Point(approx.width() - 1,approx.height() - 1),
                new Point(0, approx.height() - 1));
        Mat src = new MatOfPoint2f(new Point(0, 0),
                new Point(400, 0),
                new Point(400, 400),
                new Point(0, 400));

        Mat transform = Imgproc.getPerspectiveTransform(src, dst);
        Imgproc.warpPerspective(originalFrame, destImage, transform, destImage.size());
        getImage(destImage);
    }

    public void getImage(Mat rgba) {
        Bitmap bmp = null;
        bmp = Bitmap.createBitmap(rgba.cols(), rgba.rows(), Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(rgba, bmp);


        rgba.release();


        FileOutputStream out = null;

        String filename = "frame.png";
        File dest = new File(Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS)+"/"+ filename);

        try {
            dest.createNewFile(); // if file already exists will do nothing
            out = new FileOutputStream(dest);
            bmp.compress(Bitmap.CompressFormat.PNG, 100, out); // bmp is your Bitmap instance
            // PNG is a lossless format, the compression factor (100) is ignored
        } catch (Exception e) {
            e.printStackTrace();
            Log.d(TAG, e.getMessage());
        } finally {
            try {
            if (out != null) {
                out.close();
                Log.d(TAG, "OK!!");
            }
            } catch (IOException e) {
                Log.d(TAG, e.getMessage() + "Error");
                e.printStackTrace();
            }
            }

    }
}
