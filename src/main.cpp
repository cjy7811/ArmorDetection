#include <iostream>
#include "utils.h"

int main()
{
    string videoPath = "../vid/armor.mp4";
    VideoCapture cap(videoPath);

    if (!cap.isOpened()){
        cout<<"无法打开视频: "<< videoPath <<endl;
        return -1;
    }

    ArmorDetector detector;
    Mat frame;
    bool isRedArmor = true;
    int saveCount = 0;
    bool paused = false;
    // 视频录制相关
    bool recording = true; // 启动时自动录制
    VideoWriter writer;
    string outDir = "../results";
    string outPath = outDir + "/result.avi";

    //fps
    auto startTime = high_resolution_clock::now();
    int frameCount = 0;

    namedWindow("Armor Detection",WINDOW_NORMAL);

    while(true)
    {
        if(!paused)
        {
            cap >> frame;
            if (frame.empty()) break;
            //调整尺寸提高性能
            resize(frame,frame,Size(640,480));
        }

        // 录制初始化
        if (recording && !writer.isOpened()) {
            double fps = cap.get(CAP_PROP_FPS);
            if (fps <= 0 || std::isnan(fps)) fps = 25.0;
            int fourcc = VideoWriter::fourcc('M','J','P','G');
            writer.open(outPath, fourcc, fps, frame.size(), true);
            if (!writer.isOpened()) {
                cerr << "无法打开输出视频文件: " << outPath << endl;
                recording = false;
            } else {
                cout << "Recording to: " << outPath << " @ " << fps << " FPS" << endl;
            }
        }

        //装甲板检测流程
        Mat binary = detector.preprocess(frame,isRedArmor);
        vector<LightDescriptor> lights = detector.detectLights(binary, frame);
        vector<vector<Point2f>> armors = detector.matchArmor(lights, frame);

        // 对每个检测到的装甲做 PnP 并绘制
        for (size_t k = 0; k < armors.size(); ++k)
        {
            Mat rvec, tvec;
            detector.solvePnP(armors[k], rvec, tvec);
            double distance = tvec.empty() ? 0.0 : norm(tvec);
            cout << "Frame " << cap.get(CAP_PROP_POS_FRAMES) << ": Armor["<<k<<"] detected, dist=" << distance << " m" << endl;
            detector.drawResult(frame, armors[k], rvec, tvec);
        }
        
        //计算并显示FPS
        if(!paused)
        {
            frameCount++;
            auto currentTime = high_resolution_clock::now();
            auto duration = duration_cast<milliseconds>(currentTime - startTime);

            if(duration.count() >= 1000)
            {
                double fps = frameCount * 1000.0 / duration.count();
                string fpsText = "FPS:" + to_string(fps).substr(0,4);
                putText(frame,fpsText,Point(10,60),FONT_HERSHEY_SIMPLEX,0.7,Scalar(0,255,255),2);

                frameCount = 0;
                startTime = currentTime;
            }
        }
        
        // 显示二值化图像
        Mat binaryDisplay;
        resize(binary,binaryDisplay,Size(320,240));
        imshow("Binary", binaryDisplay);

        // 在画面上标注录制状态
        if (recording && writer.isOpened()) {
            // 红色圆点 + REC 文本
            circle(frame, Point(20,20), 6, Scalar(0,0,255), -1, LINE_AA);
            putText(frame, "REC", Point(30,26), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0,0,255), 2, LINE_AA);
        }

        imshow("Armor Detection",frame);

        // 写入输出视频
        if (recording && writer.isOpened() && !paused) {
            writer.write(frame);
        }

        //键盘控制
        int key = waitKey(1);
        if(key == 27) break; //ESC
        else if(key == 'r') isRedArmor = true;
        else if(key == 'b') isRedArmor = false;
        else if(key == 'p') paused = !paused; // 切换暂停
        else if(key == 's' || key == 'S')
        {
            // 保存彩色帧
            char fname[256];
            snprintf(fname, sizeof(fname), "%s/frame_%04d.png", outDir.c_str(), saveCount);
            imwrite(fname, frame);
            // 保存二值图
            Mat binarySave;
            resize(binary, binarySave, Size(320,240));
            snprintf(fname, sizeof(fname), "%s/binary_%04d.png", outDir.c_str(), saveCount);
            imwrite(fname, binarySave);
            cout << "Saved " << saveCount << endl;
            saveCount++;
        }
    }

    cap.release();
    if (writer.isOpened()) writer.release();
    destroyAllWindows();
    return 0;
}
