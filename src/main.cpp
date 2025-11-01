#include <iostream>
#include <unistd.h>
#include <limits.h>
#include "utils.h"
#include <sys/stat.h>
#include <errno.h>

int main()
{
    string videoPath = "../vid/armor.mp4";
    VideoCapture cap(videoPath);

    // 打印当前工作目录和将要打开的视频路径，便于诊断是否打开了期望的文件
    char cwd[PATH_MAX];
    if (getcwd(cwd, sizeof(cwd)) != NULL) {
        cout << "CWD: " << cwd << endl;
    }
    cout << "Opening video: " << videoPath << endl;

    if (!cap.isOpened()){
        cout<<"无法打开视频: "<< videoPath <<endl;
        return -1;
    }

    ArmorDetector detector;
    Mat frame;
    bool isRedArmor = true;
    int saveCount = 0;
    bool paused = false;

    //fps
    auto startTime = high_resolution_clock::now();
    int frameCount = 0;

    namedWindow("Armor Detection",WINDOW_NORMAL);

    while(true)
    {
        cap >> frame;
        if (frame.empty()) break;

        //调整尺寸提高性能
        resize(frame,frame,Size(640,480));

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
        
    // 显示二值化图像在单独窗口，避免覆盖主画面（便于与本地视频对比）
    Mat binaryDisplay;
    resize(binary,binaryDisplay,Size(320,240));
    imshow("Binary", binaryDisplay);

    imshow("Armor Detection",frame);

        //键盘控制
        int key = waitKey(1);
        if(key == 27) break; //ESC
        else if(key == 'r') isRedArmor = true;
        else if(key == 'b') isRedArmor = false;
        else if(key == 'p') paused = !paused; // 切换暂停
        else if(key == 's' || key == 'S')
        {
            // 保存当前帧与二值图到 captures/ 目录，避免使用系统截图
            const char *dir = "captures";
            if (mkdir(dir, 0755) != 0 && errno != EEXIST) {
                cerr << "无法创建目录 captures" << endl;
            }
            // 保存彩色帧
            char fname[256];
            snprintf(fname, sizeof(fname), "%s/frame_%04d.png", dir, saveCount);
            imwrite(fname, frame);
            // 保存二值图窗口内容（Binary 窗口中显示的图）
            Mat binarySave;
            // binaryDisplay 不是可见于这里，重新生成 small binary
            // 先从 detector.preprocess 得到的 binary would be needed; reuse previous variable by reprocessing
            Mat tempBinary = detector.preprocess(frame, isRedArmor);
            resize(tempBinary, binarySave, Size(320,240));
            snprintf(fname, sizeof(fname), "%s/binary_%04d.png", dir, saveCount);
            imwrite(fname, binarySave);
            cout << "Saved " << saveCount << " to captures/" << endl;
            saveCount++;
        }

        // 如果处于暂停状态，等待按键恢复，同时仍响应 s/r/b/ESC
        while (paused)
        {
            int k = waitKey(0);
            if (k == 27) { paused = false; break; }
            else if (k == 'p') { paused = false; break; }
            else if (k == 's' || k == 'S')
            {
                const char *dir = "captures";
                if (mkdir(dir, 0755) != 0 && errno != EEXIST) {
                    cerr << "无法创建目录 captures" << endl;
                }
                char fname[256];
                snprintf(fname, sizeof(fname), "%s/frame_%04d.png", dir, saveCount);
                imwrite(fname, frame);
                Mat tempBinary = detector.preprocess(frame, isRedArmor);
                Mat binarySave;
                resize(tempBinary, binarySave, Size(320,240));
                snprintf(fname, sizeof(fname), "%s/binary_%04d.png", dir, saveCount);
                imwrite(fname, binarySave);
                cout << "Saved " << saveCount << " to captures/" << endl;
                saveCount++;
            }
            else if (k == 'r') isRedArmor = true;
            else if (k == 'b') isRedArmor = false;
        }

    }

    cap.release();
    // destroyAllWindows();
    return 0;
}
