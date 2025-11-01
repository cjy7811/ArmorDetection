#ifndef UTILS_H
#define UTILS_H

#include <vector>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <chrono>

using namespace cv;
using namespace std;
using namespace std::chrono;

//灯条描述类
class LightDescriptor
{
public:
    float width,height,angle;
    Point2f center;
    RotatedRect rect; // 保存拟合椭圆/矩形以便调试绘制
    float meanV = 0.0f; // V 通道平均亮度，用于过滤反射

    LightDescriptor(){}
    LightDescriptor(const RotatedRect& light)
    {
        width = light.size.width;
        height = light.size.height;
        center = light.center;
        angle = light.angle;
        rect = light;
    }
};

//装甲板识别类
class ArmorDetector
{
private:
    //相机内参
    Mat cameraMatrix = (Mat_<double>(3,3)<<
    9.28130989e+02,0,3.77572945e+02,0,9.30138391e+02,2.83892859e+02,0,0,1.0000);
    //畸变系数
    Mat distCoeffs = (Mat_<double>(1,5)<<
    -2.54433647e-01,5.69431382e-01,3.65405229e-03,-1.09433818e-03,-1.33846840e+00);

    //装甲板3D模型点（单位：米）
    std::vector<Point3f> objectPoints = 
    {
        Point3f(-0.065,0.025,0),
        Point3f(-0.065,-0.025,0),
        Point3f(0.065,-0.025,0),
        Point3f(0.065,0.025,0)
    };

    // 将四个角点按顺序排列为: top-left, top-right, bottom-right, bottom-left
    vector<Point2f> orderCorners(const vector<Point2f>& corners)
    {
        if (corners.size() != 4) return {};
        vector<Point2f> pts = corners;
        // 按 y 排序，前两个为上排，后两个为下排
        sort(pts.begin(), pts.end(), [](const Point2f &a, const Point2f &b){ return a.y < b.y; });

        vector<Point2f> top = {pts[0], pts[1]};
        vector<Point2f> bottom = {pts[2], pts[3]};

        // 在每排内按 x 排序，使得 left 在前
        if (top[0].x > top[1].x) swap(top[0], top[1]);
        if (bottom[0].x > bottom[1].x) swap(bottom[0], bottom[1]);

        // 返回顺时针或逆时针一致的顺序：TL, TR, BR, BL
        return { top[0], top[1], bottom[1], bottom[0] };
    }

public:
    // （调试函数已移除）
    //预处理函数
    Mat preprocess(Mat& frame,bool isRed)
    {
        Mat hsv,binary;
        // 转为 HSV 并平滑，减少局部亮度波动导致的断裂
        cvtColor(frame,hsv,COLOR_BGR2HSV);
        GaussianBlur(hsv, hsv, Size(5,5), 0);

        // 根据敌方颜色选择阈值（扩大红色范围以包含偏橙的灯条，并降低 S/V 下限以容忍局部较暗区域）
        if (isRed)
        {
            // 红色/橙色：把上限扩展到 20 度，以包含橙色灯条
            Scalar lower_red1(0,30,30);
            Scalar upper_red1(20,255,255);
            Scalar lower_red2(160,30,30);
            Scalar upper_red2(180,255,255);

            Mat binary1,binary2;
            inRange(hsv,lower_red1,upper_red1,binary1);
            inRange(hsv,lower_red2,upper_red2,binary2);
            binary = binary1 | binary2;
        }
        else
        {
            // 蓝色
            Scalar lower_blue(100,30,30);
            Scalar upper_blue(130,255,255);
            inRange(hsv,lower_blue,upper_blue,binary);
        }

        // 另外使用 V 通道的高亮掩模来捕捉亮点（在极暗背景下单纯色阈值可能失效）
        vector<Mat> hsvChannels;
        split(hsv, hsvChannels);
        Mat vmask;
        // V 阈值可适当调低以捕获较暗环境中的高亮点
        threshold(hsvChannels[2], vmask, 50, 255, THRESH_BINARY);
        binary = binary | vmask;

        // 使用先纵向再横向的闭运算连接竖直灯条（避免中间断裂被分割）
        Mat kernelV = getStructuringElement(MORPH_RECT, Size(3,7));
        morphologyEx(binary, binary, MORPH_CLOSE, kernelV, Point(-1,-1), 2);
        Mat kernelH = getStructuringElement(MORPH_RECT, Size(7,3));
        morphologyEx(binary, binary, MORPH_CLOSE, kernelH, Point(-1,-1), 1);
        // 额外的开运算去小噪声
        Mat kernel = getStructuringElement(MORPH_RECT, Size(3,3));
        morphologyEx(binary, binary, MORPH_OPEN, kernel, Point(-1,-1), 1);

        return binary;
    }

    //灯条检测
    // detectLights 现在需要原始帧以便计算亮度信息（V 通道）用于过滤反射
    vector<LightDescriptor> detectLights(Mat& binary, Mat& frame)
    {
        vector<vector<Point>> contours;
        vector<Vec4i> hierarchy;
        findContours(binary,contours,hierarchy,RETR_EXTERNAL,CHAIN_APPROX_SIMPLE);

        vector<LightDescriptor> lights;

    // 先准备 HSV V 通道用于计算每个轮廓的平均亮度
    Mat hsvFrame;
    cvtColor(frame, hsvFrame, COLOR_BGR2HSV);
    vector<Mat> hsvCh;
    split(hsvFrame, hsvCh);

    // 初步宽松筛选并保存候选轮廓与拟合矩形，以便后续合并近邻轮廓
        vector<RotatedRect> rects;
        vector<vector<Point>> rectContours;
    vector<float> rectMeanV;
        for(size_t i = 0;i < contours.size();i++)
        {
            // 过滤小面积轮廓（进一步放宽阈值以提高召回，thin lights 会很窄）
            double area = contourArea(contours[i]);
            if (area < 8 || contours[i].size() < 5)
                continue;

            // 椭圆拟合（对候选轮廓）
            RotatedRect lightRect = fitEllipse(contours[i]);

            // 确保 size.height 为长边
            if (lightRect.size.width > lightRect.size.height)
            {
                swap(lightRect.size.width, lightRect.size.height);
                lightRect.angle += 90.0f;
                if (lightRect.angle > 180.0f) lightRect.angle -= 180.0f;
            }

            // 更宽松的长宽比与角度筛选，允许更多候选进入合并阶段
            float ratio = lightRect.size.height / lightRect.size.width;
            float ang = lightRect.angle;
            if (ang > 90.0f) ang -= 180.0f;
            if (ratio < 1.5f || ratio > 9.0f) continue;
            if (abs(ang) > 40.0f) continue;

            rects.push_back(lightRect);
            rectContours.push_back(contours[i]);
            // 计算该轮廓区域的 mean V
            Mat mask = Mat::zeros(frame.size(), CV_8UC1);
            drawContours(mask, contours, (int)i, Scalar(255), FILLED);
            Scalar meanV = cv::mean(hsvCh[2], mask);
            rectMeanV.push_back((float)meanV[0]);
        }

        // 合并近邻轮廓：如果两个候选的中心很接近且角度相近，则把它们的点集合合并并重新拟合
        int n = (int)rects.size();
        vector<bool> removed(n,false);
        for (int a = 0; a < n; ++a)
        {
            if (removed[a]) continue;
            for (int b = a+1; b < n; ++b)
            {
                if (removed[b]) continue;
                Point2f ca = rects[a].center;
                Point2f cb = rects[b].center;
                float dist = norm(ca - cb);
                float meanH = (rects[a].size.height + rects[b].size.height) * 0.5f;
                float angA = rects[a].angle; if (angA > 90.0f) angA -= 180.0f;
                float angB = rects[b].angle; if (angB > 90.0f) angB -= 180.0f;
                float angDiff = abs(angA - angB);
                // 如果中心非常接近或高度接近且角度差不大，则合并
                if (dist < 0.6f * meanH && angDiff < 25.0f)
                {
                    // 合并点集
                    vector<Point> mergedPts = rectContours[a];
                    mergedPts.insert(mergedPts.end(), rectContours[b].begin(), rectContours[b].end());
                    // 重新拟合椭圆（需要至少 5 点）
                    if (mergedPts.size() >= 5)
                    {
                        RotatedRect newRect = fitEllipse(mergedPts);
                        if (newRect.size.width > newRect.size.height)
                        {
                            swap(newRect.size.width, newRect.size.height);
                            newRect.angle += 90.0f;
                            if (newRect.angle > 180.0f) newRect.angle -= 180.0f;
                        }
                        rects[a] = newRect;
                        rectContours[a] = mergedPts;
                        removed[b] = true;
                        // 合并后更新 meanV（重新计算）
                        Mat mask = Mat::zeros(frame.size(), CV_8UC1);
                        drawContours(mask, rectContours[a], -1, Scalar(255), FILLED);
                        Scalar meanV = cv::mean(hsvCh[2], mask);
                        rectMeanV[a] = (float)meanV[0];
                    }
                }
            }
        }

        // 将合并结果转换为最终灯条描述，并做最终较严格筛选
        for (int i = 0; i < n; ++i)
        {
            if (removed[i]) continue;
            RotatedRect lr = rects[i];
            float meanV = rectMeanV[i];
            float ratio = lr.size.height / lr.size.width;
            float ang = lr.angle; if (ang > 90.0f) ang -= 180.0f;
            // 最终筛选：恢复适度严格的阈值，减少误报
            if (ratio < 1.4f || ratio > 8.0f) continue;
            if (abs(ang) > 35.0f) continue;

            LightDescriptor ld(lr);
            ld.meanV = meanV;
            lights.push_back(ld);
        }

        // 过滤可能的反射/投影：如果有一对灯条在 x 方向接近，且一个在另一个下方且亮度显著更低且高度更小，则很可能是反射，去掉下面的
        vector<bool> isRef(lights.size(), false);
        for (size_t i = 0; i < lights.size(); ++i)
        {
            for (size_t j = 0; j < lights.size(); ++j)
            {
                if (i == j) continue;
                // 如果 j 在 i 的下方（y 更大）且 x 接近
                float dx = abs(lights[i].center.x - lights[j].center.x);
                float dy = lights[j].center.y - lights[i].center.y;
                if (dy > 5.0f && dy < max(lights[i].height, lights[j].height) * 5.0f && dx < max(lights[i].width, lights[j].width) * 0.6f)
                {
                    // j 更暗且更小 -> 视为反射
                    if (lights[j].meanV < 0.75f * lights[i].meanV && lights[j].height < 0.95f * lights[i].height)
                    {
                        isRef[j] = true;
                    }
                }
            }
        }
        vector<LightDescriptor> filtered;
        for (size_t i = 0; i < lights.size(); ++i) if (!isRef[i]) filtered.push_back(lights[i]);
        return filtered;

        return lights;
    }

    //装甲板匹配
    // 参数 frame 用于 debug 模式下在画面上标注每对候选灯条的判定信息
    // 返回所有匹配到的装甲，每个装甲由按序排列的 4 个角点组成 (TL,TR,BR,BL)
    vector<vector<Point2f>> matchArmor(vector<LightDescriptor>& lights, Mat& frame)
    {
        if(lights.size()<2)
            return{};

        vector<vector<Point2f>> armors;

        for(size_t i = 0;i < lights.size();i++)
        {
            for(size_t j = i+1;j < lights.size();j++)
            {
                LightDescriptor& leftLight = lights[i];
                LightDescriptor& rightLight = lights[j];

                // 计算判定指标
                // 先将角度归一化到 [-90,90] 再比较，避免 177/ -177 之类的假差异
                float a1 = leftLight.angle;
                float a2 = rightLight.angle;
                if (a1 > 90.0f) a1 -= 180.0f;
                if (a2 > 90.0f) a2 -= 180.0f;
                float angleDiff = abs(a1 - a2);
                float heightDiff = abs(leftLight.height - rightLight.height);
                float heightMax = max(leftLight.height, rightLight.height);
                float heightDiffRel = heightDiff / (heightMax + 1e-6f);
                float xDiff = abs(leftLight.center.x - rightLight.center.x);
                float yDiff = abs(leftLight.center.y - rightLight.center.y);
                float distance = norm(leftLight.center - rightLight.center);
                float meanHeight = (leftLight.height + rightLight.height) / 2.0f;
                float ratio = distance / (meanHeight + 1e-6f);

                // 判定规则（可视化输出用标签 reason）
                string reason = "OK";
                // 放宽部分阈值以容忍真实世界波动
                if (angleDiff > 15.0f) reason = "angle";
                else if (heightDiffRel > 0.6f) reason = "hDiff";
                else if (xDiff < 1.2f * max(leftLight.width, rightLight.width)) reason = "xTooClose";
                else if (yDiff > meanHeight * 0.6f) reason = "yOffset";
                else if (ratio < 0.8f || ratio > 6.0f) reason = "ratio";

                // (已移除调试输出与可视化)

                if (reason == "OK")
                {
                    // 确保 left/right 按 x 坐标顺序（left 在左侧）
                    LightDescriptor &L = (leftLight.center.x <= rightLight.center.x) ? leftLight : rightLight;
                    LightDescriptor &R = (&L == &leftLight) ? rightLight : leftLight;

                    vector<Point2f> corners = getArmorCorners(L, R);
                    // 将角点排序为 TL,TR,BR,BL，便于按顺序绘制矩形
                    vector<Point2f> ordered = orderCorners(corners);
                    // 过滤重复：如果已有装甲中心与本装甲中心太近，则跳过
                    Point2f center(0,0);
                    for (auto &p : ordered) center += p;
                    center *= 0.25f;
                    bool isDuplicate = false;
                    for (auto &a : armors)
                    {
                        Point2f c(0,0);
                        for (auto &p : a) c += p;
                        c *= 0.25f;
                        if (norm(c - center) < 10.0f) { isDuplicate = true; break; }
                    }
                    if (!isDuplicate) armors.push_back(ordered);
                    // 不 break，继续寻找同一帧的其他装甲
                }
            }
        }
        return armors;
    }

    //获取角点
    vector<Point2f> getArmorCorners(LightDescriptor& left,LightDescriptor& right)
    {
        vector<Point2f> corners;
        //计算灯条方向向量
        // 不在原对象上直接修改角度，使用局部变量进行归一化和计算
        double leftAngleDeg = left.angle;
        double rightAngleDeg = right.angle;
        if (leftAngleDeg > 90.0) leftAngleDeg -= 180.0;
        if (rightAngleDeg > 90.0) rightAngleDeg -= 180.0;
        float leftAngle = static_cast<float>(leftAngleDeg * CV_PI / 180.0);
        float rightAngle = static_cast<float>(rightAngleDeg * CV_PI / 180.0);

        Point2f leftDir(sin(leftAngle), -cos(leftAngle));
        Point2f rightDir(sin(rightAngle), -cos(rightAngle));

        Point2f leftTop = left.center+leftDir*left.height*0.5f;
        Point2f leftBottom = left.center-leftDir*left.height*0.5f;
        Point2f rightTop = right.center+rightDir*right.height*0.5f;
        Point2f rightBottom = right.center-rightDir*right.height*0.5f;

        corners = {leftTop,leftBottom,rightTop,rightBottom};
        return corners;
    }

    //PnP解算
    void solvePnP(vector<Point2f>& imagePoints,Mat& rvec,Mat& tvec)
    {
        if (imagePoints.size()==4)
        {
            cv::solvePnP(objectPoints,imagePoints,cameraMatrix,distCoeffs,rvec,tvec);
        }
    }

    //绘制结果：在装甲板中心绘制 Oxyz 坐标系（需要 rvec 和 tvec）
    void drawResult(Mat& frame,vector<Point2f>& armorPoints,Mat& rvec,Mat& tvec)
    {
        if (armorPoints.size() == 4)
        {
            for(int i = 0;i<4;i++)
            {
                line(frame,armorPoints[i],armorPoints[(i+1)%4],Scalar(0,255,0),2);
            }

            // 计算图像中心（装甲中心）
            Point2f center(0,0);
            for(auto& point : armorPoints)
            {
                center += point;
            }
            center *= 0.25f;

            // 绘制坐标系：使用三维坐标轴点，基于相机内参投影到图像平面
            if(!rvec.empty() && !tvec.empty())
            {
                // 轴长度（米） — 与 objectPoints 单位一致
                float axisLen = 0.08f; // 调整可视大小
                vector<Point3f> axesPts;
                axesPts.push_back(Point3f(0,0,0)); // 原点
                axesPts.push_back(Point3f(axisLen,0,0)); // X
                axesPts.push_back(Point3f(0,axisLen,0)); // Y
                axesPts.push_back(Point3f(0,0,axisLen)); // Z

                vector<Point2f> imgPts;
                projectPoints(axesPts, rvec, tvec, cameraMatrix, distCoeffs, imgPts);

                if (imgPts.size() >= 4)
                {
                    // 原点
                    Point2f o = imgPts[0];
                    // Function to draw axis with arrowhead
                    auto drawAxis = [&](const Point2f &tip, const Scalar &color){
                        // shaft
                        line(frame, o, tip, color, 3, LINE_AA);
                        // arrowhead parameters (pixels)
                        float fullLen = norm(tip - o);
                        float arrowLen = min(20.0f, max(8.0f, fullLen * 0.18f));
                        float arrowWidth = arrowLen * 0.5f;
                        Point2f dir = tip - o;
                        if (norm(dir) < 1e-3) return;
                        dir *= (1.0f / norm(dir));
                        // base point of arrow
                        Point2f base = tip - dir * arrowLen;
                        // perpendicular
                        Point2f perp(-dir.y, dir.x);
                        Point2f p1 = tip;
                        Point2f p2 = base + perp * arrowWidth;
                        Point2f p3 = base - perp * arrowWidth;
                        Point pts[1][3];
                        pts[0][0] = p1; pts[0][1] = p2; pts[0][2] = p3;
                        const Point* ppt[1] = { pts[0] };
                        int npt[] = { 3 };
                        fillPoly(frame, ppt, npt, 1, color, LINE_AA);
                    };

                        // Before drawing, ensure axis directions are consistent and intuitive:
                        // - X should project to the right in image (positive image x)
                        // - Y should project upwards in image (negative image y because image y grows down)
                        // - Z should point toward the camera (negative camera Z)
                        Mat R;
                        Rodrigues(rvec, R);
                        // R is 3x3 double; its columns are the object axes in camera coordinates
                        double r00 = R.at<double>(0,0);
                        double r11 = R.at<double>(1,1);
                        double r22 = R.at<double>(2,2);
                        // Build axes in object space, flipping signs if necessary
                        vector<Point3f> axesToProj;
                        float ax = axisLen;
                        // X: if r00 < 0, flip X so it projects to the right
                        axesToProj.push_back(Point3f(0,0,0));
                        axesToProj.push_back(Point3f(r00 < 0 ? -ax : ax, 0, 0));
                        // Y: if r11 > 0, object +Y maps to camera +Y (downwards), so flip to make it point up in image
                        axesToProj.push_back(Point3f(0, r11 > 0 ? -ax : ax, 0));
                        // Z: if r22 > 0, object +Z maps to camera +Z (away from camera), so flip to point toward camera
                        axesToProj.push_back(Point3f(0,0, r22 > 0 ? -ax : ax));

                        vector<Point2f> imgPts2;
                        projectPoints(axesToProj, rvec, tvec, cameraMatrix, distCoeffs, imgPts2);
                        // Draw using adjusted projections
                        // Draw the adjusted projected axes (no labels)
                        drawAxis(imgPts2[1], Scalar(0,0,255));
                        drawAxis(imgPts2[2], Scalar(0,255,0));
                        drawAxis(imgPts2[3], Scalar(255,0,0));

                    // draw origin with small filled circle and border
                    circle(frame, o, 4, Scalar(255,255,255), -1, LINE_AA);
                    circle(frame, o, 6, Scalar(0,0,0), 1, LINE_AA);
                }
            }
            else
            {
                // 如果没有 rvec/tvec，仍然在像素中心画一个点
                circle(frame, center, 5, Scalar(0,0,255), -1);
            }

            // 显示距离信息（如果有 tvec）
            if(!tvec.empty())
            {
                double distance = norm(tvec);
                string distanceText = "Dist:" + to_string(distance).substr(0,4) + "m";
                putText(frame,distanceText,Point(10,30),FONT_HERSHEY_SIMPLEX,0.7,Scalar(0,255,255),2);
            }
        }
    }
};

#endif
