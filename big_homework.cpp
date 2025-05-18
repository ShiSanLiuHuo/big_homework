#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include<cmath>
#include<string>
#include <Eigen/Dense>
#include<arpa/inet.h>
#include<sys/socket.h>
#include <unistd.h>

class Euler_angle;

// 全局控制参数
int brightness_delta = -30;
int brightness_threshold = 120;
int iLowH = 85, iHighH = 120;
int iLowS = 40, iHighS = 255;
int iLowV = 100, iHighV = 255;
cv::Mat image, hsv;
cv::Mat kernel1 = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3,3));  //change!!!!!
cv::Mat kernel2 = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(1,1));

std::vector<cv::Mat> rvec_list,tvec_list;
std::vector<Euler_angle> Euler_angle_list;

std::vector<cv::Mat> linear_velocity_list;
std::vector<cv::Mat> omega_list;
std::vector<cv::Mat> RotationCenter_list;

double fps = 60;
double dt = 1 / fps; //时间间格
int num = 0;

int n = 3;      //参数数量
int m = 1;      //观测量数量     

Eigen::MatrixXd A(n,n), H(m,n), Q(n,n), R(m,m),P(n,n);
Eigen::VectorXd x0(n);


    

int light_min_area = 100; //灯条最小面积
float light_max_angle = 60.0f; //灯条最大的倾斜角
int light_min_size = 5.0;
float light_contour_min_solidity = 0.3; //灯条最小凸度
float light_max_ratio = 0.8; //灯条最大长宽比
float light_max_y_diffence_ratio = 0.5;
float light_min_x_diffence_ratio =0.5;
float light_max_angle_diffence = 15.0;
float light_max_height_diffence_ratio = 0.2;
float armor_max_aspect_ratio = 2.5;
float armor_min_aspect_ratio = 1.0;


cv::Mat camera_matrix = (cv::Mat_<double>(3, 3) << 
    2065.0580175762857, 0.0, 658.9098266395495,
    0.0, 2086.886458338243, 531.5333174739342,
    0.0, 0.0, 1.0);

cv::Mat distortion_coefficients = (cv::Mat_<double>(1, 5) << 
    -0.051836613762195866, 
    0.29341513924119095, 
    0.001501183796729562, 
    0.0009386915104617738, 
    0.0);
          

enum MessageType {
    STRING_MSG = 0x0000,
    IMAGE_MSG = 0x1145,
    CAMERA_INFO = 0x1419,
    TRANSFORM = 0x1981,
    TRANSFORM_REQUEST = 0x1982
};
const unsigned short END_SYMBOL = 0x000721;
const unsigned short START_SYMBOL = 0x0D00;

#pragma pack(push,1)
struct MessageBuffer {
    unsigned short Start = 0x0D00;                // 0x0D00
    unsigned short MessageType;
    unsigned int DataID;
    unsigned int DataTotalLength;
    unsigned int Offset;
    unsigned int DataLength;
    unsigned char Data[10218]={0};
    unsigned short End = END_SYMBOL;                       // 0x0721
};
#pragma pack(pop)

void message_to_network(MessageBuffer &msg){
    msg.Start = htons(msg.Start);
    msg.MessageType = htons(msg.MessageType);
    msg.DataID = htonl(msg.DataID);
    msg.DataTotalLength = htonl(msg.DataTotalLength);
    msg.Offset = htonl(msg.Offset);
    msg.DataLength = htonl(msg.DataLength);
    msg.End = htons(msg.End);
}

void network_to_message(MessageBuffer &msg){
    msg.Start = ntohs(msg.Start);
    msg.MessageType = ntohs(msg.MessageType);
    msg.DataID = ntohl(msg.DataID);
    msg.DataTotalLength = ntohl(msg.DataTotalLength);
    msg.Offset = ntohl(msg.Offset);
    msg.DataLength = ntohl(msg.DataLength);
    msg.End = ntohs(msg.End);
}

std::map<unsigned int, std::vector<unsigned char>> data_temp;  // 临时存储分块数据

unsigned char* receive_and_decode(MessageBuffer &msg);

// 接收数据并重组
unsigned char* receive_data(int sock) {
    MessageBuffer msg;
    ssize_t total_received = 0;
    char* buffer = reinterpret_cast<char*>(&msg);

    // 循环接收完整报文
    while (total_received < sizeof(MessageBuffer)) {
        ssize_t received = recv(sock, buffer + total_received,
                              sizeof(MessageBuffer) - total_received, 0);
        if (received <= 0) {
            if (received == 0) {
                std::cerr << "Connection closed by peer" << std::endl;
            } else {
                if (errno == EAGAIN || errno == EWOULDBLOCK) {
                    std::cerr << "Receive timeout" << std::endl;
                } else {
                    perror("recv error");
                }
            }
            close(sock);
            return nullptr;
        }
        total_received += received;
    }

    // 验证数据完整性
    if (total_received != sizeof(MessageBuffer)) {
        std::cerr << "Incomplete header received (" 
                << total_received << "/" << sizeof(MessageBuffer) 
                << " bytes)" << std::endl;
        return nullptr;
    }

    // 反序列化网络字节序
    network_to_message(msg);

    // 验证协议标识
    if (msg.Start != START_SYMBOL || msg.End != END_SYMBOL) {
        std::cerr << "Invalid protocol markers: "
                << std::hex << msg.Start << " vs " << START_SYMBOL
                << ", " << msg.End << " vs " << END_SYMBOL 
                << std::dec << std::endl;
        return nullptr;
    }

    // 验证数据长度有效性
    if (msg.DataTotalLength > sizeof(msg.Data)) {
        std::cerr << "Invalid DataTotalLength: " << msg.DataTotalLength 
                << " (max " << sizeof(msg.Data) << ")" << std::endl;
        return nullptr;
    }

    // 调用分块重组函数
    return receive_and_decode(msg);
}

// 重组分块数据
unsigned char* receive_and_decode(MessageBuffer &msg) {
    unsigned int data_id = msg.DataID;
    unsigned int offset = msg.Offset;
    unsigned int length = msg.DataLength;
    unsigned int total_length = msg.DataTotalLength;

    if (data_temp.find(data_id) == data_temp.end()) {
        data_temp[data_id] = std::vector<unsigned char>(total_length);
    }

    std::memcpy(data_temp[data_id].data() + offset, msg.Data, length);

    if (offset + length >= total_length) {
        unsigned char *data = new unsigned char[total_length];
        std::memcpy(data, data_temp[data_id].data(), total_length);
        data_temp.erase(data_id);
        return data;
    } else {
        return nullptr;  
    }
}

cv::Mat calculateLinearVelocity(
    const cv::Mat& prev_pos,
    const cv::Mat& curr_pos){
        return (curr_pos - prev_pos)/dt;
    }

cv::Mat calculateAngleVelocity(
    const cv::Mat& prev_angle,
    const cv::Mat& curr_angle){
        cv::Mat r1,r2;
        cv::Rodrigues(prev_angle,r1);
        cv::Rodrigues(curr_angle,r2);

        cv::Mat delta_r = r2 *r1.t();

        cv::Mat delta_rvec;
        cv::Rodrigues(delta_r,delta_rvec);

        cv::Mat omega = delta_rvec/dt;
        return omega;
    }


cv::Mat findRotationCenter(
    const cv::Mat& p,
    const cv::Mat& omega,
    const cv::Mat& v){
        float omega_squared = cv::norm(omega, cv::NORM_L2SQR);
        if(omega_squared < 1e-6){
            std::cerr<< "omega is too small!!!" <<  std::endl;
            return p.clone();
        }

        cv::Mat c0 = omega.cross(v)/omega_squared;
        std::cout << cv::norm(c0) << std::endl;
        return p + c0;
    }


//armor
class LightDescriptor {

public:
    // 1. 默认构造函数
    LightDescriptor() = default;
    // 构造函数
    LightDescriptor(const cv::RotatedRect& rec) {
        cv::RotatedRect adjusted = adjustRec(rec, "ANGLE_WRONG"); 
        width = adjusted.size.width;
        length = adjusted.size.height;
        center = adjusted.center;
        angle = adjusted.angle;
        area = width * length;
    }

    // 4. 拷贝赋值运算符
    LightDescriptor& operator=(const LightDescriptor& L) {
        if (this == &L) return *this;
        width = L.width;
        length = L.length;
        center = L.center;
        angle = L.angle;
        area = L.area;
        return *this;
    }

    // 提供swap支持
    friend void swap(LightDescriptor& a, LightDescriptor& b) noexcept {
        using std::swap;
        swap(a.width, b.width);
        swap(a.length, b.length);
        swap(a.center, b.center);
        swap(a.angle, b.angle);
        swap(a.area, b.area);
    }

    cv::RotatedRect rec() const {
        return cv::RotatedRect(center, cv::Size2f(width, length), angle);
    }

public:
    float width;
    float length;
    cv::Point2f center;
    float angle;
    float area;

private:
    cv::RotatedRect adjustRec(const cv::RotatedRect& rec, const std::string& mode) {
        float width = rec.size.width;
        float height = rec.size.height;
        float angle = rec.angle;

        if (width > height) {
            std::swap(width, height);
            angle += 90.0f;
        }

        else if (mode == "ANGLE_WRONG") {
            // 确保角度在[-45°, 45°]范围内
            if (angle >= 90.0f) angle -= 180.0f;
            if (angle < -90.0f) angle += 180.0f;

            if (std::abs(angle) > 45.0f) {
                std::swap(width, height);
                angle = angle >= 0 ? angle - 90.0f : angle + 90.0f;
            }
        }

        while (angle >= 90.0f) angle -= 180.0f;
        while (angle < -90.0f) angle += 180.0f;

        return cv::RotatedRect(rec.center, cv::Size2f(width, height), angle);
    }
};

class ArmorDescriptor {
    public:
        ArmorDescriptor(const ArmorDescriptor& other)
            : leftLight(other.leftLight), 
              rightLight(other.rightLight) {
            std::copy(other.vertex, other.vertex + 4, vertex);
        }
    
       
        ArmorDescriptor(const LightDescriptor& l, const LightDescriptor& r) 
            : leftLight(l), rightLight(r) {
            
            // 计算装甲板的四个顶点
            const cv::Point2f& left_center = l.center;
            const cv::Point2f& right_center = r.center;
        
            // 使用灯条自身的角度计算装甲板角度
            float angle = (l.angle + r.angle) / 2.0f;
        
            // 计算装甲板的实际宽度
            cv::Point2f center_diff = right_center - left_center;
            float width = std::sqrt(center_diff.ddot(center_diff)); 
        
            // 计算高度为灯条长度的平均值
            float height = (l.length + r.length) / 2.0f ;
        
            // 创建旋转矩形
            cv::RotatedRect armor_rect(
                (left_center + right_center) / 2.0f, // 中心点
                cv::Size2f(width, height), 
                angle // 使用灯条角度
            );
            
            // 获取并调整顶点顺序（左上、右上、右下、左下）
            armor_rect.points(vertex);
            adjustVertexOrder(vertex);
        }

 

/*   
        //计算装甲板的四个顶点
        ArmorDescriptor(const LightDescriptor& L,const LightDescriptor& R)
        :leftLight(L),rightLight(R){
            cv::Point2f upper_left ;
            upper_left.x = leftLight.center.x + std::sin(leftLight.angle)*leftLight.length;
            upper_left.y = leftLight.center.y - leftLight.length*std::cos(leftLight.angle);
            
            cv::Point2f upper_right ;
            upper_right.x = rightLight.center.x + std::sin(rightLight.angle)*rightLight.length;
            upper_right.y = rightLight.center.y - rightLight.length*std::cos(rightLight.angle);
            cv::Point2f bottom_right ;
            bottom_right.x = rightLight.center.x - std::sin(rightLight.angle)*rightLight.length;
            bottom_right.y = rightLight.center.y + rightLight.length*std::cos(rightLight.angle);

            cv::RotatedRect armor_rect(upper_left,upper_right,bottom_right);
            armor_rect.points(vertex);
            adjustVertexOrder(vertex);
        }

*/
       


        LightDescriptor leftLight;
        LightDescriptor rightLight;
        cv::Point2f vertex[4];
    
        private:
        // 调整顶点顺序为顺时针
        void adjustVertexOrder(cv::Point2f* pts) {
            // 找到最左上的点作为起点
            int idx = 0;
            for (int i = 1; i < 4; ++i) {
                if (pts[i].x + pts[i].y < pts[idx].x + pts[idx].y)
                    idx = i;
            }
            // 重新排序为顺时针
            cv::Point2f temp[4];
            for (int i = 0; i < 4; ++i)
                temp[i] = pts[(idx + i) % 4];
            std::copy(temp, temp + 4, pts);
        }
    };

float distance(const cv::Point2f& p1,const cv::Point2f& p2){
    float x_diffence = p1.x - p2.x;
    float y_diffence = p1.y - p2.y;
    return sqrt(x_diffence*x_diffence + y_diffence*y_diffence);
}

//RGB better
cv::Mat splitColors(const cv::Mat& hsv_image) {
    cv::Mat mask;
    cv::inRange(hsv_image, 
                cv::Scalar(iLowH, iLowS, iLowV),
                cv::Scalar(iHighH, iHighS, iHighV), 
                mask);
    return mask;
}

cv::Mat morphologyImage(cv::Mat mask) {
    cv::morphologyEx(mask, mask, cv::MORPH_OPEN, kernel1);
    cv::dilate(mask, mask, kernel2);

    //cv::Mat horizontal_kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(1,7));
    //cv::dilate(mask, mask, horizontal_kernel);

    return mask;
}

void processImage(std::vector<std::vector<cv::Point>>& contours) {

    cv::Mat color_mask = splitColors(hsv);
    cv::Mat processed_mask = morphologyImage(color_mask);

    cv::findContours(processed_mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    cv::Mat result = image.clone();
    cv::drawContours(result, contours, -1, cv::Scalar(0,255,0), 2);


    cv::imshow("result",result);
}

void findlight(const std::vector<std::vector<cv::Point>>& contours, std::vector<LightDescriptor>& lights) {
    for (const auto& contour : contours) {
        double lightContourArea = cv::contourArea(contour);
        if (lightContourArea <= light_min_area) continue;
        if (contour.size() < 5) continue;

        // 矩形
        cv::RotatedRect lightRec = cv::minAreaRect(contour);

        LightDescriptor ld(lightRec);
        lightRec = ld.rec();

        cv::Mat hull;
        cv::convexHull(contour, hull);
        double hull_area = cv::contourArea(hull);
        double solidity = lightContourArea / hull_area;

        if (lightRec.size.width / lightRec.size.height > light_max_ratio || 
            solidity < light_contour_min_solidity)
            continue;

        // 按需调整灯条尺寸扩展
        lightRec.size.width *= 1.05f;
        lightRec.size.height *= 1.05f;

        lights.emplace_back(lightRec);
    }
}

std::vector<ArmorDescriptor> matchArmor( std::vector<LightDescriptor> lights){
    std::vector<ArmorDescriptor> armors;
    sort(lights.begin(),lights.end(),[](const LightDescriptor& ld1,const LightDescriptor& ld2){
        return ld1.center.x < ld2.center.x;
    });
    for(size_t i = 0; i < lights.size();i++){
        for(size_t j = i+1; j < lights.size();j++){
            const LightDescriptor& leftLight = lights[i];
            const LightDescriptor& rightLight = lights[j];

            //角差
            float angleDiffence = std::abs(leftLight.angle - rightLight.angle);

            //长度差比率
            float LenDiffence_ratio = std::abs(leftLight.length - rightLight.length) / std::max(leftLight.length, rightLight.length);

            //筛选
            if(angleDiffence > light_max_angle_diffence ||
               LenDiffence_ratio > light_max_height_diffence_ratio){

                continue;
            }


            //左右灯条相距距离
            float dis = distance(leftLight.center, rightLight.center);

            //左右灯条长度的平均值
            float meanLen = (leftLight.length + rightLight.length) / 2;

            //左右灯条中心点y的差值
            float yDiffence = abs(leftLight.center.y - rightLight.center.y);

            //y差比率
            float yDiffence_ratio = yDiffence / meanLen;

            //左右灯条中心点x的差值
            float xDiffence = abs(leftLight.center.x - rightLight.center.x);

            //x差比率
            float xDiffence_ratio = xDiffence / meanLen;

            float ratio = dis / meanLen;

            //筛选
            if( yDiffence_ratio > light_max_y_diffence_ratio ||
                xDiffence_ratio < light_min_x_diffence_ratio || 
               ratio > armor_max_aspect_ratio ||
               ratio < armor_min_aspect_ratio ){
                continue;
            }
            ArmorDescriptor armor(leftLight,rightLight);

            armors.emplace_back(std::move(armor));

        }
    }
    return armors;
}

std::vector<cv::Point2f> prev_center_list;

void addressImage(const cv::Mat& Image){
    image = Image.clone();
    if (image.empty()) {
        std::cerr << "Error: Failed to load image!" << std::endl;
    }

    cv::Mat undistorted;
    cv::undistort(image, undistorted, camera_matrix, distortion_coefficients);
    cv::cvtColor(undistorted, hsv, cv::COLOR_BGR2HSV);

    std::vector<std::vector<cv::Point>> contours;
    processImage(contours);
    std::vector<LightDescriptor> lights;
    findlight(contours, lights);

    cv::Mat result1 = image.clone();
    for (const auto& light : lights) {
        cv::RotatedRect rect = light.rec();
        cv::Point2f vertices[4];
        rect.points(vertices); 
        for (int i = 0; i < 4; i++) {
            cv::line(result1, vertices[i], vertices[(i+1)%4], cv::Scalar(0, 255, 0), 2);
        }
    }

    cv::imshow("result1",result1);

    std::vector<ArmorDescriptor> _armor = matchArmor(lights);
    if(_armor.empty()){
        std::cerr << "No armor!!!" << std::endl;
        cv::waitKey(1);
        return;
    }

    float min_dist = INFINITY;
    int best_idx = 0;

    if (!_armor.empty()) {
        cv::Point2f prev_center;
        for (size_t i = 0; i < _armor.size(); ++i) {
            cv::Point2f curr_center = (_armor[i].leftLight.center + _armor[i].rightLight.center) * 0.5f;
            float dist = cv::norm(curr_center - prev_center_list.back());
            if (dist < min_dist) {
                min_dist = dist;
                best_idx = i;
            }
        }
        prev_center = (_armor[best_idx].leftLight.center + _armor[best_idx].rightLight.center) * 0.5f;
        prev_center_list.push_back(prev_center);

    }
    cv::Mat _debugImg = image.clone(); 
    // 定义装甲板在3D空间中的实际坐标 左上、右上、右下、左下
    std::vector<cv::Point3f> object_points = {
        cv::Point3f(-0.0675f,  0.0275f, 0.0f), // 左上
        cv::Point3f( 0.0675f,  0.0275f, 0.0f), // 右上
        cv::Point3f( 0.0675f, -0.0275f, 0.0f), // 右下
        cv::Point3f(-0.0675f, -0.0275f, 0.0f) // 左下
    };

    
    std::vector<cv::Point> contour;
    for (int j = 0; j < 4; j++) {
        contour.emplace_back(_armor[best_idx].vertex[j]);  
    }
    cv::polylines(_debugImg, contour, true, cv::Scalar(0, 255, 0), 2);

    // 图像点
    std::vector<cv::Point2f> image_points;
    for (int j = 0; j < 4; j++) {
        image_points.push_back(_armor[best_idx].vertex[j]);
    }

    // PnP
    cv::Mat rvec, tvec;
    try {
        cv::solvePnP(object_points, image_points, 
                    camera_matrix, distortion_coefficients,
                    rvec, tvec, false, cv::SOLVEPNP_IPPE);


        rvec_list.push_back(rvec);
        tvec_list.push_back(tvec);


        // 输出
        std::cout << "tvec (meters):\n" << tvec.t() << "\n"
                << "rvec (radians):\n" << rvec.t() << "\n\n";

    } catch (const cv::Exception& e) {
        std::cerr << "Error in solvePnP: " << e.what() << std::endl;
    }
    

    cv::imshow("Armor Detection", _debugImg);
    cv::waitKey(1);
}

//tf
class Euler_angle;

class Quaternions {
public:
    Quaternions() = default;
    Quaternions(double w0, double x0, double y0, double z0)
        : w(w0), x(x0), y(y0), z(z0) {};

    Quaternions operator-(const Quaternions& other) const {
        return Quaternions(
            w - other.w,
            x - other.x,
            y - other.y,
            z - other.z
        );
    }

    double Quaternions_module() const {
        return std::sqrt(w*w + x*x + y*y + z*z);
    }

    Quaternions conjugate() const {
        return Quaternions(w, -x, -y, -z);
    }

    Quaternions inverse() const {
        double q_module = Quaternions_module();
        if (q_module == 0) {
            std::cerr << "module is zero" << std::endl;
            return Quaternions(0, 0, 0, 0);
        }
        Quaternions q_conj = conjugate();
        double scale = 1.0 / (q_module * q_module);
        return Quaternions(
            q_conj.w * scale,
            q_conj.x * scale,
            q_conj.y * scale,
            q_conj.z * scale
        );
    }

    Quaternions operator*(const Quaternions& other) const {
        double new_w = w*other.w - x*other.x - y*other.y - z*other.z;
        double new_x = w*other.x + x*other.w + y*other.z - z*other.y;
        double new_y = w*other.y - x*other.z + y*other.w + z*other.x;
        double new_z = w*other.z + x*other.y - y*other.x + z*other.w;
        return Quaternions(new_w, new_x, new_y, new_z);
    }

    Quaternions operator+(const Quaternions& other) const {
        return Quaternions(w + other.w, x + other.x, y + other.y, z + other.z);
    }

    friend Euler_angle Quaternions2Euler_angle(const Quaternions& q);
    friend std::ostream& operator<<(std::ostream& os, const Quaternions& q);

private:
    double w;
    double x;
    double y;
    double z;
};

class Euler_angle {
public:
    Euler_angle() = default;
    Euler_angle(double roll_0, double pitch_0, double yaw_0)
        : roll(roll_0), pitch(pitch_0), yaw(yaw_0) {};

    friend Quaternions Euler_angle2Quaternions(const Euler_angle& e);
    friend std::ostream& operator<<(std::ostream& os, const Euler_angle& e);

private:
    double roll;
    double pitch;
    double yaw;
};

Quaternions Euler_angle2Quaternions(const Euler_angle& e) {
    double cr = cos(e.roll / 2), sr = sin(e.roll / 2);
    double cp = cos(e.pitch / 2), sp = sin(e.pitch / 2);
    double cy = cos(e.yaw / 2), sy = sin(e.yaw / 2);

    double w = cr * cp * cy + sr * sp * sy;
    double x = sr * cp * cy - cr * sp * sy;
    double y = cr * sp * cy + sr * cp * sy;
    double z = cr * cp * sy - sr * sp * cy;

    return Quaternions(w, x, y, z);
}

Euler_angle Quaternions2Euler_angle(const Quaternions& q) {
    double mod = q.Quaternions_module();
    if (mod == 0) {
        std::cerr << "Quaternion module is zero!" << std::endl;
        return Euler_angle(0, 0, 0);
    }

    double w = q.w / mod, x = q.x / mod, y = q.y / mod, z = q.z / mod;

    double sinp = 2 * (w * y - z * x);
    if (fabs(sinp) >= 1.0) {
        std::cerr << "Euler_angle is 90!!!" << std::endl;
        return Euler_angle(
            atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)),
            copysign(M_PI / 2, sinp),
            0
        );
    }

    double yaw = atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z));
    double pitch = asin(sinp);
    double roll = atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y));
    
    return Euler_angle(roll, pitch, yaw);
}

std::ostream& operator<<(std::ostream& os, const Quaternions& q) {
    os << "(" << q.x << ", " << q.y<< ", " << q.z << ")";
    return os;
}

std::ostream& operator<<(std::ostream& os, const Euler_angle& e) {
    os << "(yaw:" << e.yaw << ", pitch:" << e.pitch << ", roll:" << e.roll << ")";
    return os;
}

Quaternions attitude_transform(const Quaternions& q1, const Quaternions& q2) {
    return q1 * q2;
}

Quaternions position_transform(const Quaternions& q, const Quaternions& t1, const Quaternions& t2) {

    Quaternions rotated = q * t1 * q.inverse();
    return rotated + t2; 
}

//kalman
class Kalman {
    public:
        Kalman(const Eigen::MatrixXd& A, const Eigen::MatrixXd& H, 
                    const Eigen::MatrixXd& Q, const Eigen::MatrixXd& R,
                    const Eigen::VectorXd& x0, const Eigen::MatrixXd& P0)
            : A(A), H(H), Q(Q), R(R), x_hat(x0), P(P0),I(Eigen:: MatrixXd::Identity(A.rows(), A.cols())) {}
        
        std::vector<Eigen::VectorXd> process(const Eigen::VectorXd& measurements) {
            std::vector<Eigen::VectorXd> estimates;
            int steps = measurements.size();
            
            for(int i=0; i<steps; i++) {
                // Prediction
                x_hat_new = A * x_hat;
                P =  A * P * A.transpose() + Q;
                
                // Update
                Eigen::VectorXd z = measurements.segment(i, 1);
                K = P * H.transpose() *(H * P * H.transpose() + R).inverse();
                x_hat_new += K * (z - H * x_hat_new);
                P = (I - K * H) * P;
                x_hat = x_hat_new;
                estimates.push_back(x_hat);
    
            }
            return estimates;
        }
    
    private:
        Eigen::MatrixXd A;      //状态转移矩阵
        Eigen::MatrixXd P;      //当前误差协方差估计
        Eigen::MatrixXd Q;      //过程噪声协方差矩阵
        Eigen::MatrixXd H;      //观测矩阵
        Eigen::MatrixXd K;      
        Eigen::MatrixXd R;      //观测噪声协方差矩阵
        Eigen::MatrixXd B;      //控制输入矩阵 =0
    
        Eigen::MatrixXd I;
    
    
        Eigen::VectorXd x_hat,x_hat_new;  //当前状态估计值
    };


void send_rotation_center(int sock, const cv::Mat& rotation_center) {
    MessageBuffer msg;
    msg.MessageType = TRANSFORM;  
    msg.DataID = 1;               // 固定 ID
    
    // 字节流
    double x = rotation_center.at<double>(0);
    double y = rotation_center.at<double>(1);
    double z = rotation_center.at<double>(2);
    
    // 填充 Data 字段
    std::memcpy(msg.Data, &x, sizeof(double));
    std::memcpy(msg.Data + 8, &y, sizeof(double));
    std::memcpy(msg.Data + 16, &z, sizeof(double));
    
    // 设置协议字段
    msg.DataTotalLength = 24;
    msg.DataLength = 24;
    msg.Offset = 0;
    
    // 序列化并发送
    message_to_network(msg);
    if (send(sock, &msg, sizeof(msg), 0) != sizeof(msg)) {
        std::cerr << "Failed to send rotation center!!!!!!" << std::endl << std::endl;
    }
}


int main() {
//kalman
    Kalman* kf_x = nullptr;
    Kalman* kf_y = nullptr;
    Kalman* kf_z = nullptr;

    A << 1, dt, 0.5*dt*dt,
         0, 1,dt,
         0, 0, 1;

    H << 1, 0, 0;

    Q << 1000, 0, 0,
         0, 1000000, 0,
         0, 0, 10;

    R << 5000;

    P << 100000, 0, 0,
         0, 100000, 0,
         0, 0, 100000;


//socket
    int client_sock = socket(AF_INET, SOCK_STREAM, 0);
    if (client_sock == -1) {
        std::cerr << "Socket creation failed" << std::endl;
        return -1;
    }
    
    sockaddr_in server_addr{};
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(8080);  // 服务端端口
    inet_pton(AF_INET, "10.2.20.28", &server_addr.sin_addr); // 服务端IP
    
    if (connect(client_sock, (sockaddr*)&server_addr, sizeof(server_addr)) == -1) {
        std::cerr << "Connection failed" << std::endl;
        close(client_sock);
        return -1;
    }



    //video
    //cv::VideoCapture cap("../10.mp4",cv::CAP_FFMPEG);


    // if(!cap.isOpened()){
    //     std::cerr << "Video not opened!!!" << std::endl;
    //     return -1;
    // }
    

    //cv::Mat prevFrame, nextFrame;
    //!!
    //cv::Mat diffFrame;

    //cap >> prevFrame;
    
    //addressImage(prevFrame);
    while(true){
        //cap >> nextFrame;
        //if(nextFrame.empty())   break;

        

        unsigned char* img_data = receive_data(client_sock);
        if (!img_data) {
            std::cerr << "Failed to receive image data" << std::endl;
            continue;
        }


        // 获取实际数据长度
        MessageBuffer tmp_msg;
        network_to_message(tmp_msg);
        int img_length = ntohl(tmp_msg.DataTotalLength);
        

        // 解码为OpenCV Mat
        cv::Mat img = cv::imdecode(cv::Mat(1, img_length, CV_8UC1, img_data), cv::IMREAD_COLOR);
        delete[] img_data; // 释放内存

        if (img.empty()) {
            std::cerr << "Decoded image is empty" << std::endl;
            continue;
        }
        if(prev_center_list.empty()){
            cv::Point2f prev_center;
            prev_center = cv::Point2f(img.cols/2,img.rows/2);
            prev_center_list.push_back(prev_center);
        }

        // 处理当前帧
        addressImage(img);

        // 计算并发送旋转中心
        if (tvec_list.size() >= 2 && rvec_list.size() >= 2) {
            // 计算线速度和角速度
            cv::Mat linear_velocity = calculateLinearVelocity(tvec_list[tvec_list.size()-2], tvec_list.back());
            cv::Mat omega = calculateAngleVelocity(rvec_list[rvec_list.size()-2], rvec_list.back());
            
            // 计算旋转中心
            cv::Mat RotationCenter = findRotationCenter(tvec_list.back(), omega, linear_velocity);
            
            // position
            double x = RotationCenter.at<double>(0);
            double y = RotationCenter.at<double>(1);
            double z = RotationCenter.at<double>(2);

            // new kalman
            if (kf_x == nullptr) {
                Eigen::VectorXd x0(3);
                x0 << x, 0, 0;
                kf_x = new Kalman(A, H, Q, R, x0, P);
                
                x0 << y, 0, 0;
                kf_y = new Kalman(A, H, Q, R, x0, P);
                
                x0 << z, 0, 0;
                kf_z = new Kalman(A, H, Q, R, x0, P);
            }

            // kalman
            Eigen::VectorXd measurement(1);
            measurement << x;
            auto filtered_x = kf_x->process(measurement).back()[0];

            measurement << y;
            auto filtered_y = kf_y->process(measurement).back()[0];

            measurement << z;
            auto filtered_z = kf_z->process(measurement).back()[0];

            // after kalman
            cv::Mat filtered_RotationCenter = (cv::Mat_<double>(3,1) << filtered_x, filtered_y, filtered_z);
            //send_rotation_center(client_sock, filtered_RotationCenter);

            RotationCenter_list.push_back(filtered_RotationCenter);
            std::cout << "x: " << filtered_x << std::endl << "y: " << filtered_y << std::endl << "z: " <<filtered_z << std::endl <<std::endl << std::endl;
            // 发送数据
            send_rotation_center(client_sock, filtered_RotationCenter);
        }
        
    }

    
    delete kf_x;
    delete kf_y;
    delete kf_z;
    cv::waitKey(0);
    //cap.release();
    close(client_sock);
    return 0;
}