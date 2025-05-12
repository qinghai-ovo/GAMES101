#include "Triangle.hpp"
#include "rasterizer.hpp"
#include <eigen3/Eigen/Eigen>
#include <iostream>
#include <opencv2/opencv.hpp>

constexpr double MY_PI = 3.1415926;

Eigen::Matrix4f get_view_matrix(Eigen::Vector3f eye_pos)
{
    Eigen::Matrix4f view = Eigen::Matrix4f::Identity();

    Eigen::Matrix4f translate;
    translate << 1, 0, 0, -eye_pos[0], 
                 0, 1, 0, -eye_pos[1], 
                 0, 0, 1, -eye_pos[2], 
                 0, 0, 0, 1;

    view = translate * view;

    return view;
}

Eigen::Matrix4f get_model_matrix(float rotation_angle)
{
    Eigen::Matrix4f model = Eigen::Matrix4f::Identity();
    
    // TODO: Implement this function
    // Create the model matrix for rotating the triangle around the Z axis.
    Eigen::Matrix4f translate;
    float theta = rotation_angle * M_PI / 180.f;  //将传入的转换角度为弧度制
   
    translate << cos(theta), -sin(theta), 0, 0,
                 sin(theta), cos(theta), 0, 0,
                 0,          0,           1, 0,
                 0,          0,           0, 1;

    model = translate * model;

    return model;
}

//TODO：需要验证方法
Eigen::Matrix4f get_rotation(Vector3f axis, float angle){
    Eigen::Matrix4f rotation = Eigen::Matrix4f::Identity();
    Eigen::Matrix4f n ; //X乘矩阵
    float angle_rad = angle * M_PI / 180.f; //trans angle to rad
    
    n << 0, -axis(2), axis(1),
         axis(2), 0, -axis(0),
         -axis(1), axis(0), 0;

    //change rotate matix by Rodrigues` Rotation Formula
    rotation.block<3,3>(0,0) = cos(angle_rad)*Eigen::Matrix3f::Identity() + (1-cos(angle_rad))*axis*axis.transpose() + sin(angle_rad)*n;

    return rotation;
}

Eigen::Matrix4f get_projection_matrix(float eye_fov, float aspect_ratio,
                                      float zNear, float zFar)
{
    // Students will implement this function

    Eigen::Matrix4f projection = Eigen::Matrix4f::Identity();
    
    float theta = (eye_fov / 2.0f) * M_PI / 180.f;

    float t = zNear * tan(theta);
    float b = -t;
    float r = t * aspect_ratio;
    float l = -r;
    float n = -zNear;
    float f = -zFar;
     
    // TODO: Implement this function
    // Create the projection matrix for the given parameters.
    Eigen::Matrix4f persptoortho;
    Eigen::Matrix4f orthoTrans;
    orthoTrans << 2.0f/(r-l), 0, 0, (r+l)/ (l-r),
                  0, 2.0f / (t-b), 0, (t+b)/(b-t),
                  0, 0, 2.0f/(n-f), (n+f)/(f-n),
                  0, 0, 0, 1;

    persptoortho << n, 0, 0, 0,
                    0, n, 0, 0,
                    0, 0, n+f, -(n*f),
                    0, 0, 1, 0;
    
    projection =  orthoTrans * persptoortho * projection;
    // Then return it.

    return projection;
}

int main(int argc, const char** argv)
{
    float angle = 0;
    bool command_line = false;
    std::string filename = "output.png";

    if (argc >= 3) {
        command_line = true;
        angle = std::stof(argv[2]); // -r by default
        if (argc == 4) {
            filename = std::string(argv[3]);
        }
        else
            return 0;
    }

    rst::rasterizer r(700, 700);

    Eigen::Vector3f eye_pos = {0, 0, 5};

    std::vector<Eigen::Vector3f> pos{{2, 0, -2}, {0, 2, -2}, {-2, 0, -2}};

    std::vector<Eigen::Vector3i> ind{{0, 1, 2}};

    auto pos_id = r.load_positions(pos);
    auto ind_id = r.load_indices(ind);

    int key = 0;
    int frame_count = 0;

    if (command_line) {
        r.clear(rst::Buffers::Color | rst::Buffers::Depth);

        r.set_model(get_model_matrix(angle));
        r.set_view(get_view_matrix(eye_pos));
        r.set_projection(get_projection_matrix(45, 1, 0.1, 50));

        r.draw(pos_id, ind_id, rst::Primitive::Triangle);
        cv::Mat image(700, 700, CV_32FC3, r.frame_buffer().data());
        image.convertTo(image, CV_8UC3, 1.0f);

        cv::imwrite(filename, image);

        return 0;
    }

    while (key != 27) {
        r.clear(rst::Buffers::Color | rst::Buffers::Depth);

        r.set_model(get_model_matrix(angle));
        r.set_view(get_view_matrix(eye_pos));
        r.set_projection(get_projection_matrix(45, 1, 0.1, 50));

        r.draw(pos_id, ind_id, rst::Primitive::Triangle);

        cv::Mat image(700, 700, CV_32FC3, r.frame_buffer().data());
        image.convertTo(image, CV_8UC3, 1.0f);
        cv::imshow("image", image);
        key = cv::waitKey(10);

        std::cout << "frame count: " << frame_count++ << '\n';

        if (key == 'a') {
            angle += 10;
        }
        else if (key == 'd') {
            angle -= 10;
        }
    }

    return 0;
}
