// clang-format off

#include <algorithm>
#include <vector>
#include "rasterizer.hpp"
#include <opencv2/opencv.hpp>
#include <math.h>


rst::pos_buf_id rst::rasterizer::load_positions(const std::vector<Eigen::Vector3f> &positions)
{
    auto id = get_next_id();
    pos_buf.emplace(id, positions);

    return {id};
}

rst::ind_buf_id rst::rasterizer::load_indices(const std::vector<Eigen::Vector3i> &indices)
{
    auto id = get_next_id();
    ind_buf.emplace(id, indices);

    return {id};
}

rst::col_buf_id rst::rasterizer::load_colors(const std::vector<Eigen::Vector3f> &cols)
{
    auto id = get_next_id();
    col_buf.emplace(id, cols);

    return {id};
}

auto to_vec4(const Eigen::Vector3f& v3, float w = 1.0f)
{
    return Vector4f(v3.x(), v3.y(), v3.z(), w);
}


static bool insideTriangle(float x, float y, const Vector3f* _v, const int scale = 1)
{   
    float offset = scale / 2;
    //vectro of the pixel center
    Eigen::Vector3f pixel_center;
    pixel_center << (float)x + offset , (float)y + offset, 1.0f;

    //vector for each edge
    Eigen::Vector3f edge1, edge2, edge3;

    //vector for each vertex to the pixel center
    Eigen::Vector3f edgep1, edgep2, edgep3;
    
    edge1 = _v[0] - _v[1];
    edge2 = _v[1] - _v[2];
    edge3 = _v[2] - _v[0]; 
    edgep1 = pixel_center - _v[0];
    edgep2 = pixel_center - _v[1];
    edgep3 = pixel_center - _v[2];

    if(edge1.cross(edgep1)[2] * edge2.cross(edgep2)[2] > 0){
        if(edge2.cross(edgep2)[2] * edge3.cross(edgep3)[2] > 0){
            return true;
        }
    }
    return false;
}

static std::tuple<float, float, float> computeBarycentric2D(float x, float y, const Vector3f* v)
{
    float c1 = (x*(v[1].y() - v[2].y()) + (v[2].x() - v[1].x())*y + v[1].x()*v[2].y() - v[2].x()*v[1].y()) / (v[0].x()*(v[1].y() - v[2].y()) + (v[2].x() - v[1].x())*v[0].y() + v[1].x()*v[2].y() - v[2].x()*v[1].y());
    float c2 = (x*(v[2].y() - v[0].y()) + (v[0].x() - v[2].x())*y + v[2].x()*v[0].y() - v[0].x()*v[2].y()) / (v[1].x()*(v[2].y() - v[0].y()) + (v[0].x() - v[2].x())*v[1].y() + v[2].x()*v[0].y() - v[0].x()*v[2].y());
    float c3 = (x*(v[0].y() - v[1].y()) + (v[1].x() - v[0].x())*y + v[0].x()*v[1].y() - v[1].x()*v[0].y()) / (v[2].x()*(v[0].y() - v[1].y()) + (v[1].x() - v[0].x())*v[2].y() + v[0].x()*v[1].y() - v[1].x()*v[0].y());
    return {c1,c2,c3};
}

void rst::rasterizer::draw(pos_buf_id pos_buffer, ind_buf_id ind_buffer, col_buf_id col_buffer, Primitive type)
{
    auto& buf = pos_buf[pos_buffer.pos_id];
    auto& ind = ind_buf[ind_buffer.ind_id];
    auto& col = col_buf[col_buffer.col_id];

    float f1 = (50 - 0.1) / 2.0;
    float f2 = (50 + 0.1) / 2.0;

    Eigen::Matrix4f mvp = projection * view * model;
    for (auto& i : ind)
    {
        Triangle t;
        Eigen::Vector4f v[] = {
                mvp * to_vec4(buf[i[0]], 1.0f),
                mvp * to_vec4(buf[i[1]], 1.0f),
                mvp * to_vec4(buf[i[2]], 1.0f)
        };
        //Homogeneous division
        for (auto& vec : v) {
            vec /= vec.w();
        }
        //Viewport transformation
        for (auto & vert : v)
        {
            vert.x() = 0.5*width*(vert.x()+1.0);
            vert.y() = 0.5*height*(vert.y()+1.0);
            vert.z() = vert.z() * f1 + f2;
        }

        for (int i = 0; i < 3; ++i)
        {
            t.setVertex(i, v[i].head<3>());
            t.setVertex(i, v[i].head<3>());
            t.setVertex(i, v[i].head<3>());
        }

        auto col_x = col[i[0]];
        auto col_y = col[i[1]];
        auto col_z = col[i[2]];

        t.setColor(0, col_x[0], col_x[1], col_x[2]);
        t.setColor(1, col_y[0], col_y[1], col_y[2]);
        t.setColor(2, col_z[0], col_z[1], col_z[2]);

        rasterize_triangle(t,4);
    }
}

//Screen space rasterization
void rst::rasterizer::rasterize_triangle(const Triangle& t, const int ssr = 1) {
    auto v = t.toVector4();
    std::array<Eigen::Vector3f, 3> v3f = {
        v[0].head<3>(),
        v[1].head<3>(),
        v[2].head<3>()
    };
    Eigen::Vector3f point;
    
    //boudingbox of current triangle
    int min_x = floor(std::min({v[0][0],v[1][0],v[2][0]}));
    int min_y = floor(std::min({v[0][1],v[1][1],v[2][1]})); 
    int max_x = ceil(std::max({v[0][0],v[1][0],v[2][0]}));
    int max_y = ceil(std::max({v[0][1],v[1][1],v[2][1]}));
    Eigen::Vector3f v1 = v3f[0] - v3f[1];
    Eigen::Vector3f v2 = v3f[1] - v3f[2];
    Eigen::Vector3f v3 = v3f[2] - v3f[0];
    
    //Z-buffer
    int width = max_x - min_x;
    int height = max_y - min_y;
    //Use vector because of the z is bound with x and y

    // iterate through the pixel and find if the current pixel is inside the triangle
    for (int x = min_x; x <= max_x; x++){
        for (int y = min_y; y <= max_y; y++)
        {
            
            if(insideTriangle(x, y, t.v)){
                float ssrRatio = 1.0f;
                if(ssr > 1){
                    float count = 0.f;
                    float ssr_scale = 1.0f / ssr;
                    for(int i = 1; i <= ssr; i++){
                        for(int n = 1; n <= ssr; n++){
                            if(insideTriangle(x+ ssr_scale*i,y + ssr_scale * n,t.v,ssr_scale)){
                                count += 1.0f;
                            }
                        }
                    }
                    ssrRatio *= count / (ssr*ssr);
                }
                
                auto[alpha, beta, gamma] = computeBarycentric2D(x, y, t.v);
                float w_reciprocal = 1.0/(alpha / v[0].w() + beta / v[1].w() + gamma / v[2].w());
                float z_interpolated = alpha * v[0].z() / v[0].w() + beta * v[1].z() / v[1].w() + gamma * v[2].z() / v[2].w();
                z_interpolated *= w_reciprocal;
                
                //to the color of the triangle (use getColor function) if it should be painted.
                int index = get_index(x,y);
                if(-z_interpolated < depth_buf[index] ){  // if the currten one is near
                    //set currten z info to depth buff, 
                    //due to the input z is positive and the ture depth info should be negative, so there set as -z to make sure the depth info won`t be mess up
                    depth_buf[index] = -z_interpolated; 
                    Eigen::Vector3f point; 
                    point << x,y,depth_buf[index];
                    Eigen::Vector3f color;
                    if (ssrRatio > 0){
                        set_pixel(point,ssrRatio*t.getColor());
                    }

                }
            }
        }
    }
}

void rst::rasterizer::set_model(const Eigen::Matrix4f& m)
{
    model = m;
}

void rst::rasterizer::set_view(const Eigen::Matrix4f& v)
{
    view = v;
}

void rst::rasterizer::set_projection(const Eigen::Matrix4f& p)
{
    projection = p;
}

void rst::rasterizer::clear(rst::Buffers buff)
{
    if ((buff & rst::Buffers::Color) == rst::Buffers::Color)
    {
        std::fill(frame_buf.begin(), frame_buf.end(), Eigen::Vector3f{0, 0, 0});
    }
    if ((buff & rst::Buffers::Depth) == rst::Buffers::Depth)
    {
        std::fill(depth_buf.begin(), depth_buf.end(), std::numeric_limits<float>::infinity());
    }
}

rst::rasterizer::rasterizer(int w, int h) : width(w), height(h)
{
    frame_buf.resize(w * h);
    depth_buf.resize(w * h);
}

int rst::rasterizer::get_index(int x, int y)
{
    return (height-1-y)*width + x;
}

void rst::rasterizer::set_pixel(const Eigen::Vector3f& point, const Eigen::Vector3f& color)
{
    //old index: auto ind = point.y() + point.x() * width;
    auto ind = (height-1-point.y())*width + point.x();
    frame_buf[ind] = color;

}

// clang-format on