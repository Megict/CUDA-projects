#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <bitset>
#include <chrono>
#include <cmath>

#define silent false 	//no stderr except actual errors
#define verbal true		 
#define visual false		 
#define debug  false	//do printf in kernel

#define gridsize_x 16
#define gridsize_y 16

#define EPS 1e-6

#define PI 3.14159265358979323846
#define MAX_FLOAT 340282346638528859811704183484516925440.0000000000000000

#define CSC(call)  																											\
do {																														\
	cudaError_t err = call;																									\
	if (err != cudaSuccess) {																								\
		std::cerr << "ERROR in " << __FILE__ << ":" << __LINE__ << ". Message: " << cudaGetErrorString(err) << "\n";		\
		exit(0);																											\
	}																														\
} while(0)


//вариант 4. Тетраэдер, октаэдр, додекаэдр
//---------------------------------------------------

struct material {
    uchar4 color;
    float refl_index; //коэфициент отражения 
    float tran_index; //коэфициент прозрачности
    float albedo; //коэфициент диффузного отражения

    material() {}

    material(uchar4 color_, float refl_index_, float tran_index_, float albedo_) {
        color = color_;
        refl_index = refl_index_;
        tran_index = tran_index_;
        albedo = albedo_;
    }
};

struct triangle {
    float3 a;
    float3 b;
    float3 c;
    material mat;
    float3 normal;

    triangle() {}

    triangle(float3 a_, float3 b_, float3 c_) {//для пола - свойства материала не определены
        a = a_;
        b = b_;
        c = c_;
        normal = make_float3(0,0,0);        
    }

    triangle(float3 a_, float3 b_, float3 c_, material mat_) {
        a = a_;
        b = b_;
        c = c_;
        mat = mat_;
        normal = make_float3(0,0,0);        
    }

    triangle(float3 a_, float3 b_, float3 c_, material mat_, float3 normal_) {
        a = a_;
        b = b_;
        c = c_;
        mat = mat_;
        normal = normal_;        
    }
};

struct Light {
    float3 loc;
    float3 clr;
};

//----------------------------- вспомогательные функции
__host__ __device__ float3 add(const float3 &lhs, const float3 &rhs) {
    return make_float3(lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z);
}

__host__ __device__ float3 diff(const float3 &lhs, const float3 &rhs) {
    return make_float3(lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z);
}

__host__ __device__ float dot_prod(const float3 &lhs, const float3 &rhs) {
    return lhs.x * rhs.x + lhs.y * rhs.y + lhs.z * rhs.z;
}

__host__ __device__ float3 vector_prod(const float3 &lhs, const float3 &rhs) {
    return make_float3(lhs.y * rhs.z - lhs.z * rhs.y, lhs.z * rhs.x - lhs.x * rhs.z, lhs.x * rhs.y - lhs.y * rhs.x);
}

__host__ __device__ float3 mult_const(const float3 &lhs, float rhs) {
    return make_float3(lhs.x * rhs, lhs.y * rhs, lhs.z * rhs);
}

__host__ __device__ float3 mult_4(const float3 &a, const float3 &b, const float3 &c, const float3 &d) {
    return make_float3(a.x * d.x + b.x * d.y + c.x * d.z, a.y * d.x + b.y * d.y + c.y * d.z, a.z * d.x + b.z * d.y + c.z * d.z);
}

__host__ __device__ float3 normalize(const float3 &vec) {
    float l = sqrt(dot_prod(vec, vec));
    return make_float3(vec.x / l, vec.y / l, vec.z / l);
}

__host__ __device__ float3 get_normal(const float3 &a, const float3 &b, const float3 &c) {
    return normalize(vector_prod(diff(b, a), diff(c, a)));
}

//-----------------------------
__host__ __device__ int find_intersection(const float3 &pos, const float3 &dir, float &min_dst, triangle* triangles, int triangle_cnt) {
    int clothest_triangle_num = -1;
    for (int i = 0; i < triangle_cnt; i++) {
        float3 l1 = diff(triangles[i].b, triangles[i].a);
        float3 l2 = diff(triangles[i].c, triangles[i].a);

        float3 p_p = vector_prod(dir, l2); 
        double divider = dot_prod(p_p, l1);
        if (fabs(divider) < 1e-10) {
            continue;
        }

        float3 t_p = diff(pos, triangles[i].a);
        double u_p = dot_prod(p_p, t_p) / divider;
        if (u_p < 0.0 || u_p > 1.0) {
            continue;
        }

        float3 q_p = vector_prod(t_p, l1);
        double v_p = dot_prod(q_p, dir) / divider;
        if (v_p < 0.0 || v_p + u_p > 1.0) {
            continue;
        }

        double dst = dot_prod(q_p, l2) / divider;
        if (dst < 0.0) {
            continue;
        }

        if (clothest_triangle_num == -1 || dst < min_dst) {
            clothest_triangle_num = i;
            min_dst = dst;
        }
    }
    return clothest_triangle_num;
}

__host__ __device__ float3 reflect_ray(const float3 &incoming_ray, const float3 &normal) {
    return diff(incoming_ray, mult_const(mult_const(normal, 2), dot_prod(incoming_ray,normal)));
}

__host__ __device__ uchar4 floor_col(uchar4 *floor, float x, float y, int floor_tex_size, int floor_size) {
    x = (x / floor_size*floor_tex_size + floor_tex_size/2);
    y = (y / floor_size*floor_tex_size + floor_tex_size/2);
    return floor[(int)x * floor_tex_size + (int)y];
}

__host__ __device__ uchar4 ray(float3 pos, float3 dir, uchar4 *floor, int floor_size, int floor_texture_size, triangle* triangles, int triangle_num, Light light, int depth, int max_depth) {
    uchar4 reflect_color = make_uchar4(0, 0, 0, 0);
    uchar4 transp_color = make_uchar4(0, 0, 0, 0);

    if (depth > max_depth) {
        return make_uchar4(0, 0, 0, 0);
    }

    float min_d = MAX_FLOAT;

    //ближайший реугольник
    int clothest_tri_num = find_intersection(pos, dir, min_d, triangles, triangle_num);

    if (clothest_tri_num == -1) {
        return make_uchar4(0, 0, 0, 0);
    }

    float3 int_pnt = add(pos, mult_const(dir, min_d));

    material int_mat = triangles[clothest_tri_num].mat;
    uchar4 color = make_uchar4(int_mat.color.x, int_mat.color.y, int_mat.color.z, int_mat.color.w);

    if (clothest_tri_num < 2) { //пересечение с полом
        color = floor_col(floor, int_pnt.x, int_pnt.y, floor_texture_size, floor_size);
    }

    float3 int_normal = triangles[clothest_tri_num].normal;

    //отражения 
    float3 reflect_dir = normalize(reflect_ray(dir, int_normal));
    float3 new_ray_crd_refl = make_float3(0, 0, 0);
    if(dot_prod(reflect_dir, int_normal) < 0) { //с какой стороны от треугольника пускать луч
        new_ray_crd_refl = diff(int_pnt, mult_const(int_normal, EPS));
    }
    else {
        new_ray_crd_refl = add(int_pnt, mult_const(int_normal, EPS));
    }

    reflect_color = ray(new_ray_crd_refl, reflect_dir, floor, floor_size, floor_texture_size, triangles, triangle_num, light, depth + 1, max_depth);
    //------
    
    //прозрачность
    float3 new_ray_crd_transp = make_float3(0, 0, 0);
    if(dot_prod(normalize(dir), int_normal) < 0) { //с какой стороны от треугольника пускать луч
        new_ray_crd_transp = diff(int_pnt, mult_const(int_normal, EPS));
    }
    else {
        new_ray_crd_transp = add(int_pnt, mult_const(int_normal, EPS));
    } 
    transp_color = ray(new_ray_crd_transp, normalize(dir), floor, floor_size, floor_texture_size, triangles, triangle_num, light, depth + 1, max_depth);
    //------

    float3 sum_clr = make_float3(0.0, 0.0, 0.0);
    float ambient_light = 0.1;


    //взаимодейтвие с источником света
    float diffuse_intensity = 0.0;
    
    float3 vec_from_light = normalize(diff(light.loc, int_pnt));

    min_d = MAX_FLOAT;
    int l_min = find_intersection(light.loc, mult_const(vec_from_light, -1), min_d, triangles, triangle_num);

    if (l_min == clothest_tri_num) { //проверка на тень
                                     //смотрим, есть ли что-то на пути к источнику света
        diffuse_intensity = max(0.0, dot_prod(vec_from_light, int_normal));
    }

    sum_clr.x += light.clr.x * color.x*diffuse_intensity*int_mat.albedo;
    sum_clr.y += light.clr.y * color.y*diffuse_intensity*int_mat.albedo;
    sum_clr.z += light.clr.z * color.z*diffuse_intensity*int_mat.albedo;
    
    color = make_uchar4(min(ambient_light*color.x + sum_clr.x + int_mat.refl_index*reflect_color.x + int_mat.tran_index*transp_color.x, 255.0),
                        min(ambient_light*color.y + sum_clr.y + int_mat.refl_index*reflect_color.y + int_mat.tran_index*transp_color.y, 255.0),
                        min(ambient_light*color.z + sum_clr.z + int_mat.refl_index*reflect_color.z + int_mat.tran_index*transp_color.z, 255.0),
                        color.w);
    return color;

}


__global__ void kernel_render_frame(float3 view_pnt, float3 view_dir, int frame_w, int frame_h, double view_angle, uchar4 *result, uchar4 *floor, int floor_size, int floor_texture_size, triangle* triangles, int triangle_num, Light light, int max_depth) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int offset_x = blockDim.x * gridDim.x;
    int offset_y = blockDim.y * gridDim.y;

    float w_part = 2.0 / (frame_w - 1);
    float h_part = 2.0 / (frame_h - 1);

    float angle_part = 1.0 / tan(view_angle * PI / 360.0);

    float3 rdir_z = normalize(diff(view_dir, view_pnt));
    float3 rdir_x = normalize(vector_prod(rdir_z, {0.0, 0.0, 1.0}));
    float3 rdir_y = vector_prod(rdir_x, rdir_z);

    for (int j = idy; j < frame_h; j+= offset_y) {
        for (int i = idx; i < frame_w; i+= offset_x) {
            float3 v = make_float3(-1.f + w_part * i, (-1.f + h_part * j) * frame_h / frame_w, angle_part);
            float3 dir = normalize(mult_4(rdir_x, rdir_y, rdir_z, v));
            result[(frame_h - 1 - j) * frame_w + i] = ray(view_pnt, dir, floor, floor_size, floor_texture_size, triangles, triangle_num, light, 0, max_depth);
        }
    }
}

__global__ void kernel_ssaa(uchar4* data, uchar4* result, int frame_w, int frame_h, int sqrt_rays_per_pix) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int offset_x = blockDim.x * gridDim.x;
    int offset_y = blockDim.y * gridDim.y;

    int w_coef = frame_w * sqrt_rays_per_pix;

    for (int i = idx; i < frame_h; i+= offset_x) {
        for (int j = idy; j < frame_w; j+= offset_y) {

            int y_coef = i * sqrt_rays_per_pix;
            int x_coef = j * sqrt_rays_per_pix;
            uint4 res_col = make_uint4(0, 0, 0, 0);

            for (int l = y_coef; l < y_coef + sqrt_rays_per_pix; l++) {
                for (int k = x_coef; k < x_coef + sqrt_rays_per_pix; k++) {
                    res_col.x += data[l * w_coef + k].x;
                    res_col.y += data[l * w_coef + k].y;
                    res_col.z += data[l * w_coef + k].z;
                }
            }
            float dif = sqrt_rays_per_pix * sqrt_rays_per_pix;
            result[i*frame_w + j] = make_uchar4(res_col.x / dif, res_col.y / dif, res_col.z / dif, res_col.w);
        }
    }
}



void cpu_render_frame(float3 view_pnt, float3 view_dir, int frame_w, int frame_h, double view_angle, uchar4 *result, uchar4 *floor, int floor_size, int floor_texture_size, triangle* triangles, int triangle_num, Light light, int max_depth) {
    float w_part = 2.0 / (frame_w - 1);
    float h_part = 2.0 / (frame_h - 1);

    float angle_part = 1.0 / tan(view_angle * PI / 360.0);

    float3 rdir_z = normalize(diff(view_dir, view_pnt));
    float3 rdir_x = normalize(vector_prod(rdir_z, {0.0, 0.0, 1.0}));
    float3 rdir_y = vector_prod(rdir_x, rdir_z);

    for (int i = 0; i < frame_w; i++) {
        for (int j = 0; j < frame_h; j++) {
            float3 v = {-1.f + w_part * i, (-1.f + h_part * j) * frame_h / frame_w, angle_part};
            float3 dir = normalize(mult_4(rdir_x, rdir_y, rdir_z, v));
            result[(frame_h - 1 - j) * frame_w + i] = ray(view_pnt, dir, floor, floor_size, floor_texture_size, triangles, triangle_num, light, 0, max_depth);
        }
    }
}

void cpu_ssaa(uchar4* data, uchar4* result, int frame_w, int frame_h, int sqrt_rays_per_pix) {

    int w_coef = frame_w * sqrt_rays_per_pix;
    for (int i = 0; i < frame_h; i++) {
        for (int j = 0; j < frame_w; j++) {
            int y_coef = i * sqrt_rays_per_pix;
            int x_coef = j * sqrt_rays_per_pix;
            uint4 res_col = make_uint4(0, 0, 0, 0);

            for (int l = y_coef; l < y_coef + sqrt_rays_per_pix; l++) {
                for (int k = x_coef; k < x_coef + sqrt_rays_per_pix; k++) {
                    res_col.x += data[l * w_coef + k].x;
                    res_col.y += data[l * w_coef + k].y;
                    res_col.z += data[l * w_coef + k].z;
                }
            }
            float dif = sqrt_rays_per_pix * sqrt_rays_per_pix;
            result[i*frame_w + j] = make_uchar4(res_col.x / dif, res_col.y / dif, res_col.z / dif, res_col.w);
        }
    }
}

void fill_norms(triangle* triangles, int from, int cnt) {
    for (int i = from; i < from + cnt; i++) {
        float3 normal = get_normal(triangles[i].a, triangles[i].b, triangles[i].c);
        triangles[i].normal = normal;
    }
}

class Scene {
    bool gpu_available;

    triangle* triangles;
    int triangle_cnt;

    uchar4* floor;
    int floor_w, floor_h;

    Light light;

    int frame_w, frame_h, frame_ang;
    int rays_per_pix, depth_lim;
    uchar4 *data_full;
    uchar4 *data_short; //здесь хранится отрендерендый кадр

    uchar4 *dev_data_full;
    uchar4 *dev_data_short;

    triangle *dev_triangles;
    uchar4 *dev_floor;

    void load_floor(std::string floor_path){
        int w, h;
        FILE *floor_file = fopen(floor_path.c_str(), "rb");

        fread(&w, sizeof(int), 1, floor_file);
        fread(&h, sizeof(int), 1, floor_file);

        floor = (uchar4 *)malloc(sizeof(uchar4) * w * h);

        fread(floor, sizeof(uchar4), w * h, floor_file);
        fclose(floor_file);

        floor_w = w;
        floor_h = h;
    }

    void init_light(float3 location, float3 color) {
        light.loc = location;
        light.clr = color;
    }

    void gpu_init() {
        CSC(cudaMalloc(&dev_data_full, sizeof(uchar4) * frame_w * frame_h * rays_per_pix * rays_per_pix));
        CSC(cudaMalloc(&dev_data_short, sizeof(uchar4) * frame_w * frame_h));
        CSC(cudaMalloc(&dev_triangles, sizeof(triangle) * triangle_cnt));
        CSC(cudaMalloc(&dev_floor, sizeof(uchar4) * floor_w * floor_h));

        CSC(cudaMemcpy(dev_floor, floor, sizeof(uchar4) * floor_w * floor_h, cudaMemcpyHostToDevice));
        CSC(cudaMemcpy(dev_triangles, triangles, sizeof(triangle) * triangle_cnt, cudaMemcpyHostToDevice));
    }

    void sync_to_dev() {
        CSC(cudaMemcpy(dev_data_short, data_short, sizeof(uchar4) * frame_w * frame_h, cudaMemcpyHostToDevice));
    }

    void sync_from_dev() {
        CSC(cudaMemcpy(data_short, dev_data_short, sizeof(uchar4) * frame_w * frame_h, cudaMemcpyDeviceToHost));
    }

    void free_dev_arrays() {
        CSC(cudaFree(dev_triangles));
        CSC(cudaFree(dev_floor));

        CSC(cudaFree(dev_data_short));
        CSC(cudaFree(dev_data_full));
    }

public:    

    Scene(bool gpu_available_, std::string floor_path, float floor_refl_ind, float3 light_location, float3 light_color) {
        //создает сцену с полом
        //первые два треугольника всегда пол
        triangle_cnt = 2;
        triangles = (triangle*)malloc(sizeof(triangle) * triangle_cnt);

        triangles[0] = triangle(make_float3(-5, -5, 0), make_float3(-5, 5, 0), make_float3(5,  5, 0), 
                       material(make_uchar4(255, 255, 255, 0), floor_refl_ind, 0.0, 1), 
                       make_float3(0.0, 0.0, 1.0));

        triangles[1] = triangle(make_float3(-5, -5, 0), make_float3( 5, 5, 0), make_float3(5, -5, 0), 
                       material(make_uchar4(255, 255, 255, 0), floor_refl_ind, 0.0, 1), 
                       make_float3(0.0, 0.0, 1.0));

        load_floor(floor_path);
        init_light(light_location,light_color);

        gpu_available = gpu_available_;
    }

    ~Scene() {
        free(triangles);
        free(floor);

        free(data_full);
        free(data_short);

        if(gpu_available) {
            free_dev_arrays();
        }
    }

    void add_tetraeder(float3 location, float radius, material mat) {
        int prev_cnt = triangle_cnt;
        triangle_cnt += 4;
        triangles = (triangle*)realloc(triangles,sizeof(triangle) * triangle_cnt);

        float3 points[4];
        
        points[0] = make_float3(location.x, location.y, location.z + radius);
        points[1] = make_float3(location.x, location.y + radius*float(cos(PI/6)), location.z - radius*(float(sin(PI/6))));
        points[2] = make_float3(location.x - radius*float(cos(PI/6))*float(cos(PI/6)), location.y - radius*float(sin(PI/6))*float(cos(PI/6)), location.z - radius*(float(sin(PI/6))));
        points[3] = make_float3(location.x + radius*float(cos(PI/6))*float(cos(PI/6)), location.y - radius*float(sin(PI/6))*float(cos(PI/6)), location.z - radius*(float(sin(PI/6))));

        triangles[prev_cnt + 0] = {points[0], points[3], points[1], mat};
        triangles[prev_cnt + 1] = {points[0], points[1], points[2], mat};
        triangles[prev_cnt + 2] = {points[0], points[2], points[3], mat};
        triangles[prev_cnt + 3] = {points[1], points[2], points[3], mat};

        fill_norms(triangles, prev_cnt,4);
    }

    void add_octaeder(float3 location, float radius, material mat) {
        int prev_cnt = triangle_cnt;
        triangle_cnt += 8;
        triangles = (triangle*)realloc(triangles,sizeof(triangle) * triangle_cnt);

        float3 points[6];

        points[0] = make_float3(location.x, location.y - radius, location.z);
        points[1] = make_float3(location.x - radius, location.y, location.z);
        points[2] = make_float3(location.x, location.y, location.z - radius);
        points[3] = make_float3(location.x + radius, location.y, location.z);
        points[4] = make_float3(location.x, location.y, location.z + radius);
        points[5] = make_float3(location.x, location.y + radius, location.z);

        triangles[prev_cnt + 0] = {points[0], points[1], points[2], mat};
        triangles[prev_cnt + 1] = {points[0], points[2], points[3], mat};
        triangles[prev_cnt + 2] = {points[0], points[3], points[4], mat};
        triangles[prev_cnt + 3] = {points[0], points[4], points[1], mat};
        triangles[prev_cnt + 4] = {points[5], points[2], points[1], mat};
        triangles[prev_cnt + 5] = {points[5], points[3], points[2], mat};
        triangles[prev_cnt + 6] = {points[5], points[4], points[3], mat};
        triangles[prev_cnt + 7] = {points[5], points[1], points[4], mat};

        fill_norms(triangles, prev_cnt, 8);
    }

    void add_dodecaeder(float3 location, float radius, material mat) {
        int prev_cnt = triangle_cnt;
        triangle_cnt += 36;
        triangles = (triangle*)realloc(triangles,sizeof(triangle) * triangle_cnt);
        
        float phi = (1.0 + sqrt(5.0)) / 2.0;
        float3 points[20] = {
            make_float3(location.x + (-1.f / phi / sqrt(3.f) * radius), location.y, location.z + (phi / sqrt(3.f) * radius)),
            make_float3(location.x + (1.f / phi / sqrt(3.f) * radius), location.y, location.z + (phi / sqrt(3.f) * radius)),
            make_float3(location.x + (-1.f / sqrt(3.f) * radius), location.y + (1.f / sqrt(3.f) * radius), location.z + (1.f / sqrt(3.f) * radius)),
            make_float3(location.x + (1.f / sqrt(3.f) * radius), location.y + (1.f / sqrt(3.f) * radius), location.z + (1.f / sqrt(3.f) * radius)),
            make_float3(location.x + (1.f / sqrt(3.f) * radius), location.y + (-1.f / sqrt(3.f) * radius), location.z + (1.f / sqrt(3.f) * radius)),
            make_float3(location.x + (-1.f / sqrt(3.f) * radius), location.y + (-1.f / sqrt(3.f) * radius), location.z + (1.f / sqrt(3.f) * radius)),
            make_float3(location.x, location.y + (-phi / sqrt(3.f) * radius), location.z + (1.f / phi / sqrt(3.f) * radius)),
            make_float3(location.x, location.y + (phi / sqrt(3.f) * radius), location.z + (1.f / phi / sqrt(3.f) * radius)),
            make_float3(location.x + (-phi / sqrt(3.f) * radius), location.y + (-1.f / phi / sqrt(3.f) * radius), location.z),
            make_float3(location.x + (-phi / sqrt(3.f) * radius), location.y + (1.f / phi / sqrt(3.f) * radius), location.z),
            make_float3(location.x + (phi / sqrt(3.f) * radius), location.y + (1.f / phi / sqrt(3.f) * radius), location.z),
            make_float3(location.x + (phi / sqrt(3.f) * radius), location.y + (-1.f / phi / sqrt(3.f) * radius), location.z),
            make_float3(location.x, location.y + (-phi / sqrt(3.f) * radius), location.z + (-1.f / phi / sqrt(3.f) * radius)),
            make_float3(location.x, location.y + (phi / sqrt(3.f) * radius), location.z + (-1.f / phi / sqrt(3.f) * radius)),
            make_float3(location.x + (1.f / sqrt(3.f) * radius), location.y + (1.f / sqrt(3.f) * radius), location.z + (-1.f / sqrt(3.f) * radius)),
            make_float3(location.x + (1.f / sqrt(3.f) * radius), location.y + (-1.f / sqrt(3.f) * radius), location.z + (-1.f / sqrt(3.f) * radius)),
            make_float3(location.x + (-1.f / sqrt(3.f) * radius), location.y + (-1.f / sqrt(3.f) * radius), location.z + (-1.f / sqrt(3.f) * radius)),
            make_float3(location.x + (-1.f / sqrt(3.f) * radius), location.y + (1.f / sqrt(3.f) * radius), location.z + (-1.f / sqrt(3.f) * radius)),
            make_float3(location.x + (1.f / phi / sqrt(3.f) * radius), location.y, location.z + (-phi / sqrt(3.f) * radius)),
            make_float3(location.x + (-1.f / phi / sqrt(3.f) * radius), location.y, location.z + (-phi / sqrt(3.f) * radius))};
            
        triangles[prev_cnt + 0] = {points[4],  points[0],  points[6],  mat};
        triangles[prev_cnt + 1] = {points[0],  points[5],  points[6],  mat};
        triangles[prev_cnt + 2] = {points[4],  points[1],  points[0],  mat};

        triangles[prev_cnt + 3] = {points[7],  points[0],  points[3],  mat};
        triangles[prev_cnt + 4] = {points[0],  points[1],  points[3],  mat};
        triangles[prev_cnt + 5] = {points[7],  points[2],  points[0],  mat};

        triangles[prev_cnt + 6] = {points[10], points[1],  points[11], mat};
        triangles[prev_cnt + 7] = {points[1],  points[4],  points[11], mat};
        triangles[prev_cnt + 8] = {points[10], points[3],  points[1],  mat};

        triangles[prev_cnt + 9] = {points[8],  points[0],  points[9],  mat};
        triangles[prev_cnt + 10] = {points[0],  points[2],  points[9],  mat};
        triangles[prev_cnt + 11] = {points[8],  points[5],  points[0],  mat};

        triangles[prev_cnt + 12] = {points[12], points[5],  points[16], mat};
        triangles[prev_cnt + 13] = {points[5],  points[8],  points[16], mat};
        triangles[prev_cnt + 14] = {points[12], points[6],  points[5],  mat};

        triangles[prev_cnt + 15] = {points[15], points[4],  points[12], mat};
        triangles[prev_cnt + 16] = {points[4],  points[6],  points[12], mat};
        triangles[prev_cnt + 17] = {points[15], points[11], points[4],  mat};

        triangles[prev_cnt + 18] = {points[17], points[2],  points[13], mat};
        triangles[prev_cnt + 19] = {points[2],  points[7],  points[13], mat};
        triangles[prev_cnt + 20] = {points[17], points[9],  points[2],  mat};

        triangles[prev_cnt + 21] = {points[13], points[3],  points[14], mat};
        triangles[prev_cnt + 22] = {points[3],  points[10], points[14], mat};
        triangles[prev_cnt + 23] = {points[13], points[7],  points[3],  mat};

        triangles[prev_cnt + 24] = {points[19], points[8],  points[17], mat};
        triangles[prev_cnt + 25] = {points[8],  points[9],  points[17], mat};
        triangles[prev_cnt + 26] = {points[19], points[16], points[8],  mat};

        triangles[prev_cnt + 27] = {points[14], points[11], points[18], mat};
        triangles[prev_cnt + 28] = {points[11], points[15], points[18], mat};
        triangles[prev_cnt + 29] = {points[14], points[10], points[11], mat};

        triangles[prev_cnt + 30] = {points[18], points[12], points[19], mat};
        triangles[prev_cnt + 31] = {points[12], points[16], points[19], mat};
        triangles[prev_cnt + 32] = {points[18], points[15], points[12], mat};
        
        triangles[prev_cnt + 33] = {points[19], points[13], points[18], mat};
        triangles[prev_cnt + 34] = {points[13], points[14], points[18], mat};
        triangles[prev_cnt + 35] = {points[19], points[17], points[13], mat};

        fill_norms(triangles, prev_cnt, 36);
    }

    void set_render_params(int frame_width, int frame_heighth, int frame_angle, int sqrt_rays_per_pix, int max_depth) {
        frame_w = frame_width;
        frame_h = frame_heighth;
        frame_ang = frame_angle;
        rays_per_pix = sqrt_rays_per_pix;
        depth_lim = max_depth;

        data_full = (uchar4*)malloc(sizeof(uchar4) * frame_w * frame_h * rays_per_pix * rays_per_pix);
        data_short = (uchar4*)malloc(sizeof(uchar4) * frame_w * frame_h);


        if(gpu_available) {
            gpu_init();
            sync_to_dev();
        }
    }

    void render_frame_gpu(float3 camera_location, float3 camera_direction) {
        if(!gpu_available) {
            return;
        }

        kernel_render_frame<<<dim3(gridsize_x, gridsize_x), dim3(gridsize_y, gridsize_y)>>>(camera_location, camera_direction, frame_w * rays_per_pix, frame_h * rays_per_pix, 
                                                                                      frame_ang, dev_data_full, dev_floor, 10/*размер пола*/, floor_w, dev_triangles, triangle_cnt, light, depth_lim);
        CSC(cudaGetLastError());

        kernel_ssaa<<<dim3(gridsize_x, gridsize_x), dim3(gridsize_y, gridsize_y)>>>(dev_data_full, dev_data_short, frame_w, frame_h, rays_per_pix);
        CSC(cudaGetLastError());

        sync_from_dev();
    }

    void render_frame_cpu(float3 camera_location, float3 camera_direction) {
        cpu_render_frame(camera_location, camera_direction, frame_w * rays_per_pix, frame_h * rays_per_pix, frame_ang, data_full, floor, 10, floor_w, triangles, triangle_cnt, light, depth_lim);
        cpu_ssaa(data_full, data_short, frame_w, frame_h, rays_per_pix);
    }

    void print_frame_to_file(std::string path_to_file, int frame_id) {
        char buff[256];

        sprintf(buff, path_to_file.c_str(), frame_id);   
        FILE *out = fopen(buff, "wb");
        fwrite(&frame_w, sizeof(int), 1, out);
        fwrite(&frame_h, sizeof(int), 1, out);    
        fwrite(data_short, sizeof(uchar4), frame_w * frame_h, out);
        fclose(out);
    }
};

int main(int argc, char *argv[]) {


    int frame_cnt;
    std::string path_to_result;

    int w, h, view_angle;
    float r_c0, z_c0, phi_c0, A_cr, A_cz, omega_cr, omega_cz, omega_cphi, p_cr, p_cz;
    float r_n0, z_n0, phi_n0, A_nr, A_nz, omega_nr, omega_nz, omega_nphi, p_nr, p_nz;

    float tet_loc_x, tet_loc_y, tet_loc_z, tet_col_r, tet_col_g, tet_col_b, tet_rad, tet_refl_ind, tet_tran_ind;
    float oct_loc_x, oct_loc_y, oct_loc_z, oct_col_r, oct_col_g, oct_col_b, oct_rad, oct_refl_ind, oct_tran_ind;
    float dod_cnt_x, dod_cnt_y, dod_cnt_z, dod_col_r, dod_col_g, dod_col_b, dod_rad, dod_refl_ind, dod_tran_ind;
    
    int lights_on_edge; //не используется
    float fl_pt, fl_cl; //не используется
    
    std::string path_to_floor;
    float floor_refl_ind;

    int lights_num; //не используется
    float light_loc_x, light_loc_y, light_loc_z;
    float light_col_r, light_col_g, light_col_b;

    int max_depth, sqrt_rays_per_pix;

    bool use_gpu = true;

    if (argc > 2) {
        return -1;
    }
    if (argc == 2) {
        std::string arg = argv[1];
        if (arg == "--default") {

            std::cout << "120\n";
            std::cout << "res/%d.data\n";

            std::cout << "1200 900 120\n";
            //параметры камеры
            std::cout << "7.0 3.0 0.0  2.0 1.0\n";
            std::cout << "2.0 6.0 1.0  0.0 0.0\n";

            std::cout << "2.0 0.0 0.0  0.5 0.1\n";
            std::cout << "1.0 4.0 1.0  0.0 0.0\n";
            //параметры тел
            std::cout << "3 1 2 1 0 1 1 0.5 0.2 0\n";
            std::cout << "0 0 1.5 1 0 0 1 0.5 0.9 0\n";
            std::cout << "-1 -3 2 0 1 1 1 0.5 0.3 0\n";
            //параметры пола
            std::cout << "-5 -5 0 -5  5 0 5 5 0 5 -5 0\n";
            std::cout << "floor_2.data\n";
            std::cout << "1 1 1 0\n";
            //параметры источника света
            std::cout << "1 7 -7 5 1 1 1\n";
            //параметры лучей
            std::cout << "2 4\n";
            return 0;

        } 
        if (arg == "--cpu") {
            use_gpu = false;
        }
    }
    
    //---------------------- ввод данных
    std::cin >> frame_cnt;

    std::cin >> path_to_result;
    std::cin >> w >> h >> view_angle;

    std::cin >> r_c0 >> z_c0 >> phi_c0 >> A_cr >> A_cz >> omega_cr >> omega_cz >> omega_cphi >> p_cr >> p_cz;
    std::cin >> r_n0 >> z_n0 >> phi_n0 >> A_nr >> A_nz >> omega_nr >> omega_nz >> omega_nphi >> p_nr >> p_nz;

    std::cin >> tet_loc_x >> tet_loc_y >> tet_loc_z >> tet_col_r >> tet_col_g >> tet_col_b >> tet_rad >> tet_refl_ind >> tet_tran_ind >> lights_on_edge;
    std::cin >> oct_loc_x >> oct_loc_y >> oct_loc_z >> oct_col_r >> oct_col_g >> oct_col_b >> oct_rad >> oct_refl_ind >> oct_tran_ind >> lights_on_edge;
    std::cin >> dod_cnt_x >> dod_cnt_y >> dod_cnt_z >> dod_col_r >> dod_col_g >> dod_col_b >> dod_rad >> dod_refl_ind >> dod_tran_ind >> lights_on_edge;

    std::cin >> fl_pt >> fl_pt >> fl_pt >> fl_pt >> fl_pt >> fl_pt >> fl_pt >> fl_pt >> fl_pt >> fl_pt >> fl_pt >> fl_pt;
    std::cin >> path_to_floor;
    std::cin >> fl_cl >> fl_cl >> fl_cl >> floor_refl_ind;

    std::cin >> lights_num;
    std::cin >> light_loc_x >> light_loc_y >> light_loc_z;
    std::cin >> light_col_r >> light_col_g >> light_col_b;

    std::cin >> max_depth >> sqrt_rays_per_pix;
    //----------------------

    float3 view_pnt, view_dir;

    Scene scene(use_gpu ? true : false, path_to_floor, floor_refl_ind, make_float3(light_loc_x, light_loc_y, light_loc_z), make_float3(light_col_r, light_col_g, light_col_b));
    scene.add_tetraeder(make_float3(tet_loc_x, tet_loc_y, tet_loc_z), tet_rad, material(make_uchar4(tet_col_r * 255, tet_col_g * 255, tet_col_b * 255, 0), tet_refl_ind, tet_tran_ind, 1));
    scene.add_octaeder(make_float3(oct_loc_x, oct_loc_y, oct_loc_z), oct_rad, material(make_uchar4(oct_col_r * 255, oct_col_g * 255, oct_col_b * 255, 0), oct_refl_ind, oct_tran_ind, 1));
    scene.add_dodecaeder(make_float3(dod_cnt_x, dod_cnt_y, dod_cnt_z), dod_rad, material(make_uchar4(dod_col_r * 255, dod_col_g * 255, dod_col_b * 255, 0), dod_refl_ind, dod_tran_ind, 1));

    scene.set_render_params(w,h,view_angle,sqrt_rays_per_pix,max_depth);
    

    cudaEvent_t begin, end;
    cudaEventCreate(&begin);
    cudaEventCreate(&end);
    float gpu_time;
    float cpu_time;

    float gpu_sum = 0;
    float cpu_sum = 0;


    cudaEvent_t ovl_begin, ovl_end;
    cudaEventCreate(&ovl_begin);
    cudaEventCreate(&ovl_end);

    cudaEventRecord(ovl_begin);
    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
    for (int i = 0; i < frame_cnt; ++i) {
        //------------------ расчет положения и направления камеры
        float step = 2 * PI * i / frame_cnt; 

        float r_ct = r_c0 + A_cr * sin(omega_cr * step + p_cr);
        float z_ct = z_c0 + A_cz * sin(omega_cz * step + p_cz);
        float phi_ct = phi_c0 + omega_cphi * step;

        float r_nt = r_n0 + A_nr * sin(omega_nr * step + p_nr);
        float z_nt = z_n0 + A_nz * sin(omega_nz * step + p_nz);
        float phi_nt = phi_n0 + omega_nphi * step;

        view_pnt = make_float3(r_ct * cos(phi_ct), r_ct * sin(phi_ct), z_ct);
        view_dir = make_float3(r_nt * cos(phi_nt), r_nt * sin(phi_nt), z_nt);
        //------------------

        if(use_gpu) {
            cudaEventRecord(begin);
            scene.render_frame_gpu(view_pnt,view_dir);
            cudaEventRecord(end);

            cudaEventSynchronize(end);
            cudaEventElapsedTime(&gpu_time, begin, end);
            std::cout << "frame: " << i << "\t gpu time: " << gpu_time << "\t rays_emited: " << w*h*sqrt_rays_per_pix*sqrt_rays_per_pix << "\n";
            //std::cout << i << "\t" << gpu_time << "\t" << w*h*sqrt_rays_per_pix*sqrt_rays_per_pix << "\n";
            gpu_sum += gpu_time;
        }
        else {
            std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

            scene.render_frame_cpu(view_pnt,view_dir);
            
            std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
            cpu_time = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() / 1000;

            std::cout << "frame: " << i << "\t cpu time: " << cpu_time << "\t rays_emited: " << w*h*sqrt_rays_per_pix*sqrt_rays_per_pix << "\n";
            //std::cout << i << "\t" << cpu_time << "\t" << w*h*sqrt_rays_per_pix*sqrt_rays_per_pix << "\n";
            cpu_sum += cpu_time;
        }


        scene.print_frame_to_file(path_to_result,i);
    }
    cudaEventRecord(ovl_end);
    cudaEventSynchronize(ovl_end);
    cudaEventElapsedTime(&gpu_time, ovl_begin, ovl_end);

    std::chrono::steady_clock::time_point stop = std::chrono::steady_clock::now();
    cpu_time = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count() / 1000;

    std::cout << "-----------------------------\n";
    std::cout << "overall       | cpu time: " << cpu_time << "ms\t gpu time: " << gpu_time << "ms\n";
    std::cout << "summary       | cpu time: " << cpu_sum << "ms\t gpu time: " << gpu_sum << "ms\n";
    std::cout << "avg per frame | cpu time: " << cpu_sum / frame_cnt << "ms\t gpu time: " << gpu_sum / frame_cnt << "ms\n";
    
    return 0;
}