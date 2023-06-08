#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <bitset>

#define silent false 	//no stderr except actual errors
#define verbal false	//print every component of every pixel before and after applying classification
#define visual false	//print avg of each pixel components in grid with img sides
#define debug  false	//do printf in kernel

#define CSC(call)  																											\
do {																														\
	cudaError_t err = call;																									\
	if (err != cudaSuccess) {																								\
		std::cerr << "ERROR in " << __FILE__ << ":" << __LINE__ << ". Message: " << cudaGetErrorString(err) << "\n";		\
		exit(0);																											\
	}																														\
} while(0)

//вариант 2
//Метод расстояния Махаланобиса.

struct pixel{ //обертка над uchar4, позволяющая читать/писать в файл
	uchar4 pix;

	pixel() {
		pix = make_uchar4(0,0,0,0);
	}

	pixel(char red,char green, char blue, char alpha) {
		pix = make_uchar4((unsigned char)(red),
						  (unsigned char)(green),
						  (unsigned char)(blue),
						  (unsigned char)(alpha));
	}

	pixel read_from_file(std::ifstream& in) {
		char tmp;
		in.get(tmp);
		pix.x = (unsigned char)(tmp);
		in.get(tmp);
		pix.y = (unsigned char)(tmp);
		in.get(tmp);
		pix.z = (unsigned char)(tmp);
		in.get(tmp);
		pix.w = (unsigned char)(tmp);
		return *this;
	}

	pixel print_to_file(std::ofstream& out) {
		out.put(char(pix.x));
		out.put(char(pix.y));
		out.put(char(pix.z));
		out.put(char(pix.w));
		return *this;
	}

	void print() {
		std::cerr << "| " << +pix.x << " " << +pix.y << " " << +pix.z << " |" << +pix.w << "|\n";
	}

	void print_avg() {
		int avg = ((unsigned char)(pix.x) + (unsigned char)(pix.y) + (unsigned char)(pix.z))/3;
		
		if(avg < 100) {
			std::cerr << " ";
		}
		if(avg < 10) {
			std::cerr << " ";
		}

		std::cerr << avg;
	}
};


__constant__ int CL_NUM;
__constant__ float3 AVG[32];
__constant__ float3 COV[96];

__global__ void kernel_classify(cudaTextureObject_t pix, int size_w, int size_h, pixel* res) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x; 
	int offset_x = blockDim.x * gridDim.x; 

	if(debug && idx == 0) { 
		printf("KERNEL: %d | %d\n",CL_NUM,CL_NUM);
		printf("KERNEL: AVG pix\n");
		for (int i =0 ; i < CL_NUM; ++i) {
			printf("KERNEL: %f %f %f\n",(AVG[i].x), (AVG[i].y), (AVG[i].z));
		}
		printf("KERNEL: inv COV matrs\n");
		for (int i =0 ; i < CL_NUM*3; ++i) {
			if(i %3 == 0) { 
				printf("\n");
			}
			printf("KERNEL: %f %f %f\n",(COV[i].x), (COV[i].y), (COV[i].z));
		}
		printf("\n");
		printf("%d \n\n",size_w);
	}

	while(idx < size_w * size_h) { 
		int cur_x = (idx/size_w);
		int cur_y = (idx%size_w);



		uchar4 cur_pix = tex2D<uchar4>(pix,cur_x,cur_y);/*первая - высотная координата, вторая - строчная*/
		
		if(debug) printf("KERNEL: %d %d | : %d %d %d %d\n",cur_x,cur_y,cur_pix.x,cur_pix.y,cur_pix.z,cur_pix.w);

		int min_dist = INT_MAX;
		unsigned char cl = 0;
		for (int i =0; i < CL_NUM; ++i) {
			float d_x = (float)cur_pix.x - AVG[i].x;
			float d_y = (float)cur_pix.y - AVG[i].y;
			float d_z = (float)cur_pix.z - AVG[i].z;

			float calculated_dist = d_x*d_x*COV[3*i].x + d_x*d_y*COV[3*i + 1].x + d_x*d_z*COV[3*i + 2].x +
								  d_y*d_x*COV[3*i].y + d_y*d_y*COV[3*i + 1].y + d_y*d_z*COV[3*i + 2].y +
								  d_z*d_x*COV[3*i].z + d_z*d_y*COV[3*i + 1].z + d_z*d_z*COV[3*i + 2].z;

			if(calculated_dist < min_dist) { 
				cl = (unsigned char)i;
				min_dist = calculated_dist;
			}
			if(debug) printf("KERNEL: %d %d | cd for %d = %f\n",cur_x,cur_y,i,calculated_dist);
		}
		if(debug) printf("KERNEL: %d %d |%d -- \n",cur_x,cur_y,cl);

		res[cur_y * (size_h) + cur_x].pix.x  = (cur_pix.x);
		res[cur_y * (size_h) + cur_x].pix.y  = (cur_pix.y);
		res[cur_y * (size_h) + cur_x].pix.z  = (cur_pix.z);
		res[cur_y * (size_h) + cur_x].pix.w  = cl;

		idx += offset_x;
	}
}



__global__ void kernel_ssaa(cudaTextureObject_t pix,int size_w, int size_h , int coef_w, int coef_h, pixel* res) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x; 
	int idy = blockIdx.y * blockDim.y + threadIdx.y; 
	int offset_x = blockDim.x * gridDim.x; 
	int offset_y = blockDim.y * gridDim.y; 

	int idy_start = idy;
	if(debug) printf("coefs  %d <-w|h-> %d\n\n",coef_w,coef_h);
	if(debug) printf("sizes  %d <-w|h-> %d\n\n",size_w,size_h);

	while(coef_h*idx < size_h) {
		while(coef_w*idy < size_w) {

			int cur_x = coef_h*idx ;
			int cur_y = coef_w*idy ;

			uchar4 cur_pix = tex2D<uchar4>(pix,cur_x,cur_y);/*первая - высотная координата, вторая - строчная*/
			if(debug) printf("(%d %d) %d ",cur_x,cur_y,(cur_pix.x + cur_pix.y + cur_pix.z)/3);
			
			unsigned int r = 0,g = 0,b = 0,a = 0;
			if(debug) printf("(%d %d) %d %d | %d %d %d | %d\n",idx,idy,cur_x,cur_y,cur_pix.x,cur_pix.y,cur_pix.z,cur_pix.w);

			if(debug) printf("filter shape: %d <-h|w-> %d \n",coef_h,coef_w);
			for(int i = 0; i < coef_h; ++i) {
				for(int j = 0; j < coef_w; ++j) {
					uchar4 add_pix = tex2D<uchar4>(pix,cur_x + i,cur_y + j);
					r += (unsigned int) add_pix.x;
					g += (unsigned int) add_pix.y;
					b += (unsigned int) add_pix.z;
					a += (unsigned int) add_pix.w;
					if(debug) printf("(%d %d) aded from %d %d |%d %d %d|\n",cur_x,cur_y,cur_x + i,cur_y + j,add_pix.x,add_pix.y,add_pix.z);
				}
			}

			if(debug) printf("recorded from %d %d (%d %d) to res at (%d) put |%u %u %u| / |%d|\n",idx,idy,cur_x,cur_y,idy * (size_h/coef_h) + idx,r,g,b,(coef_w * coef_h));
			res[idy * (size_h/coef_h) + idx].pix.x  = (unsigned char) (r / (coef_w * coef_h));
			res[idy * (size_h/coef_h) + idx].pix.y  = (unsigned char) (g / (coef_w * coef_h));
			res[idy * (size_h/coef_h) + idx].pix.z  = (unsigned char) (b / (coef_w * coef_h));
			res[idy * (size_h/coef_h) + idx].pix.w  = (unsigned char) (a / (coef_w * coef_h));
			
			idy += offset_y;
		}

		if(debug) printf("\n");
		idx += offset_x;
		idy = idy_start;
	}
}



float det3x3(float* matr) { //только матрицы 3х3 в линейной записи
	return matr[0]*matr[4]*matr[8] + matr[1]*matr[5]*matr[6] + matr[2]*matr[3]*matr[7] -
		   matr[2]*matr[4]*matr[6] - matr[0]*matr[5]*matr[7] - matr[1]*matr[3]*matr[8];
}

float det2x2(float matr_0, float matr_1, float matr_2, float matr_3) { //только матрицы 2х2 в линейной записи
	return matr_0*matr_3 - matr_1*matr_2;
}


float3* inverse(float3* matr) { //только матрицы 3х3 в линейной записи || работает верно
	float* matr_by_char = (float*)malloc(9*sizeof(float));
	matr_by_char[0] = matr[0].x; matr_by_char[1] = matr[0].y; matr_by_char[2] = matr[0].z; 
	matr_by_char[3] = matr[1].x; matr_by_char[4] = matr[1].y; matr_by_char[5] = matr[1].z; 
	matr_by_char[6] = matr[2].x; matr_by_char[7] = matr[2].y; matr_by_char[8] = matr[2].z; 

	float det_main = det3x3(matr_by_char);

	float* matr_minore = (float*)malloc(9*sizeof(float));
	matr_minore[0] = (float)( det2x2(matr_by_char[4],matr_by_char[5],matr_by_char[7],matr_by_char[8])/det_main);
	matr_minore[3] = (float)(-det2x2(matr_by_char[3],matr_by_char[5],matr_by_char[6],matr_by_char[8])/det_main);
	matr_minore[6] = (float)( det2x2(matr_by_char[3],matr_by_char[4],matr_by_char[6],matr_by_char[7])/det_main);

	matr_minore[1] = (float)(-det2x2(matr_by_char[1],matr_by_char[2],matr_by_char[7],matr_by_char[8])/det_main);
	matr_minore[4] = (float)( det2x2(matr_by_char[0],matr_by_char[2],matr_by_char[6],matr_by_char[8])/det_main);
	matr_minore[7] = (float)(-det2x2(matr_by_char[0],matr_by_char[1],matr_by_char[6],matr_by_char[7])/det_main);

	matr_minore[2] = (float)( det2x2(matr_by_char[1],matr_by_char[2],matr_by_char[4],matr_by_char[5])/det_main);
	matr_minore[5] = (float)(-det2x2(matr_by_char[0],matr_by_char[2],matr_by_char[3],matr_by_char[5])/det_main);
	matr_minore[8] = (float)( det2x2(matr_by_char[0],matr_by_char[1],matr_by_char[3],matr_by_char[4])/det_main);

	matr[0].x = matr_minore[0];
	matr[0].y = matr_minore[1];
	matr[0].z = matr_minore[2];

	matr[1].x = matr_minore[3];
	matr[1].y = matr_minore[4];
	matr[1].z = matr_minore[5];

	matr[2].x = matr_minore[6];
	matr[2].y = matr_minore[7];
	matr[2].z = matr_minore[8];

	return matr;
}

float3* avg(std::vector<std::vector<pixel>> classes) { 
	float3* res = (float3*)calloc(classes.size(),sizeof(float3));
	for (int i = 0; i < classes.size(); ++i) {
		for (int j = 0; j < classes[i].size(); ++j){
			res[i].x += (float)classes[i][j].pix.x;
			res[i].y += (float)classes[i][j].pix.y;
			res[i].z += (float)classes[i][j].pix.z;
		}

		res[i].x /= (float)classes[i].size();
		res[i].y /= (float)classes[i].size();
		res[i].z /= (float)classes[i].size();
	}
	return res;
}

float3* cov(std::vector<std::vector<pixel>> classes) { //заполняем массив *обратными* матрицами ковариации
	float3* res = (float3*)calloc(classes.size()*3, sizeof(float3));
	float3* avg_ = avg(classes);

	for (int i = 0; i < classes.size(); ++i) { 
		if(debug) printf("COV: avg of cur class:\n");
		if(debug) printf("COV: %f %f %f\n",avg_[i].x,avg_[i].y,avg_[i].z);
		if(debug) printf("COV: ------\n");
		for (int j = 0; j < classes[i].size(); ++j){
			float3 dif = make_float3((float)classes[i][j].pix.x - avg_[i].x, (float)classes[i][j].pix.y - avg_[i].y, (float)classes[i][j].pix.z - avg_[i].z);
			if(debug) printf("COV: dif:\n %f %f %f\n ",dif.x,dif.y,dif.z);
			if(debug) printf("COV: for pix:\n  ");
			if(debug) classes[i][j].print();

			res[3*i].x += dif.x * dif.x;
			res[3*i].y += dif.x * dif.y;
			res[3*i].z += dif.x * dif.z;

			res[3*i + 1].x += dif.y * dif.x;
			res[3*i + 1].y += dif.y * dif.y;
			res[3*i + 1].z += dif.y * dif.z;

			res[3*i + 2].x += dif.z * dif.x;
			res[3*i + 2].y += dif.z * dif.y;
			res[3*i + 2].z += dif.z * dif.z;
		}
		if(debug) printf("COV: cov: %d\n",i);


		res[3*i].x /= (float)(classes[i].size() - 1);
		res[3*i].y /= (float)(classes[i].size() - 1);
		res[3*i].z /= (float)(classes[i].size() - 1);

		res[3*i + 1].x /= (float)(classes[i].size() - 1);
		res[3*i + 1].y /= (float)(classes[i].size() - 1);
		res[3*i + 1].z /= (float)(classes[i].size() - 1);

		res[3*i + 2].x /= (float)(classes[i].size() - 1);
		res[3*i + 2].y /= (float)(classes[i].size() - 1);
		res[3*i + 2].z /= (float)(classes[i].size() - 1);

		if(debug) printf("\nCOV:  matrix:\n");
		if(debug) printf("COV: %f %f %f\n",res[3*i].x,res[3*i].y,res[3*i].z);
		if(debug) printf("COV: %f %f %f\n",res[3*i+1].x,res[3*i+1].y,res[3*i+1].z);
		if(debug) printf("COV: %f %f %f\n",res[3*i+2].x,res[3*i+2].y,res[3*i+2].z);



		float3* matr = (float3*)malloc(3*sizeof(float3));
		matr[0] = res[3*i];
		matr[1] = res[3*i + 1];
		matr[2] = res[3*i + 2];

		float3* real_matr = inverse(matr);
		res[3*i] = real_matr[0];
		res[3*i + 1] = real_matr[1];
		res[3*i + 2] = real_matr[2];
		free(matr);

		if(debug) printf("\nCOV:  after inv:\n");
		if(debug) printf("COV: %f %f %f\n",res[3*i].x,res[3*i].y,res[3*i].z);
		if(debug) printf("COV: %f %f %f\n",res[3*i+1].x,res[3*i+1].y,res[3*i+1].z);
		if(debug) printf("COV: %f %f %f\n",res[3*i+2].x,res[3*i+2].y,res[3*i+2].z);
	}

	return res;
}



class image{
	int w;
	int h;
	pixel* pixels;
	
	public:

	image() {
		w = 0; h = 0;
		pixels = (pixel*)malloc(sizeof(pixel)*w*h);
	}

	image(int w_,int h_, pixel* array) {
		w = w_; h = h_;
		pixels = array;
	}

	image(std::string in_adress) { //счиатать изображение из файла
		std::ifstream fin(in_adress,std::ios::binary);

		if(fin.fail()) {
			throw 101;
		}

		fin.seekg(0); 
		for (int i =0; i < 10; ++i){

		}

		fin.seekg(0); 
		fin.get((char*)&w,5);
		fin.get((char*)&h,5);

		std::bitset<8*4> x1(w);
		std::bitset<8*4> x2(h);
		std::cerr << x1 << "\n" << x2 << "\n";
		std::cerr << "1234567812345678123456781234567812345678\n";

		//00000000 01100110 10101101 00000000
		//00000000 00000000 00000000 00000000

		//return;


		pixels = (pixel*)malloc(sizeof(pixel)*w*h);

		for(int i = 0; i < h; ++i){
			for(int j = 0;j < w; ++j){
				struct pixel cur_pixel = pixel().read_from_file(fin);
				pixels[j*h + i] = cur_pixel;
			}
		}

		fin.close();		
	}

	~image() {
		free(pixels);
	}

	pixel* Pixels() {
		return pixels;
	}

	std::pair<int,int> Size() {
		return std::pair<int,int> (w,h);
	}

	void print_visual() {
		for(int i = 0; i < h; ++i) {
			for(int j = 0; j < w; ++j) {
				pixels[j*h + i].print_avg();
				std::cerr << " ";
			}
			std::cerr << "\n";
		}
	}

	void print() {
		std::cerr << "size of img: " << w << " " << h << "\n";

		for(int i=0;i<w;++i) {
			for(int j=0;j<h;++j) {
				pixels[i*h + j].print();
			}
			std::cerr << "\n";
		}
	}

	void print_non_zero() {
		std::cerr << "size of img: " << w << " " << h << "\n";

		for(int i=0;i<w;++i) {
			for(int j=0;j<h;++j) {
				if(pixels[i*h + j].pix.x != 0 && pixels[i*h + j].pix.y != 0 && pixels[i*h + j].pix.z != 0){
					pixels[i*h + j].print();
				}
			}
		}
	}

	void print_size() {
		std::cerr << "size of img: " << w << " " << h << "\n";
	}

	void print_to_file(std::string filename) { //записать изображение в файл
		std::ofstream fout(filename,std::ios::binary);

		fout.write((char*)&w,4);
		fout.write((char*)&h,4);

		for(int i=0;i< h;++i) {
			for(int j=0;j< w;++j) {
				pixels[j*h + i].print_to_file(fout);
			}
		}

		fout.close();
	}

	int SSAA(int new_w, int new_h) {
		cudaEvent_t start, stop;
		CSC(cudaEventCreate(&start));
		CSC(cudaEventCreate(&stop));
		int coef_w = w / new_w,
			coef_h = h / new_h;

		if(!silent)	std::cerr << "SSAA: current size: " << w << " " << h << " required size: " << new_w << " " << new_h << "\n";
		
		pixel* res_pix; //массив для записи результата на устройстве
		pixel* res = (pixel*)malloc(new_w*new_h*sizeof(pixel)); //массив для принятия результата на хосте 

		CSC(cudaMalloc (&res_pix, sizeof(pixel)*new_w*new_h));

		cudaArray* tex_array; //создание массива в текстурной памяти
		cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<uchar4>();
		CSC(cudaMallocArray(&tex_array,&channelDesc, h,w));

		CSC(cudaMemcpy2DToArray(tex_array, 0, 0, pixels, sizeof(uchar4)*h,sizeof(uchar4)*h,w, cudaMemcpyHostToDevice));

		cudaResourceDesc resD; //дескриптор ресурса
		memset(&resD, 0, sizeof(resD));
		resD.resType = cudaResourceTypeArray; // тип контейнера
		resD.res.array.array = tex_array; //указатель на контейнер

		cudaTextureDesc texD;//дескриптор текстуры
		memset(&texD, 0, sizeof(texD));
		texD.readMode = cudaReadModeElementType; //приводить ли int к float
		
		cudaTextureObject_t texture; //создание текстурного объекта
		CSC(cudaCreateTextureObject(&texture, &resD, &texD, NULL));

		dim3 dimBlock(32,32); //двухмерная сетка потоков
		dim3 dimGrid(32,32);

		if(!silent)	std::cerr << "SSAA: preparations - done\n";
		CSC(cudaEventRecord(start));
		kernel_ssaa<<<dimBlock, dimGrid>>>(texture,w,h,coef_w,coef_h,res_pix);
		CSC(cudaEventRecord(stop));
		CSC(cudaEventSynchronize(stop));

		CSC(cudaMemcpy(res, res_pix, sizeof(pixel)*new_w*new_h, cudaMemcpyDeviceToHost));
		if(!silent)	std::cerr << "SSAA: kernel - done\n";

		free(pixels);
		
		w = new_w;
		h = new_h;
		pixels = res;

		CSC(cudaDestroyTextureObject(texture));
		CSC(cudaFreeArray(tex_array));
		CSC(cudaFree(res_pix));

		float milliseconds = 0;
		CSC(cudaEventElapsedTime(&milliseconds, start, stop));
		if(!silent) std::cerr << "SSAA: kernel time = " << milliseconds << "\n";

		return 0;
	}

	void Classify(std::vector<std::vector<int>> class_coords) {
		std::vector<std::vector<pixel>> class_pixels;
		if(!silent)	std::cerr << "CLSS: started\n";

		if(w <= 0 || h <= 0) {
			printf("CLSS: incorrect size\n");
			return;
		}
		
		//заменить координаты на цвета
		for(int i=0;i < class_coords.size();++i) { 
			std::vector<pixel> cur_class_pixels;
			for(int j = 0; j < class_coords[i].size();j += 2) {
				
				cur_class_pixels.push_back(pixels[class_coords[i][j]*h + class_coords[i][j + 1]]);
			}
			class_pixels.push_back(cur_class_pixels);
		}
		//прогнать через avg и cov

		float3* avg_ = avg(class_pixels);
		float3* cov_ = cov(class_pixels);

		if(!silent)	std::cerr << "CLSS: copying to constant - began\n";
		int avg_size = class_pixels.size();		
		CSC(cudaMemcpyToSymbol( CL_NUM,&avg_size,sizeof(int)));

		CSC(cudaMemcpyToSymbol( AVG, avg_, sizeof(float3)*class_pixels.size()));
		CSC(cudaMemcpyToSymbol( COV, cov_, sizeof(float3)*3*class_pixels.size()));

		if(!silent)	std::cerr << "CLSS: preparations - began\n";
		//--------------для работы с текстурами и приема результата--------------------------------------------------
		pixel* res_pix; //массив для записи результата на устройстве
		pixel* res = (pixel*)malloc(w*h*sizeof(pixel)); //массив для принятия результата на хосте 

		CSC(cudaMalloc (&res_pix, sizeof(pixel)*w*h));

		cudaArray* tex_array; //создание массива в текстурной памяти
		cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<uchar4>();
		CSC(cudaMallocArray(&tex_array,&channelDesc, h,w));

		CSC(cudaMemcpy2DToArray(tex_array, 0, 0, pixels, sizeof(uchar4)*h,sizeof(uchar4)*h,w, cudaMemcpyHostToDevice));

		cudaResourceDesc resD; //дескриптор ресурса
		memset(&resD, 0, sizeof(resD));
		resD.resType = cudaResourceTypeArray; // тип контейнера
		resD.res.array.array = tex_array; //указатель на контейнер

		cudaTextureDesc texD;//дескриптор текстуры
		memset(&texD, 0, sizeof(texD));
		texD.readMode = cudaReadModeElementType; //приводить ли int к float
		
		cudaTextureObject_t texture; //создание текстурного объекта
		CSC(cudaCreateTextureObject(&texture, &resD, &texD, NULL));
		//----------------------------------------------------------------------------------------------------

		if(!silent)	std::cerr << "CLSS: preparations - done\n";
		kernel_classify<<<32,32>>> (texture,w,h,res_pix);
		CSC(cudaMemcpy(res, res_pix, sizeof(pixel)*w*h, cudaMemcpyDeviceToHost));
		if(!silent)	std::cerr << "CLSS: kernel - done\n";

		free(pixels);

		pixels = res;

		CSC(cudaDestroyTextureObject(texture));
		CSC(cudaFreeArray(tex_array));
		CSC(cudaFree(res_pix));
	}

};

int main() {

	std::string img_name, res_name;
	std::cin >> img_name;

	try{ 
		int w,h;
		int max_elm_per_class;
		std::cin >> w >> h;
		std::cin >> max_elm_per_class;
		
		pixel* pixels = (pixel*)malloc(sizeof(pixel)*w*h);
		for(int i = 0; i < h; ++i){
			for(int j = 0;j < w; ++j){
				struct pixel cur_pixel = pixel(std::rand()%255,std::rand()%255,std::rand()%255,0);
				pixels[j*h + i] = cur_pixel;
			}
		}

		int class_cnt = std::rand()%32;
		std::cout << img_name << " \n";
		std::cout << "res\n";
		std::cout << class_cnt << "\n";
		
		for(int i = 0; i < class_cnt; ++i) {
			int cur_class_cnt = std::rand()%max_elm_per_class + 1;
			std::cout << cur_class_cnt << " ";
			for(int j = 0; j < cur_class_cnt; ++j) {
				int cur_coord_x = std::rand()%w;
				int cur_coord_y = std::rand()%h;
				std::cout << cur_coord_x << " " << cur_coord_y << " ";
			}
			std::cout << "\n";
		}

		image img = image(w,h,pixels);
		img.print_to_file(img_name);

		return 0;


		//w = img.Size().first;
		//h = img.Size().second;

		//заполнение таблички с координатами пикселей классов
		int num_classes;
		std::cin >> num_classes;
		std::vector<std::vector<int>> classes_info;
		for (int i = 0; i < num_classes; ++i) { 
			int num_exmpls;
			std::cin >> num_exmpls;
			std::vector<int> cur_class;
			for(int j = 0; j < num_exmpls*2; ++j) {
				int cur_exmpl;
				std::cin >> cur_exmpl;
				cur_class.push_back(cur_exmpl);
			}
			classes_info.push_back(cur_class);
		}
		if(!silent) {
			std::cerr << "--got class info--\n";
			/*for (int i=0;i<num_classes; ++i) {
				std::cerr << classes_info[i].size() << ": ";
				for(int j = 0; j < classes_info[i].size(); ++j) {
					std::cerr << classes_info[i][j] << " ";
				}
				std::cerr << "\n";
			}*/
			//std::cerr << "--        --\n";
		}

		if(!silent) {
			//std::cerr << "Before classification:\n";
			//std::cerr << "--------------------------\n";
			if(verbal) {
				img.print();
			}
			else
			if(visual) {
				img.print_visual();
			}
			else {
				img.print_size();
			}
			std::cerr << "--------------------------\n";
		}

		img.Classify(classes_info);
		img.print_to_file(res_name);

		if(!silent) {
			std::cerr << "After classification:\n";
			std::cerr << "--------------------------\n";
			if(verbal) {
				img.print();
			}
			else
			if(visual) {
				img.print_visual();
			}
			else {
				img.print_size();
			}
			std::cerr << "--------------------------\n";
		}

	}
	catch(int err) {
		if (err == 101) {
			std::cerr << "error opening file\n";
		} else
		if (err == 105){
			std::cerr << "error new length\n";
		}
		return 0;
	}

	return 0;
}


