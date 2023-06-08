#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <bitset>

#define silent false 	//no stderr except actual errors
#define verbal false	//print every component of every pixel before and after applying classification
#define visual false	//print avg of each pixel components in grid with img sides
#define debug  false	//do printf in kernel

const unsigned long long MAX_TEX_SIZE = 100000000;

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
__constant__ double3 AVG[32];
__constant__ double3 COV[96];

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

		double min_dist;
		unsigned char cl = 0;
		for (int i =0; i < CL_NUM; ++i) {
			//вычисление расстояния
			double d_x = (double)cur_pix.x - AVG[i].x;
			double d_y = (double)cur_pix.y - AVG[i].y;
			double d_z = (double)cur_pix.z - AVG[i].z;
			double calculated_dist = d_x*d_x*COV[3*i].x + d_x*d_y*COV[3*i + 1].x + d_x*d_z*COV[3*i + 2].x +
								     d_y*d_x*COV[3*i].y + d_y*d_y*COV[3*i + 1].y + d_y*d_z*COV[3*i + 2].y +
								     d_z*d_x*COV[3*i].z + d_z*d_y*COV[3*i + 1].z + d_z*d_z*COV[3*i + 2].z;
									 
			if(i == 0) {
				min_dist = calculated_dist;
				cl = (unsigned char)i;
			}

			//сравнение
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

		if(debug) printf("KERNEL: %d %d | : %d %d %d %d\n",cur_x,cur_y,res[cur_y * (size_h) + cur_x].pix.x,
																	   res[cur_y * (size_h) + cur_x].pix.y,
																	   res[cur_y * (size_h) + cur_x].pix.z,
																	   res[cur_y * (size_h) + cur_x].pix.w);

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



double det3x3(double* matr) { //только матрицы 3х3 в линейной записи
	return matr[0]*matr[4]*matr[8] + matr[1]*matr[5]*matr[6] + matr[2]*matr[3]*matr[7] -
		   matr[2]*matr[4]*matr[6] - matr[0]*matr[5]*matr[7] - matr[1]*matr[3]*matr[8];
}

double det2x2(double matr_0, double matr_1, double matr_2, double matr_3) { //только матрицы 2х2 в линейной записи
	return matr_0*matr_3 - matr_1*matr_2;
}


double3* inverse(double3* matr) { //только матрицы 3х3 в линейной записи || работает верно
	double* matr_by_char = (double*)malloc(9*sizeof(double));
	matr_by_char[0] = matr[0].x; matr_by_char[1] = matr[0].y; matr_by_char[2] = matr[0].z; 
	matr_by_char[3] = matr[1].x; matr_by_char[4] = matr[1].y; matr_by_char[5] = matr[1].z; 
	matr_by_char[6] = matr[2].x; matr_by_char[7] = matr[2].y; matr_by_char[8] = matr[2].z; 

	double det_main = det3x3(matr_by_char);

	double* matr_minore = (double*)malloc(9*sizeof(double));
	matr_minore[0] = (double)( det2x2(matr_by_char[4],matr_by_char[5],matr_by_char[7],matr_by_char[8])/det_main);
	matr_minore[3] = (double)(-det2x2(matr_by_char[3],matr_by_char[5],matr_by_char[6],matr_by_char[8])/det_main);
	matr_minore[6] = (double)( det2x2(matr_by_char[3],matr_by_char[4],matr_by_char[6],matr_by_char[7])/det_main);

	matr_minore[1] = (double)(-det2x2(matr_by_char[1],matr_by_char[2],matr_by_char[7],matr_by_char[8])/det_main);
	matr_minore[4] = (double)( det2x2(matr_by_char[0],matr_by_char[2],matr_by_char[6],matr_by_char[8])/det_main);
	matr_minore[7] = (double)(-det2x2(matr_by_char[0],matr_by_char[1],matr_by_char[6],matr_by_char[7])/det_main);

	matr_minore[2] = (double)( det2x2(matr_by_char[1],matr_by_char[2],matr_by_char[4],matr_by_char[5])/det_main);
	matr_minore[5] = (double)(-det2x2(matr_by_char[0],matr_by_char[2],matr_by_char[3],matr_by_char[5])/det_main);
	matr_minore[8] = (double)( det2x2(matr_by_char[0],matr_by_char[1],matr_by_char[3],matr_by_char[4])/det_main);

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

double3* avg(std::vector<std::vector<pixel>> classes) { 
	double3* res = (double3*)calloc(classes.size(),sizeof(double3));
	for (int i = 0; i < classes.size(); ++i) {
		for (int j = 0; j < classes[i].size(); ++j){
			res[i].x += (double)classes[i][j].pix.x;
			res[i].y += (double)classes[i][j].pix.y;
			res[i].z += (double)classes[i][j].pix.z;
		}

		res[i].x /= (double)classes[i].size();
		res[i].y /= (double)classes[i].size();
		res[i].z /= (double)classes[i].size();
	}
	return res;
}

double3* cov(std::vector<std::vector<pixel>> classes) { //заполняем массив *обратными* матрицами ковариации
	double3* res = (double3*)calloc(classes.size()*3, sizeof(double3));
	double3* avg_ = avg(classes);

	for (int i = 0; i < classes.size(); ++i) { 
		if(debug) printf("COV: avg of cur class:\n");
		if(debug) printf("COV: %f %f %f\n",avg_[i].x,avg_[i].y,avg_[i].z);
		if(debug) printf("COV: ------\n");
		for (int j = 0; j < classes[i].size(); ++j){
			double3 dif = make_double3((double)classes[i][j].pix.x - avg_[i].x, (double)classes[i][j].pix.y - avg_[i].y, (double)classes[i][j].pix.z - avg_[i].z);
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


		res[3*i].x /= (double)(classes[i].size() - 1);
		res[3*i].y /= (double)(classes[i].size() - 1);
		res[3*i].z /= (double)(classes[i].size() - 1);

		res[3*i + 1].x /= (double)(classes[i].size() - 1);
		res[3*i + 1].y /= (double)(classes[i].size() - 1);
		res[3*i + 1].z /= (double)(classes[i].size() - 1);

		res[3*i + 2].x /= (double)(classes[i].size() - 1);
		res[3*i + 2].y /= (double)(classes[i].size() - 1);
		res[3*i + 2].z /= (double)(classes[i].size() - 1);

		if(debug) printf("\nCOV:  matrix:\n");
		if(debug) printf("COV: %f %f %f\n",res[3*i].x,res[3*i].y,res[3*i].z);
		if(debug) printf("COV: %f %f %f\n",res[3*i+1].x,res[3*i+1].y,res[3*i+1].z);
		if(debug) printf("COV: %f %f %f\n",res[3*i+2].x,res[3*i+2].y,res[3*i+2].z);



		double3* matr = (double3*)malloc(3*sizeof(double3));
		matr[0] = res[3*i];
		matr[1] = res[3*i + 1];
		matr[2] = res[3*i + 2];

		double3* real_matr = inverse(matr);
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
		w = 0; h = 0;

		if(fin.fail()) {
			throw 101;
		}

		fin.clear();
		fin.seekg(0);
		
		std::bitset<8*4> w_(int(0));
		std::bitset<8*4> h_(int(0));

		char btmp;
		for (int j = 0; j < 4; ++j) {
			fin.get(btmp);
			std::bitset<8> tmp_ (btmp);
			for (int i =0;i<8;++i) {
				w_[j*8 + i] = tmp_[i];
			}
		}
		
		for (int j = 0; j < 4; ++j) {
			fin.get(btmp);
			std::bitset<8> tmp_ (btmp);
			for (int i =0;i<8;++i) {
				h_[j*8 + i] = tmp_[i];
			}
		}

		w = int(w_.to_ulong());
		h = int(h_.to_ulong());
		std::cerr << w_ << " (" << w << ")\n" << h_ << " (" << h << ")\n";


		if(fin.rdstate() == std::ios_base::goodbit) {
			std::cout << "OK\n";
		}
		if(fin.rdstate() == std::ios_base::badbit) {
			std::cerr << "bad\n";
		}
		if(fin.rdstate() == std::ios_base::failbit) {
			std::cerr << "fail\n";
		}
		if(fin.rdstate() == std::ios_base::eofbit) {
			std::cerr << "EOF\n";
		}

		pixels = (pixel*)malloc(sizeof(pixel)*w*h);

		unsigned long long cnt = 8;
		for(int i = 0; i < h; ++i){
			for(int j = 0;j < w; ++j){
				struct pixel cur_pixel = pixel().read_from_file(fin);
				pixels[j*h + i] = cur_pixel;
				cnt ++;
			}
		}


		//полный вывод файла на stderr потому что я уже вообще не понимаю, что происходит
		/*
		fin.clear();
		fin.seekg(0); 
		unsigned long long cnt = 1;
		char tmp;
		while(fin.peek() != EOF) {
			fin.get(tmp);
			std::bitset<8> bittmp(tmp);
			std::cerr << bittmp << " ";

			if(cnt % 8 == 0) {
				std::cerr << "\n";
			}

			cnt ++;
			if(cnt > (unsigned long long)8 + (unsigned long long)65536 * (unsigned long long)65536) {
				break;
			}
		}*/

		std::cerr << "got " << cnt << " bytes\n";

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
		cudaEvent_t start, stop;
		CSC(cudaEventCreate(&start));
		CSC(cudaEventCreate(&stop));

		std::vector<std::vector<pixel>> class_pixels;
		if(!silent)	std::cerr << "CLSS: started\n";

		if(w <= 0 || h <= 0) {
			printf("CLSS: incorrect size\n");
			return;
		}
		
		//заменить координаты на цвета
		for(int i=0;i < class_coords.size();++i) { 
			std::vector<pixel> cur_class_pixels;
			for(long j = 0; j < class_coords[i].size();j += 2) {
				//pixels[class_coords[i][j]*h + class_coords[i][j + 1]].print();
				cur_class_pixels.push_back(pixels[class_coords[i][j]*h + class_coords[i][j + 1]]);
			}
			class_pixels.push_back(cur_class_pixels);
		}
		//прогнать через avg и cov

		double3* avg_ = avg(class_pixels);
		double3* cov_ = cov(class_pixels);

		if(!silent)	std::cerr << "CLSS: copying to constant - began\n";
		int avg_size = class_pixels.size();		
		CSC(cudaMemcpyToSymbol( CL_NUM,&avg_size,sizeof(int)));

		CSC(cudaMemcpyToSymbol( AVG, avg_, sizeof(double3)*class_pixels.size()));
		CSC(cudaMemcpyToSymbol( COV, cov_, sizeof(double3)*3*class_pixels.size()));

		if(!silent)	std::cerr << "CLSS: preparations - began\n";
		//начало копированния на устройство

		pixel* res = (pixel*)malloc(w*h*sizeof(pixel)); //массив для принятия результата на хосте 

		for (unsigned long long beg_pos = 0; beg_pos < w*h; beg_pos += MAX_TEX_SIZE) {
			//--------------для работы с текстурами и приема результата--------------------------------------------------
			unsigned long long end_pos;
			if(beg_pos + MAX_TEX_SIZE > w*h) {
				end_pos = w*h;
			}
			else {
				end_pos = beg_pos + MAX_TEX_SIZE;
			}

			unsigned long long dim_sum = end_pos - beg_pos; 
			if(!silent) std::cerr << "CLSS: begin: " << beg_pos << " end: " << end_pos << "\n";

			int w_ballanced = 1;
			int h_ballanced = dim_sum;
			for (int i = 1; i <= int(sqrt(dim_sum)); ++i) { //приближение размерностей к квадрату (т.к. сторона текстуры должна быть меньше 2^16)
				if(dim_sum % i == 0) {
					w_ballanced = i;
					h_ballanced = dim_sum / i;
				}
			}
			if(!silent) std::cerr << "CLSS: calculated optimal dims: " << w_ballanced << " " << h_ballanced << "\n";
			
			pixel* res_pix; //массив для записи результата на устройстве

			CSC(cudaMalloc (&res_pix, sizeof(pixel)*w_ballanced*h_ballanced));
			cudaArray* tex_array; //создание массива в текстурной памяти
			cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<uchar4>();

			CSC(cudaMallocArray(&tex_array,&channelDesc,h_ballanced,w_ballanced));

			CSC(cudaMemcpy2DToArray(tex_array, 0, 0, pixels + beg_pos, sizeof(uchar4)*h_ballanced,sizeof(uchar4)*h_ballanced,w_ballanced, cudaMemcpyHostToDevice));

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

			CSC(cudaEventRecord(start));
			kernel_classify<<<64,64>>> (texture,w_ballanced,h_ballanced,res_pix);
			CSC(cudaEventRecord(stop));
			CSC(cudaEventSynchronize(stop));

			if(!silent)	std::cerr << "CLSS: kernel - done\n";
			CSC(cudaMemcpy(res + beg_pos, res_pix, sizeof(pixel)*w_ballanced*h_ballanced, cudaMemcpyDeviceToHost));
			if(!silent)	std::cerr << "CLSS: back copy - done\n";
			
			CSC(cudaDestroyTextureObject(texture));
			CSC(cudaFreeArray(tex_array));
			CSC(cudaFree(res_pix));
			
			float milliseconds = 0;
			CSC(cudaEventElapsedTime(&milliseconds, start, stop));
			
			if(!silent) std::cerr << "SSAA: kernel time = " << milliseconds << "\n";
		}

		free(pixels);
		pixels = res;
	}

};

int main() {

	std::string img_name, res_name;
	std::cin >> img_name;
	std::cin >> res_name;

	try{ 
		//int w,h;
		
		image img = image(img_name);

		//w = img.Size().first;
		//h = img.Size().second;

		//заполнение таблички с координатами пикселей классов
		int num_classes;
		std::cin >> num_classes;
		std::vector<std::vector<int>> classes_info;
		for (int i = 0; i < num_classes; ++i) { 
			long num_exmpls;
			std::cin >> num_exmpls;
			std::vector<int> cur_class;
			for(long j = 0; j < num_exmpls*2; ++j) {
				int cur_exmpl;
				std::cin >> cur_exmpl;
				cur_class.push_back(cur_exmpl);
			}
			classes_info.push_back(cur_class);
		}
		if(!silent) {
			std::cerr << "--got class info--\n";
			for (int i=0;i<num_classes; ++i) {
				std::cerr << classes_info[i].size() << " | ";
				/*for(int j = 0; j < classes_info[i].size(); ++j) {
					std::cerr << classes_info[i][j] << " ";
				}*/
				if((i + 1) % 32 == 0) {
					std::cerr << "\n";
				}
			}
			std::cerr << "\n";
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


