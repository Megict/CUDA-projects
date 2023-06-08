#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <bitset>

#define verbal false

#define CSC(call)  																											\
do {																														\
	cudaError_t err = call;																									\
	if (err != cudaSuccess) {																								\
		std::cerr << "ERROR in " << __FILE__ << ":" << __LINE__ << ". Message: " << cudaGetErrorString(err) << "\n";		\
		exit(0);																											\
	}																														\
} while(0)

//вариант 4
//SSAA

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
		std::cerr << "| " << pix.x << " " << pix.y << " " << pix.z << " |" << pix.w << "|\n";
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

__global__ void kernel(cudaTextureObject_t pix,int size_w, int size_h , int coef_w, int coef_h, pixel* res) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x; 
	int idy = blockIdx.y * blockDim.y + threadIdx.y; 
	int offset_x = blockDim.x * gridDim.x; 
	int offset_y = blockDim.y * gridDim.y; 

	int idy_start = idy;
	//printf("coefs  %d <-w|h-> %d\n\n",coef_w,coef_h);
	//printf("sizes  %d <-w|h-> %d\n\n",size_w,size_h);

	while(coef_h*idx < size_h) {
		while(coef_w*idy < size_w) {

			int cur_x = coef_h*idx ;
			int cur_y = coef_w*idy ;

			uchar4 cur_pix = tex2D<uchar4>(pix,cur_x,cur_y);/*первая - высотная координата, вторая - строчная*/
			//printf("(%d %d) %d ",cur_x,cur_y,(cur_pix.x + cur_pix.y + cur_pix.z)/3);
			
			unsigned int r = 0,g = 0,b = 0,a = 0;
			//printf("(%d %d) %d %d | %d %d %d | %d\n",idx,idy,cur_x,cur_y,cur_pix.x,cur_pix.y,cur_pix.z,cur_pix.w);

			//printf("filter shape: %d <-h|w-> %d \n",coef_h,coef_w);
			for(int i = 0; i < coef_h; ++i) {
				for(int j = 0; j < coef_w; ++j) {
					uchar4 add_pix = tex2D<uchar4>(pix,cur_x + i,cur_y + j);
					r += (unsigned int) add_pix.x;
					g += (unsigned int) add_pix.y;
					b += (unsigned int) add_pix.z;
					a += (unsigned int) add_pix.w;
					//printf("(%d %d) aded from %d %d |%d %d %d|\n",cur_x,cur_y,cur_x + i,cur_y + j,add_pix.x,add_pix.y,add_pix.z);
				}
			}

			//printf("recorded from %d %d (%d %d) to res at (%d) put |%u %u %u| / |%d|\n",idx,idy,cur_x,cur_y,idy * (size_h/coef_h) + idx,r,g,b,(coef_w * coef_h));
			res[idy * (size_h/coef_h) + idx].pix.x  = (unsigned char) (r / (coef_w * coef_h));
			res[idy * (size_h/coef_h) + idx].pix.y  = (unsigned char) (g / (coef_w * coef_h));
			res[idy * (size_h/coef_h) + idx].pix.z  = (unsigned char) (b / (coef_w * coef_h));
			res[idy * (size_h/coef_h) + idx].pix.w  = (unsigned char) (a / (coef_w * coef_h));
			
			idy += offset_y;
		}

		//printf("\n");
		idx += offset_x;
		idy = idy_start;
	}
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
		fin.get((char*)&w,5);
		fin.get((char*)&h,5);

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
		int coef_w = w / new_w,
			coef_h = h / new_h;

		std::cerr << "current size: " << w << " " << h << " required size: " << new_w << " " << new_h << "\n";
		
		pixel* res_pix; //массив для записи результата на устройстве
		pixel* res = (pixel*)malloc(new_w*new_h*sizeof(pixel)); //массив для принятия результата на хосте 

		CSC(cudaMalloc (&res_pix, sizeof(pixel)*new_w*new_h));

		//-------------------------------------------------------------------------------
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
		//-------------------------------------------------------------------------------

		std::cerr << "preparations - done\n";
		kernel<<<dimBlock, dimGrid>>>(texture,w,h,coef_w,coef_h,res_pix);
		std::cerr << "kernel - done\n";

		CSC(cudaMemcpy(res, res_pix, sizeof(pixel)*new_w*new_h, cudaMemcpyDeviceToHost));
		free(pixels);
		
		w = new_w;
		h = new_h;
		pixels = res;

		CSC(cudaDestroyTextureObject(texture));
		CSC(cudaFreeArray(tex_array));
		CSC(cudaFree(res_pix));

		return 0;
	}

};

int main() {

	std::string img_name, res_name;
	std::cin >> img_name;
	std::cin >> res_name;

	try{ 
		int w,h,wn,hn;
		
		image img = image(img_name);
		w = img.Size().first;
		h = img.Size().second;
		std::cin >> wn >> hn;

		if(w % wn != 0 || h % hn != 0 || wn > w || hn > h) {
			throw 105; //original img sizes are not divisible by new sizes or new size is bigger then original
		}

		if(verbal) {
			img.print();
		}
		else {
			img.print_size();
		}
		//img.print_visual();

		img.SSAA(wn,hn);
		img.print_to_file(res_name);

		if(verbal) {
			img.print();
		}
		else {
			img.print_size();
		}
		//img.print_visual();

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


