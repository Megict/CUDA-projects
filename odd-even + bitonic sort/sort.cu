#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <bitset>
#include <chrono>

#define silent false 	//no stderr except actual errors
#define verbal true		 
#define visual false		 
#define debug  false	//do printf in kernel

#define gridsize_b 512
#define gridsize_t 512
//размер сетки должен быть больше или равен размеру блоков сортировки

#define INDEX_ERROR 800

#define EPS 1e-7

#define CSC(call)  																											\
do {																														\
	cudaError_t err = call;																									\
	if (err != cudaSuccess) {																								\
		std::cerr << "ERROR in " << __FILE__ << ":" << __LINE__ << ". Message: " << cudaGetErrorString(err) << "\n";		\
		exit(0);																											\
	}																														\
} while(0)

//вариант 5
//Сортировка чет-нечет с предварительной битонической сортировкой.

__global__ void kernel_bitonic_step_and_proceed_glob(int* elements,int n,int init_bs) { //помещает массив в разделяемую память
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int offset_x = blockDim.x * gridDim.x;

	
	int block_size = init_bs;
	int tmp = 0;

	while(idx < n) {
		__syncthreads();
		while(block_size >= 2) {
			//int block_num = idx / block_size;
			int block_pos = idx % block_size;

			if(block_pos < block_size / 2) {
				if(elements[idx] > elements[idx + block_size/2]) {
					if(debug) printf("KERNEL: -| %d -- %d ( %d -- %d )\n",idx, idx + block_size/2, elements[idx], elements[idx + block_size/2]);
					tmp = elements[idx];
					elements[idx] = elements[idx + block_size/2];
					elements[idx + block_size/2] = tmp;
				}
				else {
					if(debug) printf("KERNEL: -| %d -- %d ( ok )\n",idx, idx + block_size/2);
				}
			}

			block_size /= 2;
			__syncthreads();
		}
		
		idx += offset_x;
	}
	__syncthreads();
}

__global__ void kernel_bitonic_step_and_proceed(int* elements,int n,int init_bs) { //помещает массив в разделяемую память
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int offset_x = blockDim.x * gridDim.x;

	__shared__ int loc_elements[1024];
	int block_size = init_bs;
	int tmp = 0;

	while(idx < n) {
		loc_elements[threadIdx.x] = elements[idx];
		__syncthreads();
		//printf("KERNEL: -| %d (%d) \t %d\n", threadIdx.x, idx, loc_elements[threadIdx.x]);

		while(block_size >= 2) {
			
			int block_pos = idx % block_size;

			if(block_pos < block_size / 2) {
				if(loc_elements[threadIdx.x] > loc_elements[threadIdx.x + block_size/2]) {
					if(false) printf("KERNEL: -| %d -- %d ( %d -- %d ) (%d -- %d) %d -- %d\n",idx, idx + block_size/2, 
						elements[idx], elements[idx + block_size/2],loc_elements[threadIdx.x], loc_elements[threadIdx.x + block_size/2], threadIdx.x, threadIdx.x + block_size/2);
					tmp = loc_elements[threadIdx.x];
					loc_elements[threadIdx.x] = loc_elements[threadIdx.x + block_size/2];
					loc_elements[threadIdx.x + block_size/2] = tmp;
				}
				else {
					if(false) printf("KERNEL: -| %d -- %d ( %d -- %d ) (%d -- %d) %d -- %d |(ok)\n",idx, idx + block_size/2, 
						elements[idx], elements[idx + block_size/2],loc_elements[threadIdx.x], loc_elements[threadIdx.x + block_size/2], threadIdx.x, threadIdx.x + block_size/2);
				}
			}

			block_size /= 2;
			__syncthreads();
		}
		
		elements[idx] = loc_elements[threadIdx.x];
		idx += offset_x;
		block_size = init_bs;
	}
	__syncthreads();
}


__global__ void kernel_bitonic_sort_init(int* elements,int n, int block_size) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int offset_x = blockDim.x * gridDim.x;
	
	while(idx < n) {
		__syncthreads();
		int block_num = idx / block_size;
		int block_pos = idx % block_size;

		if(block_pos < block_size / 2) {
			//printf("KERNEL: [%d %d] -- %d -- %d --\n",n,block_size,idx, elements[idx]);
			if(elements[idx] > elements[block_num*block_size + block_size - 1 - block_pos]) {
				//printf("KERNEL: -| %d -- %d -- [%d -- %d]\n",idx, block_num*block_size + block_size - 1 - block_pos, elements[idx], elements[block_num*block_size + block_size - 1 - block_pos]);
				int tmp = elements[idx];
				elements[idx] = elements[block_num*block_size + block_size - 1 - block_pos];
				elements[block_num*block_size + block_size - 1 - block_pos] = tmp;
			}
		}

		idx += offset_x;
	}
	__syncthreads();
}

bool close_to_zero(double val) {
	if(fabs(val) < EPS) {
		return true;
	}
	return false;
}


class array{
	int actual_len;
	int len;
	int* elements;

	public:

	array(int len_, int* elements_) {
		len = len_;
		actual_len = len;
		elements = elements_;
	}

	array(int len_) {
		len = len_;
		actual_len = len;
		elements = (int*)malloc(sizeof(int)*len);
		for(int i = 0; i < len; ++i) {
			int elm;
			std::cin.read((char*)(&elm),sizeof(elm));
			elements[i] = elm;
		}
	}

	array(int len_, std::ifstream& fin) {
		len = len_;
		actual_len = len;
		elements = (int*)malloc(sizeof(int)*len);
		for(int i = 0; i < len; ++i) {
			int elm;
			fin.read((char*)(&elm),sizeof(elm));
			elements[i] = elm;
		}
	}

	array(array& other) {
		len = other.len;
		actual_len = len;
		elements = (int*)malloc(sizeof(int)*len);
		
		for(int i = 0; i < len; ++i) {
			elements[i] = other.elements[i];
		}
	}

	~array() {
		free(elements);
	}

	int length() {
		return len;
	}

	int* contains() {
		return elements;
	}

	void print() {
		std::cerr << len << " | ";
		for (int i = 0; i < len; ++i) {
			std::cerr << elements[i] << " ";
		}
		std::cerr << "\n";
	}

	void print(int del) {
		std::cerr << len << " | ";
		for (int i = 0; i < len; ++i) {
			if(i % del == 0) {
				std::cerr << "| ";
			}
			std::cerr << elements[i] << " ";
		}
		std::cerr << "\n";
	}

	void print_to_cout(int del) {
		std::cout << len << " | ";
		for (int i = 0; i < len; ++i) {
			if(i % del == 0) {
				std::cout << "| ";
			}
			std::cout << elements[i] << " ";
		}
		std::cout << "\n";
	}

	void print_first(int size) {
		std::cerr << len << " | ";
		for (int i = 0; i < size && i < len; ++i) {
			std::cerr << elements[i] << " ";
		}
		std::cerr << "\n";
	}

	void print_last(int size) {
		std::cerr << len << " | ";
		for (int i = len - size; i >= 0 && i < len; ++i) {
			std::cerr << elements[i] << " ";
		}
		std::cerr << "\n";
	}

	void print_uf() {
		for (int i = 0; i < len; ++i) {
			int elm = elements[i];
			std::cout.write((char*)(&elm),4);
		}
	}

	void uplen(int new_len) {
		int past_len = len;
		len = new_len;
		elements = (int*)realloc(elements, sizeof(int)*len);
		for (int i = past_len; i < len; ++i) {
			elements[i] = INT_MAX;
		}
		return;
	}

	void upfill() {
		int pastlen = len;
		double pwr = std::log2(double(len));
		if(close_to_zero(double(int(pwr)) - pwr)) {
			//длина и так степень двойки
			return;
		}
		len = std::pow(2,int(pwr) + 1);
		elements = (int*)realloc(elements, sizeof(int)*len);
		for (int i = pastlen; i < len; ++i) {
			elements[i] = INT_MAX;
		}
		return;
	}

	void relen() {
		len = actual_len;
		elements = (int*)realloc(elements, sizeof(int)*len);
	}

	void B(int n) {
		return;
	}

	void odd_even_block(int block_size) {
		cudaEvent_t start, stop;
		CSC(cudaEventCreate(&start));
		CSC(cudaEventCreate(&stop));		

		int block_cnt = len / block_size;
		
		if(len % block_size != 0) {
			block_cnt ++;
			int new_len = block_cnt * block_size;
			uplen(new_len);
		}

		int* elements_dev;
		CSC(cudaMalloc (&elements_dev, sizeof(int)*len));
		CSC(cudaMemcpy (elements_dev, elements, sizeof(int)*len, cudaMemcpyHostToDevice));
		std::cerr << "OEB: prep - done, block cnt = " << block_cnt << "\n";


		//сортировка каждого блока
		CSC(cudaEventRecord(start));
		for (int j = 2; j < block_size; j*=2) {
			kernel_bitonic_sort_init<<<gridsize_b,gridsize_t>>>(elements_dev, block_size*block_cnt , j);
			if(j != 2)
				kernel_bitonic_step_and_proceed<<<gridsize_b,gridsize_t>>>(elements_dev,  block_size*block_cnt, j/2);
			
		}
		kernel_bitonic_sort_init<<<gridsize_b,gridsize_t>>>(elements_dev, block_size*block_cnt, block_size);
		kernel_bitonic_step_and_proceed<<<gridsize_b,gridsize_t>>>(elements_dev, block_size*block_cnt, block_size/2);
		CSC(cudaEventRecord(stop));
		CSC(cudaEventSynchronize(stop));

		float all_time = 0;
		float pre_sort = 0;
		CSC(cudaEventElapsedTime(&pre_sort, start, stop));

		std::cerr << "OEB: sorted:\n";
		//CSC(cudaMemcpy (elements, elements_dev, sizeof(int)*len, cudaMemcpyDeviceToHost));//print(block_size);

		//check_if_sorted_blocks(block_size);

		int even_end_pos = block_cnt*block_size; //весь массив
		int odd_end_pos = (block_cnt - 1)*block_size; //массив за исключение последнего блока

		if(block_cnt % 2 != 0) {
			even_end_pos = (block_cnt - 1)*block_size;
			odd_end_pos = block_cnt*block_size;
		}

		//std::cerr << even_end_pos << " " << odd_end_pos << " | " << block_cnt << "\n";
		
		//CSC(cudaEventRecord(start));
		for (int k = 0; k < block_cnt; ++k) {
			if(k % 2 == 0) {
				kernel_bitonic_sort_init<<<gridsize_b,gridsize_t>>>(elements_dev, even_end_pos, 2*block_size);
				kernel_bitonic_step_and_proceed<<<gridsize_b,gridsize_t>>>(elements_dev, even_end_pos, block_size);
			}
			else {
				kernel_bitonic_sort_init<<<gridsize_b,gridsize_t>>>(elements_dev + block_size, odd_end_pos - block_size, 2*block_size);
				kernel_bitonic_step_and_proceed<<<gridsize_b,gridsize_t>>>(elements_dev + block_size, odd_end_pos - block_size, block_size);
			}
			
			//std::cerr << "\n";CSC(cudaMemcpy (elements, elements_dev, sizeof(int)*len, cudaMemcpyDeviceToHost)); print(block_size);
		}
		CSC(cudaEventRecord(stop));
		CSC(cudaEventSynchronize(stop));

		float oeb_sort = 0;
		CSC(cudaEventElapsedTime(&oeb_sort, start, stop));
		all_time = pre_sort + oeb_sort;

		std::cerr << "OEB: kernel time = " << all_time << "\n";
		std::cerr << "OEB: pre-sort kernel time = " << pre_sort << "\n";
		std::cerr << "OEB: oeb_sort kernel time = " << oeb_sort << "\n";
		
		CSC(cudaMemcpy (elements, elements_dev, sizeof(int)*len, cudaMemcpyDeviceToHost));
		relen();

		return;
	}

	bool check_if_sorted_blocks(int block_len) {
		bool correct = true;
		for(int i = 0; i < len - 1; ++i) {
			if(elements[i] > elements[i + 1] && ((i + 1) % block_len != 0)) {
				std::cerr << " | block sorting error at pos " << i << " vals: " << elements[i] << " " << elements[i + 1] << "\n";
				correct = false;
			}
		}
		return correct;
	}

	bool check_if_sorted() {
		bool correct = true;
		for(int i = 0; i < len - 1; ++i) {
			if(elements[i] > elements[i + 1]) {
				std::cerr << " | sorting error at pos " << i << " vals: " << elements[i] << " " << elements[i + 1] << "\n";
				correct = false;
			}
		}
		return correct;
	}
};

int main() {
	std::ios_base::sync_with_stdio(false);
	std::cin.tie(nullptr);
	
	try{ 
		
		std::string filename;
		std::cin >> filename;
		std::ifstream fin(filename,std::ios::binary);
		int n = 0;

		
		fin.read((char*)(&n), sizeof(n));

		
		/*int n = 0;
		std::cin.read(reinterpret_cast<char *>(&n), sizeof(n));*/
		array arr(n,fin);

		arr.print_first(10);
		//arr.print();

		std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
		arr.odd_even_block(gridsize_t / 2);
		std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
		std::cerr << "total cpu time: " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() / 1000 << " ms \n";

		//arr.print();
		arr.check_if_sorted();
		//arr.print();
		arr.print_uf();

	}
	catch(int err) {
		if (err == 101) {
			std::cerr << "error opening file\n";
		} else
		if (err == 105){
			std::cerr << "error new length\n";
		} else 
		if (err == 800) {
			std::cerr << "error index\n";
		} else{
			std::cerr << "unknown error detected\n";
		}
	}

	return 0;
}

//!! 1048581 !!