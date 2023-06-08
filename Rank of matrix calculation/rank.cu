#include <thrust/extrema.h>
#include <thrust/device_vector.h>

#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <bitset>

#define silent false 	//no stderr except actual errors
#define verbal true		 
#define visual false		 
#define debug  false	//do printf in kernel

#define gridsize 32

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

//вариант 6
//Нахождение ранга матрицы

__global__ void kernel_swap_rows(double* elements, int row_1, int row_2, int n, int m) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int offset_x = blockDim.x * gridDim.x;

	__shared__ double swp [1024];
	while(idx < m) {
		swp[threadIdx.x] = elements[idx*n + row_1];
		elements[idx*n + row_1] = elements[idx*n + row_2];
		elements[idx*n + row_2] = swp[threadIdx.x];
		
		if (debug) printf("KERNEL: swaping %lf and %lf at idx %d\n",elements[idx*n + row_1], elements[idx*n + row_2], idx);

		idx += offset_x;
	}
}

__device__ bool dev_close_to_zero(double val) {
	if(fabs(val) < EPS) {
		return true;
	}
	return false;
} 

__global__ void kernel_gaussian_step(double* elements, int n, int m, int start_row_index, int active_colomn) {
	//n - количество строк (элементов в столбце)
	//m - количество столбцов (элементов в строке)

	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int idy = blockIdx.y * blockDim.y + threadIdx.y;

	int cur_row = idx + start_row_index + 1;
	int cur_col = idy + active_colomn + 1; 
	int in_row_offset = blockDim.y * gridDim.y;
	int other_row_offset = blockDim.x * gridDim.x; 

	__shared__ double coef[32][32];

	while(cur_row < n) {
		
		coef[threadIdx.x][threadIdx.y] = elements[active_colomn*n + cur_row] / elements[active_colomn*n + start_row_index];

		while(cur_col < m) {
			elements[cur_col*n + cur_row] -= coef[threadIdx.x][threadIdx.y]*elements[cur_col*n + start_row_index];
			cur_col += in_row_offset;
		}
		cur_row += other_row_offset;
		cur_col = idy + active_colomn + 1;
	}
}

__global__ void kernel_row_is_zero(double* elements, int n, int m, int row, int start_colomn, bool* res) {
	int idx = start_colomn + blockIdx.x * blockDim.x + threadIdx.x;
	int offset_x = blockDim.x * gridDim.x;

	
	while(idx < m) { 
		if(! dev_close_to_zero(elements[idx*n + (row)])) {
			*res = true;
		}
		idx += offset_x;
	}
}

bool close_to_zero(double val) {
	if(fabs(val) < EPS) {
		return true;
	}
	return false;
}

struct Compare {
    __host__ __device__ bool operator()(double num1, double num2) {
        return fabs(num1) < fabs(num2);
    }
};


class matrix{
	int n;
	int m;
	double* array;
	double* device_matrix;
	
	double* max_elm_val; //нужно для одной функции
	int max_elm_pos;

	Compare cmp;
	thrust::device_ptr<double> device_ptr;

public:
	matrix(int n_, int m_, double* array_) {
		n = n_;
		m = m_;
		array = array_;
		CSC(cudaMalloc (&device_matrix, sizeof(double)*m*n));
		CSC(cudaMemcpy (device_matrix, array, sizeof(double)*m*n, cudaMemcpyHostToDevice));
		device_ptr = thrust::device_pointer_cast(device_matrix);

		max_elm_val = (double*)malloc(sizeof(double));
	}

	matrix(int n_, int m_){ //считывание матрицы с stdin
		n = n_; m = m_;
		//n - количество строк (элементов в столбце)
		//m - количество столбцов (элементов в строке)
		double* arr_all = (double*)malloc(sizeof(double)*m*n);

		for (int i = 0; i < n; ++i){ //проход по строкам
			for (int j = 0; j < m; ++j){ //проход по столбцам
				double elm = 0;
				std::cin >> elm;
				arr_all[j*n + i] = elm;
			}
		}

		array = arr_all;

		CSC(cudaMalloc (&device_matrix, sizeof(double)*m*n));
		CSC(cudaMemcpy (device_matrix, array, sizeof(double)*m*n, cudaMemcpyHostToDevice));
		device_ptr = thrust::device_pointer_cast(device_matrix);

		max_elm_val = (double*)malloc(sizeof(double));
	}

	~matrix() {
		free(array);
		free(max_elm_val);
		CSC(cudaFree(device_matrix));
	}

	void update_host_matrix() {
		CSC(cudaMemcpy (array, device_matrix, sizeof(double)*m*n, cudaMemcpyDeviceToHost));
	}

	void update_device_matrix() {
		CSC(cudaMemcpy (device_matrix, array, sizeof(double)*m*n, cudaMemcpyHostToDevice));
	}

	void print() {

		for (int i = 0; i < m; ++i) {
			for (int j = 0; j < n; ++j) {
				std::cerr << array[i*n + j] << " ";
			}
			//std::cerr << "\n---\n";
			std::cerr << "\n";
		}
	}

	void printf() {

		for (int j = 0; j < n; ++j) {
			for (int i = 0; i < m; ++i) {
				
				if(array[i*n + j] >= 0) {
					std::printf(" ");
				}
				std::printf("%.2lf ",array[i*n + j]);

			}
			std::printf("\n");
		}
	}

	void find_max_elm(int array_start,int array_size) {
		//находит максимальный элемент массива и возвращает его индекс
		//если максимальный элемент = 0, то возвращает -1

		max_elm_pos = thrust::max_element(device_ptr + array_start, device_ptr + array_start + array_size, cmp) - device_ptr - array_start;
		
		CSC(cudaMemcpy (max_elm_val, &device_matrix[array_start + max_elm_pos], sizeof(double), cudaMemcpyDeviceToHost));

		if(close_to_zero(max_elm_val[0])) {
			max_elm_pos = -1;
		}
	}

	void swap_rows(int lhs,int rhs) {
		//меняет местами строки lhs и rhs

		if(lhs >= n || rhs >= n) {
			throw INDEX_ERROR;
		}

		if(lhs == rhs) {
			return;
		}

		kernel_swap_rows<<<1024,1024>>>(device_matrix, lhs, rhs, n, m);

	}

	//1 - определить ведущий элемент в столбце i (thrust)
	//2 - переставить строки местами (О(m)) (параллельно, одномерной сеткой)
	//3 - вычислить коэфециенты для каждой строки (O(n)) ДЕЛАЕТСЯ ВНУТРИ ЯДРА
	//4 - записать коэфициенты в разд. память. 
	//		Каждый блок работает с одной из строк, ему нужен только один коэфициент
	//		Первый варп каждого блока считает коэфициент для соответствующей строки, следующие вары извлекают его из разделяемой памяти
	//		У каждого блока должен быть поток-лидер, который помещает нужный элемент в разделяемую память, остальные потоки должны начать работу только после завершения перемещения
	//		варп потока-лидера будет работать неоптимально
	//5 - преобразовать строки (O(n*n)) (параллельно, двумерной сеткой)

	int rank() {
		int rank = -1;
		int active_element_idx = -1;

		for (int i = 0; i < n - 1; ++i) { //i - текущая строка.
		
			if (visual) {
				std::cerr << "Starting iteration " << i << " of " << n - 1 << "\n";
			}
			active_element_idx += 1;
			if(active_element_idx == m) {
				rank = i;
				break;
			}


			find_max_elm(active_element_idx*n + i,n - i);
			max_elm_pos += i;

			if (visual) {
				std::cerr << "active colomn idx is "<< active_element_idx << " max elm pos is " << max_elm_pos << "\n";
			}
			
			while(max_elm_pos - i == -1 && active_element_idx + 1 < m) {//максимальный элемент строки равен нулю
				active_element_idx += 1;
				find_max_elm(active_element_idx*n + i,n - i);
				max_elm_pos += i;
				
				if (visual) {
					std::cerr << "active colomn idx is "<< active_element_idx << " max elm pos is " << max_elm_pos << "\n";
				}
			}
			if(active_element_idx + 1 == m && max_elm_pos - i == -1) {
				rank = i;
				break;
			}


			if (visual) {
				std::cerr << "\tcur index " << i << "\tindex with max elm " << max_elm_pos <<"\n";
			}

			if(i != max_elm_pos) {
				swap_rows(i,max_elm_pos);
			}
									
			kernel_gaussian_step<<<dim3(gridsize,gridsize),dim3(gridsize,gridsize)>>> (device_matrix, n, m, i, active_element_idx);
			
			if (visual) {
				update_host_matrix(); //нужно для вывода, для быстрой работы отключить
				std::cerr << "after transformation\n";
				printf();
			}

			if(active_element_idx + 1 == m) {
				rank = i + 1;
				break;
			}
		}
		std::cerr << "Finished cycle\n";
		//после завершения цикла нужно определить, занулилась ли последняя строка и, тем самым, понять, каков ранг
		//если цикл завершился до последней строки, то ранг уже вычислен
		if(rank == -1) {
			bool* host_zero_mark = (bool*)malloc(sizeof(bool));
			bool* zero_mark;

			CSC(cudaMalloc (&zero_mark, sizeof(bool)));
			CSC(cudaMemset (zero_mark, 0, sizeof(bool)));

			kernel_row_is_zero<<<256,256>>>(device_matrix, n, m, n - 1, n - 1,zero_mark);
			
			CSC(cudaMemcpy (host_zero_mark, zero_mark, sizeof(bool), cudaMemcpyDeviceToHost));
			
			if(*host_zero_mark) {
				rank = n;
			}
			else {
				rank = n - 1;
			}
		}

		if (verbal) {
			std::cerr << rank << "\n";
			//printf();
		}
		
		return rank;

	}

};

int main() {
	std::ios_base::sync_with_stdio(false);
	std::cin.tie(nullptr);
	
	try{ 
		int n,m;
		std::cin >> n >> m;

		matrix matr(n,m);

		std::cerr << "shape: "<< n << " " << m << "\n";

		if(visual) {
			std::cerr << "--\n";
			matr.print();
			std::cerr << "\n";
			matr.printf();
			std::cerr << "\n";
		}

		int rank = matr.rank();

		if(visual) {
			std::cerr << "-- RANK: " << rank << " --\n";
		}
		std::cout.precision(10);
		std::cout << rank << std::fixed << "\n";
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