#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#define PNG_NO_SETJMP
#include <sched.h>
#include <assert.h>
#include <png.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <pthread.h>
#include <mpi.h>
#include <omp.h>

#include <smmintrin.h> //mullo
#include <pmmintrin.h> //lddqu
#include <emmintrin.h> //add / sub / store_si128
#define CHUNKSIZE 1
// #define WTIME

#define DATA_TAG 0
#define TERMINATION_TAG 1
#define RESULT_TAG 2

#define WIDTH_MAX_SIZE 10000
// enum TAG{DATA_TAG, TERMINATION_TAG};



void write_png(const char* filename, int iters, int width, int height, const int* buffer) {
    FILE* fp = fopen(filename, "wb");
    assert(fp);
    png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    assert(png_ptr);
    png_infop info_ptr = png_create_info_struct(png_ptr);
    assert(info_ptr);
    png_init_io(png_ptr, fp);
    png_set_IHDR(png_ptr, info_ptr, width, height, 8, PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    png_set_filter(png_ptr, 0, PNG_NO_FILTERS);
    png_write_info(png_ptr, info_ptr);
    png_set_compression_level(png_ptr, 1);
    size_t row_size = 3 * width * sizeof(png_byte);
    png_bytep row = (png_bytep)malloc(row_size);
    for (int y = 0; y < height; ++y) {
        memset(row, 0, row_size);
        for (int x = 0; x < width; ++x) {
            int p = buffer[(height - 1 - y) * width + x];
            png_bytep color = row + x * 3;
            if (p != iters) {
                if (p & 16) {
                    color[0] = 240;
                    color[1] = color[2] = p % 16 * 16;
                } else {
                    color[0] = p % 16 * 16;
                }
            }
        }
        png_write_row(png_ptr, row);
    }
    free(row);
    png_write_end(png_ptr, NULL);
    png_destroy_write_struct(&png_ptr, &info_ptr);
    fclose(fp);
}



#ifdef WTIME
    double starttime, endtime;
    double Total_Compt_time = 0;
    double Compt_starttime, Compt_endtime;
    double Total_Comm_time = 0;
    double Comm_starttime, Comm_endtime;

#endif


 /////////// GLOBAL Variables ////////////// 
int iters;
double left;
double right;
double lower;
double upper;
int width; 
int height;
/* allocate memory for image */
int* image;


__m128d _2y0;
__m128d _2x0;
__m128d _x;
__m128d _y;
__m128d _length_squared;
__m128d x2;
__m128d y2;
// __m128d _2four = _mm_set_pd((double)4, (double)4);
// __m128d _2two = _mm_set_pd((double)2, (double)2);

__m128i _repeats;
__m128i _iters;
__m128i and_result;
__m128i cmp_length_result;


unsigned long long repeats[2];
// int cmp_iter_greater; // sould be 3

// Received from each process.
typedef struct MPI_send_data{
    int row_number;
    int tag;
}Send_data;

// Received from each process.
typedef struct MPI_recv_data{
    int row_number;
    int rank;
    double recv_ans[WIDTH_MAX_SIZE];
}Recv_data;


int count = 2; // 3 elements
const int array_of_blocklengths[2] = {1, 1};
const MPI_Aint array_of_displacements[2] = { offsetof(MPI_send_data, row_number), offsetof(MPI_send_data, tag)}; 
const MPI_Datatype array_of_types[2] = {MPI_INT, MPI_INT}; //row_num, tag
MPI_Datatype MPI_Send_data;



int count_ = 3; // 2 elements
const int array_of_blocklengths_[3] = {1, 1, WIDTH_MAX_SIZE};
const MPI_Aint array_of_displacements_[3] = { offsetof(MPI_recv_data, row_number), offsetof(MPI_recv_data, rank),offsetof(MPI_recv_data, recv_ans)}; 
const MPI_Datatype array_of_types_[3] = {MPI_INT, MPI_INT, MPI_DOUBLE};
MPI_Datatype MPI_Recv_data;


int mpi_rank, mpi_size, omp_threads, omp_thread;

///////////////  Dynamic Task Assignment /////////////////////
///////////////    Work Pool Approach    /////////////////////


// Master: rank 0 process
void Assign_Process(){
    int count = 0;
    int row = 0;

    for(int k=1; k< mpi_size; k++){
        //send ONE row to process k
        Send_data send_data_;
        send_data_.row_number = row;
        send_data_.tag = DATA_TAG;

        #ifdef WTIME
        Comm_starttime = MPI_Wtime();
        #endif

        MPI_Send( &send_data_ , 1 , MPI_Send_data , k , 0 , MPI_COMM_WORLD);
        #ifdef WTIME
        Comm_endtime = MPI_Wtime();
        Total_Comm_time += Comm_endtime - Comm_starttime;
        #endif
        count++;
        row++;
    }

    while(count>0){
        //recv from slave
        MPI_Status status;
        Recv_data recv_data_;

        #ifdef WTIME
        Comm_starttime = MPI_Wtime();
        #endif

        MPI_Recv( &recv_data_ , 1 , MPI_Recv_data , MPI_ANY_SOURCE , 0 , MPI_COMM_WORLD , &status);
        
        #ifdef WTIME
        Comm_endtime = MPI_Wtime();
        Total_Comm_time += Comm_endtime - Comm_starttime;
        #endif        
        
        count--;
        int index_start = recv_data_.row_number*width;
        int counter = 0;
        
        for(int i= index_start; i <index_start+width; i++){
            image[i] = recv_data_.recv_ans[i-index_start];
        }

        if(row < height){ //height: number of row
            Send_data send_data_;
            send_data_.row_number = row;
            send_data_.tag = DATA_TAG;
            
            #ifdef WTIME
            Comm_starttime = MPI_Wtime();
            #endif
            
            MPI_Send( &send_data_ , 1 , MPI_Send_data , recv_data_.rank , 0 , MPI_COMM_WORLD);
            
            #ifdef WTIME
            Comm_endtime = MPI_Wtime();
            Total_Comm_time += Comm_endtime - Comm_starttime;
            #endif
            
            count++;
            row++;
        }
        else{
            Send_data send_data_;
            send_data_.row_number = row;
            send_data_.tag = TERMINATION_TAG;
            #ifdef WTIME
            Comm_starttime = MPI_Wtime();
            #endif

            MPI_Send( &send_data_ , 1 , MPI_Send_data , recv_data_.rank , 0 , MPI_COMM_WORLD);
            #ifdef WTIME
            Comm_endtime = MPI_Wtime();
            Total_Comm_time += Comm_endtime - Comm_starttime;
            #endif        
        
        }
    }
    return;
}

// Slave: other processes
void Slave_Calculate(){

    MPI_Status status;
    // recv from master process
    Send_data recv_data;
    #ifdef WTIME
        Comm_starttime = MPI_Wtime();
    #endif

    MPI_Recv( &recv_data , 1 , MPI_Send_data , 0 , 0, MPI_COMM_WORLD , &status);
    #ifdef WTIME
        Comm_endtime = MPI_Wtime();
        Total_Comm_time += Comm_endtime - Comm_starttime;
    #endif    


    
    while(recv_data.tag == DATA_TAG){

        #ifdef WTIME
            Compt_starttime = MPI_Wtime();
        #endif

        // Calculate single row
        Recv_data return_data;
        double y0 = lower + recv_data.row_number * ((upper - lower) / height);
        
            _2y0 =  _mm_set_pd (y0,y0); // Load y0
            
            long long width2 = (width>>1)<<1; // promise the vectorized loop is 128-bits. 

            #pragma omp parallel for schedule(dynamic, CHUNKSIZE)
            for (long long i = 0; i < width2; i+=2) {
                double x0L = left + i * ((right - left) / width); // Along x-axis (Real-axis)
                double x0R = left + (i+1)*((right - left) / width);

                // _2x0[1] = High, (i+1)
                // _2x0[0] = Low, i
                _2x0 = _mm_set_pd (x0R,x0L); // {x0L | x0R}
            
                _x = _mm_setzero_pd();
                _y = _mm_setzero_pd();
                x2= _mm_setzero_pd();
                y2 = _mm_setzero_pd();
                _length_squared = _mm_setzero_pd();
                _repeats = _mm_set_epi64x(0,0);
                // repeats[1] = repeats[0] = 0;

                /////// While-loop BREAK Idea: 
                // A : CMP_ITER_GT_REP
                // B : CMP_LEN_LT_FOUR
                // AND: A && B 
                // if AND ==0  break;
                // else: keep calculating



                while(1){
                    cmp_length_result =   _mm_castpd_si128(_mm_cmplt_pd( _length_squared, _mm_set_pd((double)4, (double)4)));
                    and_result =  _mm_and_si128( _mm_cmpgt_epi64( _iters, _repeats), cmp_length_result);

                    if( _mm_movemask_epi8(and_result)== 0) break;                   
                    _repeats = _mm_add_epi64(_repeats, _mm_and_si128( cmp_length_result,  _mm_set_epi64x(1,1)));

                    /////////////// Compute Mandelbrot set ///////////////
                    _y = _mm_add_pd(_mm_mul_pd(_mm_mul_pd(_mm_set_pd((double)2, (double)2), _x), _y),_2y0);
                    _x = _mm_add_pd(_mm_sub_pd(x2,y2), _2x0);

                    // update X_sqr, Y_sqr, Len_sqr.
                    x2 = _mm_mul_pd(_x, _x);
                    y2 = _mm_mul_pd(_y, _y);
                    _length_squared = _mm_add_pd(x2,y2);
                }
                // 3. Store result back to memory
                _mm_store_si128((__m128i*)&repeats, _repeats);
                return_data.recv_ans[i] = repeats[0]; 
                return_data.recv_ans[i+1] = repeats[1];  
            }

            // If not EVEN, should do ONE more pixel.
            for(long long i=width2; i<width; i++){
                // printf("Width not even!\n");
                // int i = width-1;
                double x0 = left + i* ((right - left) / width); // Along x-axis (Real-axis)
                int repeats = 0; // number of iterations. <= iters.
                double x = 0;    // x value (Real part)
                double y = 0;    // y value (Imaginary part)
                double length_squared = 0; // |Z|                    
                while (repeats < iters && length_squared < 4) {
                    double temp = x * x - y * y + x0; // next z.real   c.real = x0
                    y = 2 * x * y + y0;               // next z.imag   c.imag = y0.
                    x = temp;
                    length_squared = x * x + y * y;   // compute | Znext |
                    ++repeats;                        // one more iteration
                }
                return_data.recv_ans[i] = repeats;   
            }

        return_data.row_number = recv_data.row_number;
        return_data.rank = mpi_rank; // my_rank
          
        #ifdef WTIME
            Compt_endtime = MPI_Wtime();
            Total_Compt_time += Compt_endtime - Compt_starttime;
        #endif


        #ifdef WTIME
            Comm_starttime = MPI_Wtime();
        #endif

        MPI_Send( &return_data , 1 ,  MPI_Recv_data , 0 , 0 , MPI_COMM_WORLD);
        MPI_Recv( &recv_data , 1 , MPI_Send_data , 0 , 0, MPI_COMM_WORLD , &status);

        #ifdef WTIME
            Comm_endtime = MPI_Wtime();
            Total_Comm_time += Comm_endtime - Comm_starttime;
        #endif

    }
    // else: terminate
    return;
}



int main(int argc, char** argv) {
    /* detect how many CPUs are available */
    cpu_set_t cpu_set;
    sched_getaffinity(0, sizeof(cpu_set), &cpu_set);
    // printf("%d cpus available\n", CPU_COUNT(&cpu_set));



    /* argument parsing */
    assert(argc == 9); // 8 arguments
    const char* filename = argv[1];
    iters = strtol(argv[2], 0, 10); //string to long
    left = strtod(argv[3], 0);
    right = strtod(argv[4], 0);
    lower = strtod(argv[5], 0);
    upper = strtod(argv[6], 0);
    width = strtol(argv[7], 0, 10);
    height = strtol(argv[8], 0, 10);

    MPI_Init(&argc, &argv);


    MPI_Type_create_struct( count,
                        array_of_blocklengths,
                        array_of_displacements,
                        array_of_types,
                        &MPI_Send_data);
    MPI_Type_commit( &MPI_Send_data );

    MPI_Type_create_struct(count_,
                       array_of_blocklengths_,
                       array_of_displacements_,
                       array_of_types_,
                       &MPI_Recv_data);
    MPI_Type_commit( &MPI_Recv_data );

    // Get #proc and rank
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);


    #ifdef WTIME
    double starttime, endtime;
    double IO_starttime, IO_endtime;
    double Total_IO_time = 0;
    if(mpi_rank==0){
        starttime = MPI_Wtime();
    }
    #endif

    /* allocate memory for image */
    image = (int*)malloc(width * height * sizeof(int));
    assert(image);

    _iters = _mm_set_epi64x((long long)iters, (long long)iters);
    

    /* mandelbrot set */

    // Load Balancing
    //////////////// Dynamic workload ////////////////

    if(mpi_size == 1){
            // Special case: ONLY ONE PROCESS
        // #pragma omp for shedule(dynamic)
    #ifdef WTIME
        Compt_starttime = MPI_Wtime();
    #endif        
        
        #pragma omp parallel for schedule(dynamic, CHUNKSIZE)
        for (int j = 0; j < height; ++j) { // Go through every pixel from y-axis view. (Along Imag. axis)
            double y0 = lower + j * ((upper - lower) / height);
           
           


            _2y0 =  _mm_set_pd (y0,y0); // Load y0
            
            long long width2 = (width>>1)<<1; // promise the vectorized loop is 128-bits. 

            #pragma omp parallel for schedule(dynamic, CHUNKSIZE)
            for (long long i = 0; i < width2; i+=2) {
                double x0L = left + i * ((right - left) / width); // Along x-axis (Real-axis)
                double x0R = left + (i+1)*((right - left) / width);

                // _2x0[1] = High, (i+1)
                // _2x0[0] = Low, i
                _2x0 = _mm_set_pd (x0R,x0L); // {x0L | x0R}
            
                _x = _mm_setzero_pd();
                _y = _mm_setzero_pd();
                x2= _mm_setzero_pd();
                y2 = _mm_setzero_pd();
                _length_squared = _mm_setzero_pd();
                _repeats = _mm_set_epi64x(0,0);
                // repeats[1] = repeats[0] = 0;

                /////// While-loop BREAK Idea: 
                // A : CMP_ITER_GT_REP
                // B : CMP_LEN_LT_FOUR
                // AND: A && B 
                // if AND ==0  break;
                // else: keep calculating



                while(1){
                    cmp_length_result =   _mm_castpd_si128(_mm_cmplt_pd( _length_squared, _mm_set_pd((double)4, (double)4)));
                    and_result =  _mm_and_si128( _mm_cmpgt_epi64( _iters, _repeats), cmp_length_result);

                    if( _mm_movemask_epi8(and_result)== 0) break;                   
                    _repeats = _mm_add_epi64(_repeats, _mm_and_si128( cmp_length_result,  _mm_set_epi64x(1,1)));

                    /////////////// Compute Mandelbrot set ///////////////
                    _y = _mm_add_pd(_mm_mul_pd(_mm_mul_pd(_mm_set_pd((double)2, (double)2), _x), _y),_2y0);
                    _x = _mm_add_pd(_mm_sub_pd(x2,y2), _2x0);

                    // update X_sqr, Y_sqr, Len_sqr.
                    x2 = _mm_mul_pd(_x, _x);
                    y2 = _mm_mul_pd(_y, _y);
                    _length_squared = _mm_add_pd(x2,y2);
                }
                // 3. Store result back to memory
                _mm_store_si128((__m128i*)&repeats, _repeats);
                image[j * width + i] = repeats[0]; 
                image[j * width + i +1] = repeats[1];
            }

            // If not EVEN, should do ONE more pixel.
            for(long long i=width2; i<width; i++){
                // printf("Width not even!\n");
                // int i = width-1;
                double x0 = left + i* ((right - left) / width); // Along x-axis (Real-axis)
                int repeats = 0; // number of iterations. <= iters.
                double x = 0;    // x value (Real part)
                double y = 0;    // y value (Imaginary part)
                double length_squared = 0; // |Z|                    
                while (repeats < iters && length_squared < 4) {
                    double temp = x * x - y * y + x0; // next z.real   c.real = x0
                    y = 2 * x * y + y0;               // next z.imag   c.imag = y0.
                    x = temp;
                    length_squared = x * x + y * y;   // compute | Znext |
                    ++repeats;                        // one more iteration
                }
                image[j * width + i] = repeats; 
            } 
            // for (int i = 0; i < width; ++i) {
            //     double x0 = left + i * ((right - left) / width); // Along x-axis (Real-axis)

            //     int repeats = 0; // number of iterations. <= iters.
            //     double x = 0;    // x value (Real part)
            //     double y = 0;    // y value (Imaginary part)
            //     double length_squared = 0; // |Z|
            //     while (repeats < iters && length_squared < 4) {
            //         double temp = x * x - y * y + x0; // next z.real   c.real = x0
            //         y = 2 * x * y + y0;               // next z.imag   c.imag = y0.
            //         x = temp;
            //         length_squared = x * x + y * y;   // compute | Znext |
            //         ++repeats;                        // one more iteration
            //     }
            //     image[j * width + i] = repeats;
            // }


        }

        #ifdef WTIME
            Compt_endtime = MPI_Wtime();
            Total_Compt_time += Compt_endtime - Compt_starttime;
        #endif           
           

    }
    else{
        // Load Balancing
        //////////////// Dynamic workload ////////////////
        if(mpi_rank == 0){
            Assign_Process();
        }else{
            Slave_Calculate();
        }
    }

    int* output_image;
    int *recv_cnt_array;
    int *displ_array;
    recv_cnt_array = (int*)malloc(sizeof(int)*mpi_size);
    displ_array = (int*)malloc(sizeof(int)*mpi_size);


    if(mpi_rank==0){
        #ifdef WTIME
        IO_starttime = MPI_Wtime();
        #endif

        write_png(filename, iters, width, height, image);
        
        #ifdef WTIME
        IO_endtime = MPI_Wtime();
        Total_IO_time += IO_endtime - IO_starttime;
        #endif        
        
        free(output_image);
        // printf("Process 0 done writing image.\n");
    }else{
        // printf("Process %d done calculation\n",mpi_rank);
    }
    free(image);
    // printf("All processes done, write image.\n");
    #ifdef WTIME
    if(mpi_rank==0){
        endtime = MPI_Wtime();
        printf("Master Took %.5f seconds.\n",endtime-starttime);
        printf("Master CPU time: %.5f seonds\n",endtime-starttime - Total_Comm_time - Total_IO_time);
        printf("Master Comm time: %.5f seonds\n",Total_Comm_time);
        printf("Master IO time: %.5f seonds\n",Total_IO_time);
        // printf("Process 0 done writing image.\n"); 
    }else{
        // printf("rank %d Took %.5f seconds.\n",mpi_rank, endtime-starttime);
        // printf("rank %d CPU time: %.5f seonds\n",mpi_rank , endtime-starttime - Total_Comm_time - Total_IO_time);
        printf("rank %d Computation time: %.5f seonds\n",mpi_rank, Total_Compt_time);
        printf("rank %d Comm time: %.5f seonds\n", mpi_rank, Total_Comm_time);
    }
    #endif

    MPI_Finalize();

}


/// This version: want to try using bitwise AND to serve as break result!
/// while(1): while(  _mm_and_pd( _mm_cmplt_pd(repeats, _iters), 0)  &&  _mm_and_pd( _mm_cmplt_pd(_len_squared, _2four), 0) )
