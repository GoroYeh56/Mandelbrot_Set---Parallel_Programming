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
#include <smmintrin.h> //mullo
#include <pmmintrin.h> //lddqu
#include <emmintrin.h> //add / sub / store_si128

// #define WTIME



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


typedef struct thread_info{
	int thread_id;
	// unsigned long long num_op;
	// unsigned long long pixels;
	// unsigned long long num_threads;
	// unsigned long long r;
	// unsigned long long k;
	// unsigned long long st;
	// unsigned long long ed;
}thread_info;

// Pthread function here : calculate
void* Calculate(void* my_thread){


#ifdef WTIME
   struct timespec start, end, temp;
   double time_used;
   clock_gettime(CLOCK_MONOTONIC, &start);
#endif 

	thread_info* thread_t = (thread_info*)my_thread;
        // global : image, iters, lower, upper, left, right, width, height.
        // j : thread->id
        int j = thread_t->thread_id;

        double y0 = lower + j * ((upper - lower) / height);
        
            _2y0 =  _mm_set_pd (y0,y0); // Load y0
            
            long long width2 = (width>>1)<<1; // promise the vectorized loop is 128-bits. 

            // #pragma omp parallel for schedule(dynamic, CHUNKSIZE)
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
                image[j * width + i + 1] = repeats[1];

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
                // return_data.recv_ans[i] = repeats;   
                image[j * width + i] = repeats;
            }


#ifdef WTIME
    ///////////// End time //////////////////
   clock_gettime(CLOCK_MONOTONIC, &end);
   if ((end.tv_nsec - start.tv_nsec) < 0) {
       temp.tv_sec = end.tv_sec-start.tv_sec-1;
       temp.tv_nsec = 1000000000 + end.tv_nsec - start.tv_nsec;
   } else {
       temp.tv_sec = end.tv_sec - start.tv_sec;
       temp.tv_nsec = end.tv_nsec - start.tv_nsec;
   }
   time_used = temp.tv_sec + (double) temp.tv_nsec / 1000000000.0;
  
   printf("Thread %d took %.5f second\n", j,time_used);
#endif


	pthread_exit(NULL);
}


int main(int argc, char** argv) {

#ifdef WTIME
   struct timespec start, end, temp;
   double time_used;
   clock_gettime(CLOCK_MONOTONIC, &start);
#endif  

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

    /* allocate memory for image */
    image = (int*)malloc(width * height * sizeof(int));
    assert(image);

    int num_threads = height;
    pthread_t threads[num_threads]; // a threads_array. 
	thread_info thread_INFO[num_threads];

    _iters = _mm_set_epi64x((long long)iters, (long long)iters);
    
    /* mandelbrot set */
    int rc;
    for (int j = 0; j < height; ++j) { // Go through every pixel from y-axis view. (Along Imag. axis)
        // create thread
        thread_INFO[j].thread_id = j;
		rc = pthread_create(&threads[j], NULL, Calculate, (void*)&thread_INFO[j]); 
        // rc = pthread_create(&threads[t], NULL, hello, (void*)&ID[t]); // ID would be the argument passesd to function "hello(threadid)"
        if (rc) {
            printf("ERROR; return code from pthread_create() is %d\n", rc);
            exit(-1);
        }
    }



    // Let C = a + ib (x0 + iy0) 
    // Initially, Z0 = 0
	/////////////// KEY : wait for the last pthread?	
	// Add Barrier 
	for(int i=0; i<num_threads; i++) pthread_join(threads[i],NULL);
	// for(int i=0; i<num_threads; i++){
	// 	// printf("pixels_thread %d: %llu\n",i,pixels_thread[i]);
	// 	pixels += pixels_thread[i];
	// }
    // printf("All threads done, write image.\n");
    /* draw and cleanup */
    write_png(filename, iters, width, height, image);
    free(image);

#ifdef WTIME
    ///////////// End time //////////////////
   clock_gettime(CLOCK_MONOTONIC, &end);
   if ((end.tv_nsec - start.tv_nsec) < 0) {
       temp.tv_sec = end.tv_sec-start.tv_sec-1;
       temp.tv_nsec = 1000000000 + end.tv_nsec - start.tv_nsec;
   } else {
       temp.tv_sec = end.tv_sec - start.tv_sec;
       temp.tv_nsec = end.tv_nsec - start.tv_nsec;
   }
   time_used = temp.tv_sec + (double) temp.tv_nsec / 1000000000.0;
  
   printf("Took %.5f second\n", time_used);
#endif

}
