#include<string.h>
#include <immintrin.h>
#include <cstdlib> // for malloc and free
 
void singleThread(register int input_row,
                          register int input_col,
                          register int *input,
                          register int kernel_row,
                          register int kernel_col,
                          register int *kernel,
                          register int output_row,
                          register int output_col,
                          register long long unsigned int *output)
{
    register int* temp = (int*)malloc(sizeof(int) * output_row * output_col);

    // int kernel_last_rows = output_row - kernel_row ;
    register int kernel_last_cols = output_col - kernel_col ;

    register int i;
    for (i = 0; i < (output_row * output_col); ++i)
        temp[i] = 0;
        // output[i] = 0;


    register int temp2, input_i, output_i, output_j, kernel_i, kernel_j, input_j, input_idx;

    if(kernel_col>3 && kernel_row>3)
{
    // Performing convolution using SIMD instructions
    for (kernel_i = 0; kernel_i < kernel_row; kernel_i++) {
        for (kernel_j = 0; kernel_j < kernel_col; kernel_j++) {
            __m256i kernel_vector = _mm256_set1_epi32(kernel[kernel_i * kernel_col + kernel_j]);

            for (output_i = 0; output_i < output_row; output_i++) {
                // int input_i = (output_i + 2 * kernel_i) % input_row;
                input_i = (output_i + kernel_i+ kernel_i) % input_row;

                input_idx = input_i * input_col;

                for (output_j = 0; output_j < kernel_last_cols; output_j += 8) {
                    // int input_j = (output_j + 2 * kernel_j);
                    input_j = (output_j + kernel_j + kernel_j);

                    __m256i output_vector = _mm256_loadu_si256((__m256i *)(input + input_idx + input_j)); 

                    __m256i required_vector = _mm256_mullo_epi32(kernel_vector, output_vector);
                    required_vector = _mm256_add_epi32(required_vector, _mm256_loadu_si256((__m256i*)(temp + output_i*output_col + output_j)));
                    _mm256_storeu_si256((__m256i*)(temp + output_i*output_col + output_j), required_vector);

                }
            }
        }
    }

    for (i = 0; i <  output_row* output_col; ++i) {
        output[i] = temp[i];
    }

    register int temp3;
    // register int temp2, input_i, output_i, output_j, kernel_i, kernel_j, input_j;
    for(output_i = 0; output_i< output_row; output_i++)
    {
        for(output_j = kernel_last_cols; output_j< output_col; output_j++)
        {
            output[output_i * output_col + output_j]=0;
            for(kernel_i = 0; kernel_i< kernel_row; kernel_i++)
            {
                temp2 = kernel_i + kernel_i+output_i;

                // input_i = temp2 % input_row;
                if (temp2<input_row){
                    input_i = temp2;
                }
                else{
                    input_i = temp2 - input_row;
                }

                for(kernel_j = 0; kernel_j< kernel_col; kernel_j++)
                {
                    temp3 = output_j + kernel_j+ kernel_j;

                    // input_j = temp3 % input_col;
                    if (temp3<input_col){
                        input_j = temp3;
                    }
                    else{
                        input_j = temp3 - input_col;
                    }

                    output[output_i * output_col + output_j] += input[input_i*input_col +input_j] 
                                                                * kernel[kernel_i*kernel_col +kernel_j];

                }
            }
        }
    }
    free(temp);
}   

    else{
    register int output_i, kernel_i, input_i, input_offset, kernel_j, kernel_val, output_j, output_index_1, output_index_2, output_index_3, output_index_4, input_j_1, input_j_2, input_j_3, input_j_4, temp, temp2, temp3;

    for (output_i = 0; output_i < output_row; output_i++) {
        for (kernel_i = 0; kernel_i < kernel_row; kernel_i++) {
            

            temp3 = output_i + kernel_i + kernel_i;

            if (temp3<input_row){
                input_i = temp3;
            }
            else{
                input_i = temp3 - input_row;
            }

            input_offset = input_i * input_col;
 
            for (kernel_j = 0; kernel_j < kernel_col; kernel_j++) {
                kernel_val = kernel[kernel_i * kernel_col + kernel_j];
                temp = kernel_j + kernel_j;

                for (output_j = 0; output_j < output_col; output_j += 4) {
                    output_index_1 = output_i * output_col + output_j;
                    output_index_2 = output_index_1 + 1;
                    output_index_3 = output_index_1 + 2;
                    output_index_4 = output_index_1 + 3;
 

                    temp2 = output_j + temp;

                    if (temp2<input_col){
                        input_j_1 = temp2;
                    }
                    else{
                        input_j_1 = temp2 - input_col;
                    }
                   
                    if ((temp2+1)<input_col){
                        input_j_2 = temp2+1;
                    }
                    else{
                        input_j_2 = temp2 - input_col + 1;
                    }


                    if ((temp2+2)<input_col){
                        input_j_3 = temp2+2;
                    }
                    else{
                        input_j_3 = temp2 - input_col + 2;
                    }


                    if ((temp2+3)<input_col){
                        input_j_4 = temp2+3;
                    }
                    else{
                        input_j_4 = temp2 - input_col + 3;
                    }
 
                    output[output_index_1] += input[input_offset + input_j_1] * kernel_val;

                    if(output_j+1>=output_col) continue;  
                    output[output_index_2] += input[input_offset + input_j_2] * kernel_val;
                    if(output_j+2>=output_col) continue;
                    output[output_index_3] += input[input_offset + input_j_3] * kernel_val;
                    if(output_j+3>=output_col) continue;
                    output[output_index_4] += input[input_offset + input_j_4] * kernel_val;
                }
            }
        }
    }
}
}




