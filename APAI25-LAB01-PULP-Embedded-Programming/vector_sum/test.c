/* 
 * Copyright (C) 2021 University of Bologna
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD license.  See the LICENSE file for details.
 *
 * Authors: Manuele Rusci 	  UniBO (manuele.rusci@unibo.it)
 * 			Lorenzo Lamberti  UniBO (lorenzo.lamberti@unibo.it)
 */

#include "pmsis.h"

#define N 350  // array size
//unsigned char array_1[N]; 	// define array of size N. 	[important note]: "char" data type means its elements are 8 bit! (range from 0 to 255) 
int array_1[N];   // Using int instead of unsigned char

//intialize an array A_ar with _size_ number of elements 
//and precompute the result (we use this as a checksum)
//int init_array(unsigned char * A_ar, int size)
int init_array(int * A_ar, int size)
{
	for(int i=0;i<size;i++) 
		A_ar[i] = i+1;	// this can be changed to whatever pattern
	
	return (size * (size + 1)) >> 1; //this computes the sum as the vector_sum() function. only needed as a checksum.
}

// print array elements
void print_array(int * A_ar, int size) //again instead of using unsigned char, I'm using int.
{
	for(int i=0;i<size;i++) 
		printf("%d ", A_ar[i]); // print array element
	printf("\n");

}

// return the index of the item==element of the array A_ar with size number of elements
int vector_sum(int * A_ar, int size){ //Same thing in here, again instead of using unsigned char, I'm using int.
	int sum = 0;	// initialize accumulator to 0

	// perform the sum
	for(int i=0; i<size; i++){
		sum += A_ar[i];
	}
	return sum;
}

int main()
{
	printf("\n\nThis example program computes the sum of %d integer numbers!\n", N);

	// initialize the array and get the expected result
	int expected_result = init_array(array_1, N);
	
	// inspect the array. let's print its values !
	print_array(array_1, N);

    // call the vector_sum function
    int computed_result = vector_sum(array_1, N);

	// check if the results is correct
    if (computed_result == expected_result)
    	printf("Result is correct. Got %d\n\n", computed_result);
    else
    	printf("Result is not correct. Got %d instead of %d\n\n", computed_result, expected_result);
}