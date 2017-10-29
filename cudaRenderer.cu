#include <string>
#include <algorithm>
#include <math.h> 
#include <stdio.h>
#include <vector>
#include <thrust/scan.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include "cudaRenderer.h"
#include "image.h"
#include "noise.h"
#include "sceneLoader.h"
#include "util.h"

#define SCAN_BLOCK_DIM   256  // needed by sharedMemExclusiveScan implementation
#include "exclusiveScan.cu_inl"


#define NUM_BANKS 16
#define LOG_NUM_BANKS 4
#ifdef ZERO_BANK_CONFLICTS
#define CONFLICT_FREE_OFFSET(n) ((n) >> NUM_BANKS + (n) >> (2 * LOG_NUM_BANKS)) 
#else
#define CONFLICT_FREE_OFFSET(n) ((n) >> LOG_NUM_BANKS)
#endif


////////////////////////////////////////////////////////////////////////////////////////
// Putting all the cuda kernels here
///////////////////////////////////////////////////////////////////////////////////////

struct GlobalConstants {

    SceneName sceneName;

    int numCircles;
    float* position;
    float* velocity;
    float* color;
    float* radius;
    int imageWidth;
    int imageHeight;
    float* imageData;
};

// Global variable that is in scope, but read-only, for all cuda
// kernels.  The __constant__ modifier designates this variable will
// be stored in special "constant" memory on the GPU. (we didn't talk
// about this type of memory in class, but constant memory is a fast
// place to put read-only variables).
__constant__ GlobalConstants cuConstRendererParams;

// read-only lookup tables used to quickly compute noise (needed by
// advanceAnimation for the snowflake scene)
__constant__ int    cuConstNoiseYPermutationTable[256];
__constant__ int    cuConstNoiseXPermutationTable[256];
__constant__ float  cuConstNoise1DValueTable[256];

// color ramp table needed for the color ramp lookup shader
#define COLOR_MAP_SIZE 5
__constant__ float  cuConstColorRamp[COLOR_MAP_SIZE][3];


// including parts of the CUDA code from external files to keep this
// file simpler and to seperate code that should not be modified
#include "noiseCuda.cu_inl"
#include "lookupColor.cu_inl"


// kernelClearImageSnowflake -- (CUDA device code)
//
// Clear the image, setting the image to the white-gray gradation that
// is used in the snowflake image
__global__ void kernelClearImageSnowflake() {

    int imageX = blockIdx.x * blockDim.x + threadIdx.x;
    int imageY = blockIdx.y * blockDim.y + threadIdx.y;

    int width = cuConstRendererParams.imageWidth;
    int height = cuConstRendererParams.imageHeight;

    if (imageX >= width || imageY >= height)
        return;

    int offset = 4 * (imageY * width + imageX);
    float shade = .4f + .45f * static_cast<float>(height-imageY) / height;
    float4 value = make_float4(shade, shade, shade, 1.f);

    // write to global memory: As an optimization, I use a float4
    // store, that results in more efficient code than if I coded this
    // up as four seperate fp32 stores.
    *(float4*)(&cuConstRendererParams.imageData[offset]) = value;
}

// kernelClearImage --  (CUDA device code)
//
// Clear the image, setting all pixels to the specified color rgba
__global__ void kernelClearImage(float r, float g, float b, float a) {

    int imageX = blockIdx.x * blockDim.x + threadIdx.x;
    int imageY = blockIdx.y * blockDim.y + threadIdx.y;

    int width = cuConstRendererParams.imageWidth;
    int height = cuConstRendererParams.imageHeight;

    if (imageX >= width || imageY >= height)
        return;

    int offset = 4 * (imageY * width + imageX);
    float4 value = make_float4(r, g, b, a);

    // write to global memory: As an optimization, I use a float4
    // store, that results in more efficient code than if I coded this
    // up as four seperate fp32 stores.
    *(float4*)(&cuConstRendererParams.imageData[offset]) = value;
}

// kernelAdvanceFireWorks
//
// Update the position of the fireworks (if circle is firework)
__global__ void kernelAdvanceFireWorks() {
    const float dt = 1.f / 60.f;
    const float pi = 3.14159;
    const float maxDist = 0.25f;

    float* velocity = cuConstRendererParams.velocity;
    float* position = cuConstRendererParams.position;
    float* radius = cuConstRendererParams.radius;

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= cuConstRendererParams.numCircles)
        return;

    if (0 <= index && index < NUM_FIREWORKS) { // firework center; no update
        return;
    }

    // determine the fire-work center/spark indices
    int fIdx = (index - NUM_FIREWORKS) / NUM_SPARKS;
    int sfIdx = (index - NUM_FIREWORKS) % NUM_SPARKS;

    int index3i = 3 * fIdx;
    int sIdx = NUM_FIREWORKS + fIdx * NUM_SPARKS + sfIdx;
    int index3j = 3 * sIdx;

    float cx = position[index3i];
    float cy = position[index3i+1];

    // update position
    position[index3j] += velocity[index3j] * dt;
    position[index3j+1] += velocity[index3j+1] * dt;

    // fire-work sparks
    float sx = position[index3j];
    float sy = position[index3j+1];

    // compute vector from firework-spark
    float cxsx = sx - cx;
    float cysy = sy - cy;

    // compute distance from fire-work
    float dist = sqrt(cxsx * cxsx + cysy * cysy);
    if (dist > maxDist) { // restore to starting position
        // random starting position on fire-work's rim
        float angle = (sfIdx * 2 * pi)/NUM_SPARKS;
        float sinA = sin(angle);
        float cosA = cos(angle);
        float x = cosA * radius[fIdx];
        float y = sinA * radius[fIdx];

        position[index3j] = position[index3i] + x;
        position[index3j+1] = position[index3i+1] + y;
        position[index3j+2] = 0.0f;

        // travel scaled unit length
        velocity[index3j] = cosA/5.0;
        velocity[index3j+1] = sinA/5.0;
        velocity[index3j+2] = 0.0f;
    }
}

// kernelAdvanceHypnosis
//
// Update the radius/color of the circles
__global__ void kernelAdvanceHypnosis() {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= cuConstRendererParams.numCircles)
        return;

    float* radius = cuConstRendererParams.radius;

    float cutOff = 0.5f;
    // place circle back in center after reaching threshold radisus
    if (radius[index] > cutOff) {
        radius[index] = 0.02f;
    } else {
        radius[index] += 0.01f;
    }
}


// kernelAdvanceBouncingBalls
//
// Update the positino of the balls
__global__ void kernelAdvanceBouncingBalls() {
    const float dt = 1.f / 60.f;
    const float kGravity = -2.8f; // sorry Newton
    const float kDragCoeff = -0.8f;
    const float epsilon = 0.001f;

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= cuConstRendererParams.numCircles)
        return;

    float* velocity = cuConstRendererParams.velocity;
    float* position = cuConstRendererParams.position;

    int index3 = 3 * index;
    // reverse velocity if center position < 0
    float oldVelocity = velocity[index3+1];
    float oldPosition = position[index3+1];

    if (oldVelocity == 0.f && oldPosition == 0.f) { // stop-condition
        return;
    }

    if (position[index3+1] < 0 && oldVelocity < 0.f) { // bounce ball
        velocity[index3+1] *= kDragCoeff;
    }

    // update velocity: v = u + at (only along y-axis)
    velocity[index3+1] += kGravity * dt;

    // update positions (only along y-axis)
    position[index3+1] += velocity[index3+1] * dt;

    if (fabsf(velocity[index3+1] - oldVelocity) < epsilon
        && oldPosition < 0.0f
        && fabsf(position[index3+1]-oldPosition) < epsilon) { // stop ball
        velocity[index3+1] = 0.f;
        position[index3+1] = 0.f;
    }
}

// kernelAdvanceSnowflake -- (CUDA device code)
//
// move the snowflake animation forward one time step.  Updates circle
// positions and velocities.  Note how the position of the snowflake
// is reset if it moves off the left, right, or bottom of the screen.
__global__ void kernelAdvanceSnowflake() {

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= cuConstRendererParams.numCircles)
        return;

    const float dt = 1.f / 60.f;
    const float kGravity = -1.8f; // sorry Newton
    const float kDragCoeff = 2.f;

    int index3 = 3 * index;

    float* positionPtr = &cuConstRendererParams.position[index3];
    float* velocityPtr = &cuConstRendererParams.velocity[index3];

    // loads from global memory
    float3 position = *((float3*)positionPtr);
    float3 velocity = *((float3*)velocityPtr);

    // hack to make farther circles move more slowly, giving the
    // illusion of parallax
    float forceScaling = fmin(fmax(1.f - position.z, .1f), 1.f); // clamp

    // add some noise to the motion to make the snow flutter
    float3 noiseInput;
    noiseInput.x = 10.f * position.x;
    noiseInput.y = 10.f * position.y;
    noiseInput.z = 255.f * position.z;
    float2 noiseForce = cudaVec2CellNoise(noiseInput, index);
    noiseForce.x *= 7.5f;
    noiseForce.y *= 5.f;

    // drag
    float2 dragForce;
    dragForce.x = -1.f * kDragCoeff * velocity.x;
    dragForce.y = -1.f * kDragCoeff * velocity.y;

    // update positions
    position.x += velocity.x * dt;
    position.y += velocity.y * dt;

    // update velocities
    velocity.x += forceScaling * (noiseForce.x + dragForce.y) * dt;
    velocity.y += forceScaling * (kGravity + noiseForce.y + dragForce.y) * dt;

    float radius = cuConstRendererParams.radius[index];

    // if the snowflake has moved off the left, right or bottom of
    // the screen, place it back at the top and give it a
    // pseudorandom x position and velocity.
    if ( (position.y + radius < 0.f) ||
         (position.x + radius) < -0.f ||
         (position.x - radius) > 1.f)
    {
        noiseInput.x = 255.f * position.x;
        noiseInput.y = 255.f * position.y;
        noiseInput.z = 255.f * position.z;
        noiseForce = cudaVec2CellNoise(noiseInput, index);

        position.x = .5f + .5f * noiseForce.x;
        position.y = 1.35f + radius;

        // restart from 0 vertical velocity.  Choose a
        // pseudo-random horizontal velocity.
        velocity.x = 2.f * noiseForce.y;
        velocity.y = 0.f;
    }

    // store updated positions and velocities to global memory
    *((float3*)positionPtr) = position;
    *((float3*)velocityPtr) = velocity;
}

// shadePixel -- (CUDA device code)
//
// given a pixel and a circle, determines the contribution to the
// pixel from the circle.  Update of the image is done in this
// function.  Called by kernelRenderCircles()
__device__ __inline__ void
shadePixel(int circleIndex, float2 pixelCenter, float3 p, float4* imagePtr) {

    float diffX = p.x - pixelCenter.x;
    float diffY = p.y - pixelCenter.y;
    float pixelDist = diffX * diffX + diffY * diffY;

    float rad = cuConstRendererParams.radius[circleIndex];;
    float maxDist = rad * rad;

    // circle does not contribute to the image
    if (pixelDist > maxDist)
        return;

    float3 rgb;
    float alpha;

    // there is a non-zero contribution.  Now compute the shading value

    // This conditional is in the inner loop, but it evaluates the
    // same direction for all threads so it's cost is not so
    // bad. Attempting to hoist this conditional is not a required
    // student optimization in Assignment 2

    if (cuConstRendererParams.sceneName == SNOWFLAKES || cuConstRendererParams.sceneName == SNOWFLAKES_SINGLE_FRAME) {

        const float kCircleMaxAlpha = .5f;
        const float falloffScale = 4.f;

        float normPixelDist = sqrt(pixelDist) / rad;
        rgb = lookupColor(normPixelDist);

        float maxAlpha = .6f + .4f * (1.f-p.z);
        maxAlpha = kCircleMaxAlpha * fmaxf(fminf(maxAlpha, 1.f), 0.f); // kCircleMaxAlpha * clamped value
        alpha = maxAlpha * exp(-1.f * falloffScale * normPixelDist * normPixelDist);

    } else {
        // simple: each circle has an assigned color
        int index3 = 3 * circleIndex;
        rgb = *(float3*)&(cuConstRendererParams.color[index3]);
        alpha = .5f;
    }

    float oneMinusAlpha = 1.f - alpha;


    // BEGIN SHOULD-BE-ATOMIC REGION
    // global memory read


    float4 existingColor = *imagePtr;
    float4 newColor;
    newColor.x = alpha * rgb.x + oneMinusAlpha * existingColor.x;
    newColor.y = alpha * rgb.y + oneMinusAlpha * existingColor.y;
    newColor.z = alpha * rgb.z + oneMinusAlpha * existingColor.z;
    newColor.w = alpha + existingColor.w;

    // global memory write
    *imagePtr = newColor;

    // END SHOULD-BE-ATOMIC REGION

}

/*
// kernelRenderCircles -- (CUDA device code)
//
// Each thread renders a circle.  Since there is no protection to
// ensure order of update or mutual exclusion on the output image, the
// resulting image will be incorrect.  
__global__ void kernelRenderCircles() {
    //int table[1024][1024] = {0};
    __shared__ int table[1024][1024];
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    //__shared__ float shmImgPtr[256][180];
    if (index >= cuConstRendererParams.numCircles)
        return;

    int index3 = 3 * index;

    // read position and radius
    float3 p = *(float3*)(&cuConstRendererParams.position[index3]);
    float  rad = cuConstRendererParams.radius[index];

    // compute the bounding box of the circle. The bound is in integer
    // screen coordinates, so it's clamped to the edges of the screen.
    short imageWidth = cuConstRendererParams.imageWidth;
    short imageHeight = cuConstRendererParams.imageHeight;
    short minX = static_cast<short>(imageWidth * (p.x - rad));
    short maxX = static_cast<short>(imageWidth * (p.x + rad)) + 1;
    short minY = static_cast<short>(imageHeight * (p.y - rad));
    short maxY = static_cast<short>(imageHeight * (p.y + rad)) + 1;

    // a bunch of clamps.  Is there a CUDA built-in for this?
    short screenMinX = (minX > 0) ? ((minX < imageWidth) ? minX : imageWidth) : 0;
    short screenMaxX = (maxX > 0) ? ((maxX < imageWidth) ? maxX : imageWidth) : 0;
    short screenMinY = (minY > 0) ? ((minY < imageHeight) ? minY : imageHeight) : 0;
    short screenMaxY = (maxY > 0) ? ((maxY < imageHeight) ? maxY : imageHeight) : 0;

    float invWidth = 1.f / imageWidth;
    float invHeight = 1.f / imageHeight;

    //printf("screenMaxX - screenMinX: %d\n", screenMaxX- screenMinX);
    // for all pixels in the bonding box

    for (int pixelY=screenMinY; pixelY<screenMaxY; pixelY++) {
	for (int pixelX=screenMinX; pixelX<screenMaxX; pixelX++) {
	    float2 pixelCenterNorm = make_float2(invWidth * (static_cast<float>(pixelX) + 0.5f),
						 invHeight * (static_cast<float>(pixelY) + 0.5f));
	    float diffX = p.x - pixelCenterNorm.x;
	    float diffY = p.y - pixelCenterNorm.y;
	    float pixelDist = diffX * diffX + diffY * diffY;

	    float rad = cuConstRendererParams.radius[index];;
	    float maxDist = rad * rad;

	    // circle does not contribute to the image
	    if (pixelDist <= maxDist)
		table[pixelX][pixelY]++;
	    //shadePixel(index, pixelCenterNorm, p, imgPtr,);//&shmImgPtr[threadIdx.x][4 * a]);
	    //imgPtr++;
	}
    }
}
*/
////////////////////////////////////////////////////////////////////////////////////////
__device__ void prescan(uint *g_odata, uint *g_idata, int n)
{
	__shared__ uint temp[512];// allocated on invocation
	int thid = threadIdx.x;
	int offset = 1;

	int ai = thid;
	int bi = thid + (n/2);
	int bankOffsetA = CONFLICT_FREE_OFFSET(ai);
	int bankOffsetB = CONFLICT_FREE_OFFSET(ai);
	temp[ai + bankOffsetA] = g_idata[ai];
	temp[bi + bankOffsetB] = g_idata[bi]; 



	for (int d = n>>1; d > 0; d >>= 1) // build sum in place up the tree
	{
 		__syncthreads();
		if (thid < d)
		{ 
			int ai = offset*(2*thid+1)-1;
			int bi = offset*(2*thid+2)-1;
			ai += CONFLICT_FREE_OFFSET(ai);
			bi += CONFLICT_FREE_OFFSET(bi);

			temp[bi] += temp[ai];
		}
		offset *= 2;
	} 

	if (thid==0) {
		//temp[n – 1 /*+ CONFLICT_FREE_OFFSET(n - 1)*/ ] = 0;
		temp[n - 1 + CONFLICT_FREE_OFFSET(n - 1)] = 0;
	}
	for (int d = 1; d < n; d *= 2) // traverse down tree & build scan
	{
		offset >>= 1;
		__syncthreads();
		if (thid < d)
		{ 
			int ai = offset*(2*thid+1)-1;
			int bi = offset*(2*thid+2)-1;
			ai += CONFLICT_FREE_OFFSET(ai);
			bi += CONFLICT_FREE_OFFSET(bi); 


			float t = temp[ai];
			temp[ai] = temp[bi];
			temp[bi] += t;
		}
	}
 	__syncthreads(); 


	g_odata[ai] = temp[ai + bankOffsetA];
	g_odata[bi] = temp[bi + bankOffsetB]; 
} 





__global__ void kernelRenderCircles() {
    /* this queue is intended for remembering circle index */
    int queue[50];
    int queueIndex = 0;

    /* These sharemd memory array will be used in prefixSum function */
    __shared__ uint shmQueue[256];	//input of prefixSum : the role of this array is to divide index of order[] array
    __shared__ uint prefixSum[256];	//output of prefixSum
    __shared__ uint prefixSumScratch[2 * 256];	//The comments inside a prefixSum library file says we need this to calculate it


    /* This array contains circle indices that is colored inside a threa block boundary(32 x 32 pixels),
       and they are sorted by ascending order */
    __shared__ int order[3000];


    /* Statement shown in line 542(extern keyword) used for dynamic allocation of shared memory.
       Reducing the size of shared memory array has positive impact on the execution time.
       From the fact that each image(e.g., rgb, littlebig, rand10k, ...) needs different array size,
       I tried to allocate different array size according to image(e.g., rgb, littlebing, ...),
       but when I use it, it gives me wrong result. I don't know why. */

    //extern __shared__ int order[];

    int blockThreadIndex = blockDim.x * threadIdx.y + threadIdx.x; 

    int numCircles = cuConstRendererParams.numCircles;
    int threadsPerBlock = blockDim.x * blockDim.y;

    /* each thread will handle the number of circles stored in variable 'circle' */ 
    int circle = (numCircles + threadsPerBlock - 1) / threadsPerBlock;


    /* imageX and imageY are the location of image pixels assigned for this thread within boundary. */
    //int imageX = blockIdx.x * blockDim.x + threadIdx.x; // This is intended for assiging each thread 1x1 pixel.
    //int imageY = blockIdx.y * blockDim.y + threadIdx.y; 

    /*Each thread will deal with 2x2 pixels, not 1x1 pixel by multiplying 2.*/
    int imageX = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    int imageY = (blockIdx.y * blockDim.y + threadIdx.y) * 2;

    short imageWidth = cuConstRendererParams.imageWidth;
    short imageHeight = cuConstRendererParams.imageHeight; 


    /* Thess variables describe pixel boundary of thread block. */
    
    //int pixelXFrom = blockDim.x * blockIdx.x;	//e.g., 0, 16, 32, ...
    //int pixelXTo = blockDim.x * (blockIdx.x + 1) - 1;	// 15, 31, 63, ...
    //int pixelYFrom = blockDim.y * blockIdx.y;
    //int pixelYTo = blockDim.y * (blockIdx.y + 1) - 1;

    /* Number 2 is intended for 32 x 32 pixels, not 16 x 16 pixels. */
    int pixelXFrom = blockDim.x * blockIdx.x * 2;	//e.g., 0, 64, 128, ...
    int pixelXTo = 2 * blockDim.x * (blockIdx.x + 1) - 1;	// 63, 127, 255, ...
    int pixelYFrom = blockDim.y * blockIdx.y * 2;
    int pixelYTo = 2 * blockDim.y * (blockIdx.y + 1) - 1;

    float invWidth = 1.f / imageWidth;
    float invHeight = 1.f / imageHeight;

    /* each thread only handles their pixel boundary(2 x 2 pixels),
       and these are used to copy global memory data into local memory. */
    float4 *imgPtr0 = (float4*)(&cuConstRendererParams.imageData[4 * (imageY * imageWidth + imageX)]);
    float4 *imgPtr1 = (float4*)(&cuConstRendererParams.imageData[4 * (imageY * imageWidth + imageX + 1)]);
    float4 *imgPtr2 = (float4*)(&cuConstRendererParams.imageData[4 * ((imageY + 1) * imageWidth + imageX)]);
    float4 *imgPtr3 = (float4*)(&cuConstRendererParams.imageData[4 * ((imageY + 1) * imageWidth + imageX + 1)]);
/*
    float4 *imgPtr4 = (float4*)(&cuConstRendererParams.imageData[4 * ((imageY + 1) * imageWidth + imageX)]);
    float4 *imgPtr5 = (float4*)(&cuConstRendererParams.imageData[4 * ((imageY + 1) * imageWidth + imageX + 1)]);
    float4 *imgPtr6 = (float4*)(&cuConstRendererParams.imageData[4 * ((imageY + 1) * imageWidth + imageX + 2)]);
    float4 *imgPtr7 = (float4*)(&cuConstRendererParams.imageData[4 * ((imageY + 1) * imageWidth + imageX + 3)]);
    float4 *imgPtr8 = (float4*)(&cuConstRendererParams.imageData[4 * ((imageY + 2)* imageWidth + imageX)]);
    float4 *imgPtr9 = (float4*)(&cuConstRendererParams.imageData[4 * ((imageY + 2) * imageWidth + imageX + 1)]);
    float4 *imgPtr10 = (float4*)(&cuConstRendererParams.imageData[4 * ((imageY + 2) * imageWidth + imageX + 2)]);
    float4 *imgPtr11 = (float4*)(&cuConstRendererParams.imageData[4 * ((imageY + 2)* imageWidth + imageX + 3)]);
    float4 *imgPtr12 = (float4*)(&cuConstRendererParams.imageData[4 * ((imageY + 3) * imageWidth + imageX)]);
    float4 *imgPtr13 = (float4*)(&cuConstRendererParams.imageData[4 * ((imageY + 3) * imageWidth + imageX + 1)]);
    float4 *imgPtr14 = (float4*)(&cuConstRendererParams.imageData[4 * ((imageY + 3)* imageWidth + imageX + 2)]);
    float4 *imgPtr15 = (float4*)(&cuConstRendererParams.imageData[4 * ((imageY + 3)* imageWidth + imageX + 3)]);
*/
    /* Copy rgb data in global memory into local memory */
    float4 localImgData0 = *imgPtr0;
    float4 localImgData1 = *imgPtr1;
    float4 localImgData2 = *imgPtr2;
    float4 localImgData3 = *imgPtr3;
/*
    float4 localImgData4 = *imgPtr4;
    float4 localImgData5 = *imgPtr5;
    float4 localImgData6 = *imgPtr6;
    float4 localImgData7 = *imgPtr7;
    float4 localImgData8 = *imgPtr8;
    float4 localImgData9 = *imgPtr9;
    float4 localImgData10 = *imgPtr10;
    float4 localImgData11 = *imgPtr11;
    float4 localImgData12 = *imgPtr12;
    float4 localImgData13 = *imgPtr13;
    float4 localImgData14 = *imgPtr14;
    float4 localImgData15 = *imgPtr15;

*/
    /* Each thread deals with circle indices(From and To) shown in below to
       check whether they are within or across the boundary of this thread block */
    /* When there exist only three circles to be drawn, then each thread has variable
       circleIndexFrom: 0, 1, 2, 3, ... , circleIndexTo: 0, 1, 2, 3, ... , which means
       , in this case, thread number from 3 to 255 will execute for loop described in below.
       However, it doesn't matter because variable "p" and "rad"(in for looop) will have zero valuee */

    int circleIndexFrom = blockThreadIndex * circle;
    int circleIndexTo = (blockThreadIndex + 1) * circle - 1;

    for (int i = circleIndexFrom; i <= circleIndexTo; i++) {
	int index3 = 3 * i;
	float3 p = *(float3*)(&cuConstRendererParams.position[index3]);
	float rad = cuConstRendererParams.radius[i];
	//float newRadWidth = rad * imageWidth;
	//float newRadHeight = rad * imageHeight;	

	/* "rad" is normalized to 0 ~ 1023.xxxxxx */
	float extendXLeft = pixelXFrom - (rad * imageWidth);
	float extendXRight = pixelXTo + (rad * imageWidth);
	float extendYTop = pixelYFrom - (rad * imageHeight);
	float extendYBottom = pixelYTo + (rad * imageHeight);
        /* "circle coordinate" is normailzed to 0 ~ 1023.xxxxxx */
	float circleX = p.x * imageWidth;
	float circleY = p.y * imageHeight;

	
	/* This will check whether the circle index "i" exist within or across the boundary of this thread block's pixels */
	/* Multiplying the value 1.01 and 0.99 is very important to work correctly,
	   Due to the small error from above(maybe the gap between normalized value(~1023) and floating value(0.xxx),
	   I have to multply these constant, it is similar to extend the boundary of thread block's pixel */
	/* I found this fact unexpectedly, because some of the results show me "correctness failed", others "correctness pass" */
	if (extendXLeft <= circleX * 1.01  && extendXRight >= circleX * 0.99 && extendYTop <= circleY * 1.01 && extendYBottom >= circleY * 0.99) {
		queue[queueIndex++] = i;
	}
    }

    shmQueue[blockThreadIndex] = queueIndex;
    __syncthreads();

    /* "prescan" is prefixSum algorithm providied by nVidia. I tried to use this to get
       fast execution time, but failed to get correct result. Maybe I missed something. */
    //prescan(prefixSum, shmQueue, 256);
    //__syncthreads();
    
    /* All threads, together,  in this thread block will calculate prefixSum. */
    sharedMemExclusiveScan(blockThreadIndex, shmQueue, prefixSum, prefixSumScratch, 256);
    __syncthreads();

    /* We have to guarantee that all threads must be located at this point. This is because
       if some of threads are still in shareMemExclusiveScan, which means
       they are still calculating prefixSum, other threads that is executing below code will
       get incorrect value of prefixSum[255] */

    int globalIndex = prefixSum[255] + shmQueue[255];

    int start = prefixSum[blockThreadIndex];
    int end = start + shmQueue[blockThreadIndex];

    //int start = (blockThreadIndex == 0) ? 0 : prefixSum[blockThreadIndex - 1];
    //int end =prefixSum[blockThreadIndex];
    

    int localIndex = 0;

    for (int i = start; i < end; i++) {
	order[i] = queue[localIndex++];
    }
    __syncthreads();
   

    /* Loop circle indices that are stored in shared memory array "order[]" */
    for (int i= 0 ; i < globalIndex; i++) {
	int a = order[i];
	int index3 = 3 * a;
	float3 p = *(float3*)(&cuConstRendererParams.position[index3]);

        /* calculate center point of each pixel which is manged by a thread */
	float2 pixelCenterNorm0 = make_float2(invWidth * (static_cast<float>(imageX) + 0.5f),
			invHeight * (static_cast<float>(imageY) + 0.5f));
	float2 pixelCenterNorm1 = make_float2(invWidth * (static_cast<float>(imageX + 1) + 0.5f),
			invHeight * (static_cast<float>(imageY) + 0.5f));
	float2 pixelCenterNorm2 = make_float2(invWidth * (static_cast<float>(imageX) + 0.5f),
			invHeight * (static_cast<float>(imageY+ 1) + 0.5f));
	float2 pixelCenterNorm3 = make_float2(invWidth * (static_cast<float>(imageX + 1) + 0.5f),
			invHeight * (static_cast<float>(imageY + 1) + 0.5f));
/*
	float2 pixelCenterNorm4 = make_float2(invWidth * (static_cast<float>(imageX) + 0.5f),
			invHeight * (static_cast<float>(imageY + 1) + 0.5f));
	float2 pixelCenterNorm5 = make_float2(invWidth * (static_cast<float>(imageX + 1) + 0.5f),
			invHeight * (static_cast<float>(imageY + 1) + 0.5f));
	float2 pixelCenterNorm6 = make_float2(invWidth * (static_cast<float>(imageX + 2) + 0.5f),
			invHeight * (static_cast<float>(imageY + 1) + 0.5f));
	float2 pixelCenterNorm7 = make_float2(invWidth * (static_cast<float>(imageX + 3) + 0.5f),
			invHeight * (static_cast<float>(imageY + 1) + 0.5f));
	float2 pixelCenterNorm8 = make_float2(invWidth * (static_cast<float>(imageX) + 0.5f),
			invHeight * (static_cast<float>(imageY + 2) + 0.5f));
	float2 pixelCenterNorm9 = make_float2(invWidth * (static_cast<float>(imageX + 1) + 0.5f),
			invHeight * (static_cast<float>(imageY + 2) + 0.5f));
	float2 pixelCenterNorm10 = make_float2(invWidth * (static_cast<float>(imageX + 2) + 0.5f),
			invHeight * (static_cast<float>(imageY + 2) + 0.5f));
	float2 pixelCenterNorm11 = make_float2(invWidth * (static_cast<float>(imageX + 3) + 0.5f),
			invHeight * (static_cast<float>(imageY + 2) + 0.5f));
	float2 pixelCenterNorm12 = make_float2(invWidth * (static_cast<float>(imageX) + 0.5f),
			invHeight * (static_cast<float>(imageY + 3) + 0.5f));
	float2 pixelCenterNorm13 = make_float2(invWidth * (static_cast<float>(imageX + 1) + 0.5f),
			invHeight * (static_cast<float>(imageY + 3) + 0.5f));
	float2 pixelCenterNorm14 = make_float2(invWidth * (static_cast<float>(imageX + 2) + 0.5f),
			invHeight * (static_cast<float>(imageY + 3) + 0.5f));
	float2 pixelCenterNorm15 = make_float2(invWidth * (static_cast<float>(imageX + 3) + 0.5f),
			invHeight * (static_cast<float>(imageY + 3) + 0.5f));
*/
	/* each pixel will color RGB in parallel, because each thread has their own range of boundary of pixels */
	shadePixel(a, pixelCenterNorm0, p, &localImgData0);
	shadePixel(a, pixelCenterNorm1, p, &localImgData1);
	shadePixel(a, pixelCenterNorm2, p, &localImgData2);
	shadePixel(a, pixelCenterNorm3, p, &localImgData3);
/*
	shadePixel(a, pixelCenterNorm4, p, &localImgData4);
	shadePixel(a, pixelCenterNorm5, p, &localImgData5);
	shadePixel(a, pixelCenterNorm6, p, &localImgData6);
	shadePixel(a, pixelCenterNorm7, p, &localImgData7);
	shadePixel(a, pixelCenterNorm8, p, &localImgData8);
	shadePixel(a, pixelCenterNorm9, p, &localImgData9);
	shadePixel(a, pixelCenterNorm10, p, &localImgData10);
	shadePixel(a, pixelCenterNorm11, p, &localImgData11);
	shadePixel(a, pixelCenterNorm12, p, &localImgData12);
	shadePixel(a, pixelCenterNorm13, p, &localImgData13);
	shadePixel(a, pixelCenterNorm14, p, &localImgData14);
	shadePixel(a, pixelCenterNorm15, p, &localImgData15);
	//shadePixel(a, pixelCenterNorm2, p, &localImgData2);
	//shadePixel(a, pixelCenterNorm3, p, &localImgData3);
	//shadePixel(a, pixelCenterNorm4, p, &localImgData4);
	//shadePixel(a, pixelCenterNorm, p, &shmImgData[threadIdx.y * 16 + threadIdx.x]);
*/
    }

    /* finally 2x2 pixels' imgData is copied into global memory */
    *imgPtr0 = localImgData0;
    *imgPtr1 = localImgData1;
    *imgPtr2 = localImgData2;
    *imgPtr3 = localImgData3;
/*
    *imgPtr4 = localImgData4;
    *imgPtr5 = localImgData5;
    *imgPtr6 = localImgData6;
    *imgPtr7 = localImgData7;
    *imgPtr8 = localImgData8;
    *imgPtr9 = localImgData9;
    *imgPtr10 = localImgData10;
    *imgPtr11 = localImgData11;
    *imgPtr12 = localImgData12;
    *imgPtr13 = localImgData13;
    *imgPtr14 = localImgData14;
    *imgPtr15 = localImgData15;
*/
}


CudaRenderer::CudaRenderer() {
    image = NULL;

    numCircles = 0;
    position = NULL;
    velocity = NULL;
    color = NULL;
    radius = NULL;

    cudaDevicePosition = NULL;
    cudaDeviceVelocity = NULL;
    cudaDeviceColor = NULL;
    cudaDeviceRadius = NULL;
    cudaDeviceImageData = NULL;
}

CudaRenderer::~CudaRenderer() {

    if (image) {
        delete image;
    }

    if (position) {
        delete [] position;
        delete [] velocity;
        delete [] color;
        delete [] radius;
    }

    if (cudaDevicePosition) {
        cudaFree(cudaDevicePosition);
        cudaFree(cudaDeviceVelocity);
        cudaFree(cudaDeviceColor);
        cudaFree(cudaDeviceRadius);
        cudaFree(cudaDeviceImageData);
    }
}

const Image*
CudaRenderer::getImage() {

    // need to copy contents of the rendered image from device memory
    // before we expose the Image object to the caller

    printf("Copying image data from device\n");

    cudaMemcpy(image->data,
               cudaDeviceImageData,
               sizeof(float) * 4 * image->width * image->height,
               cudaMemcpyDeviceToHost);

    return image;
}

void
CudaRenderer::loadScene(SceneName scene) {
    sceneName = scene;
    loadCircleScene(sceneName, numCircles, position, velocity, color, radius);
}

void
CudaRenderer::setup() {

    int deviceCount = 0;
    std::string name;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    printf("---------------------------------------------------------\n");
    printf("Initializing CUDA for CudaRenderer\n");
    printf("Found %d CUDA devices\n", deviceCount);

    for (int i=0; i<deviceCount; i++) {
        cudaDeviceProp deviceProps;
        cudaGetDeviceProperties(&deviceProps, i);
        name = deviceProps.name;

        printf("Device %d: %s\n", i, deviceProps.name);
        printf("   SMs:        %d\n", deviceProps.multiProcessorCount);
        printf("   Global mem: %.0f MB\n", static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024));
        printf("   CUDA Cap:   %d.%d\n", deviceProps.major, deviceProps.minor);
    }
    printf("---------------------------------------------------------\n");

    // By this time the scene should be loaded.  Now copy all the key
    // data structures into device memory so they are accessible to
    // CUDA kernels
    //
    // See the CUDA Programmer's Guide for descriptions of
    // cudaMalloc and cudaMemcpy

    cudaMalloc(&cudaDevicePosition, sizeof(float) * 3 * numCircles);
    cudaMalloc(&cudaDeviceVelocity, sizeof(float) * 3 * numCircles);
    cudaMalloc(&cudaDeviceColor, sizeof(float) * 3 * numCircles);
    cudaMalloc(&cudaDeviceRadius, sizeof(float) * numCircles);
    cudaMalloc(&cudaDeviceImageData, sizeof(float) * 4 * image->width * image->height);

    cudaMemcpy(cudaDevicePosition, position, sizeof(float) * 3 * numCircles, cudaMemcpyHostToDevice);
    cudaMemcpy(cudaDeviceVelocity, velocity, sizeof(float) * 3 * numCircles, cudaMemcpyHostToDevice);
    cudaMemcpy(cudaDeviceColor, color, sizeof(float) * 3 * numCircles, cudaMemcpyHostToDevice);
    cudaMemcpy(cudaDeviceRadius, radius, sizeof(float) * numCircles, cudaMemcpyHostToDevice);

    // Initialize parameters in constant memory.  We didn't talk about
    // constant memory in class, but the use of read-only constant
    // memory here is an optimization over just sticking these values
    // in device global memory.  NVIDIA GPUs have a few special tricks
    // for optimizing access to constant memory.  Using global memory
    // here would have worked just as well.  See the Programmer's
    // Guide for more information about constant memory.

    GlobalConstants params;
    params.sceneName = sceneName;
    params.numCircles = numCircles;
    params.imageWidth = image->width;
    params.imageHeight = image->height;
    params.position = cudaDevicePosition;
    params.velocity = cudaDeviceVelocity;
    params.color = cudaDeviceColor;
    params.radius = cudaDeviceRadius;
    params.imageData = cudaDeviceImageData;

    cudaMemcpyToSymbol(cuConstRendererParams, &params, sizeof(GlobalConstants));

    // also need to copy over the noise lookup tables, so we can
    // implement noise on the GPU
    int* permX;
    int* permY;
    float* value1D;
    getNoiseTables(&permX, &permY, &value1D);
    cudaMemcpyToSymbol(cuConstNoiseXPermutationTable, permX, sizeof(int) * 256);
    cudaMemcpyToSymbol(cuConstNoiseYPermutationTable, permY, sizeof(int) * 256);
    cudaMemcpyToSymbol(cuConstNoise1DValueTable, value1D, sizeof(float) * 256);

    // last, copy over the color table that's used by the shading
    // function for circles in the snowflake demo

    float lookupTable[COLOR_MAP_SIZE][3] = {
        {1.f, 1.f, 1.f},
        {1.f, 1.f, 1.f},
        {.8f, .9f, 1.f},
        {.8f, .9f, 1.f},
        {.8f, 0.8f, 1.f},
    };

    cudaMemcpyToSymbol(cuConstColorRamp, lookupTable, sizeof(float) * 3 * COLOR_MAP_SIZE);

}

// allocOutputImage --
//
// Allocate buffer the renderer will render into.  Check status of
// image first to avoid memory leak.
void
CudaRenderer::allocOutputImage(int width, int height) {

    if (image)
        delete image;
    image = new Image(width, height);
}

// clearImage --
//
// Clear's the renderer's target image.  The state of the image after
// the clear depends on the scene being rendered.
void
CudaRenderer::clearImage() {

    // 256 threads per block is a healthy number
    dim3 blockDim(16, 16, 1);
    dim3 gridDim(
        (image->width + blockDim.x - 1) / blockDim.x,
        (image->height + blockDim.y - 1) / blockDim.y);

    if (sceneName == SNOWFLAKES || sceneName == SNOWFLAKES_SINGLE_FRAME) {
        kernelClearImageSnowflake<<<gridDim, blockDim>>>();
    } else {
        kernelClearImage<<<gridDim, blockDim>>>(1.f, 1.f, 1.f, 1.f);
    }
    cudaDeviceSynchronize();
}

// advanceAnimation --
//
// Advance the simulation one time step.  Updates all circle positions
// and velocities
void
CudaRenderer::advanceAnimation() {
     // 256 threads per block is a healthy number
    dim3 blockDim(256, 1);
    dim3 gridDim((numCircles + blockDim.x - 1) / blockDim.x);

    // only the snowflake scene has animation
    if (sceneName == SNOWFLAKES) {
        kernelAdvanceSnowflake<<<gridDim, blockDim>>>();
    } else if (sceneName == BOUNCING_BALLS) {
        kernelAdvanceBouncingBalls<<<gridDim, blockDim>>>();
    } else if (sceneName == HYPNOSIS) {
        kernelAdvanceHypnosis<<<gridDim, blockDim>>>();
    } else if (sceneName == FIREWORKS) {
        kernelAdvanceFireWorks<<<gridDim, blockDim>>>();
    }
    cudaDeviceSynchronize();
}

void
CudaRenderer::render() {

    // 256 threads per block is a healthy number
	/*
    dim3 blockDim(256, 1);
    dim3 gridDim((numCircles + blockDim.x - 1) / blockDim.x);
*/
/*
    int size = 2000;
    if (sceneName == CIRCLE_RGB || sceneName == CIRCLE_RGBY)
	size = 300;
    else if (sceneName == CIRCLE_TEST_10K) 
	size = 300;
    else if (sceneName == CIRCLE_TEST_100K)
	size = 1900;
    else
	size = 2800;
   

    printf("before kenrel size: %d\n", size);
*/
    dim3 blockDim(16, 16);
    dim3 gridDim(
        (image->width + (blockDim.x * 2) - 1) / (blockDim.x * 2),
        (image->height + (blockDim.y * 2) - 1) / (blockDim.y * 2));
    kernelRenderCircles<<<gridDim, blockDim>>>();
    cudaDeviceSynchronize();
}
