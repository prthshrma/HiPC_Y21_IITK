#include<bits/stdc++.h>
#include<thrust/device_vector.h>
using namespace std;

bool *adjacencyMatrix;
int *degree;

int Count[1];

__device__ int glock=0;

__global__ void kclique(int* ddegree,int start, int presentNodes, int K, int N,bool* adjacencyMat,int *KCount)
{
    int tid=threadIdx.x;
    int bid=blockIdx.x;
    
    int t = bid*128+tid;
    
    if(t<N)
    {
        int st[10000][4];
        int top=-1;
        int dclique[10000];
       
        top++;
        
        dclique[1]=t+1;
        st[top][0]=t+2,st[top][1]=1,st[top][2]=2,st[top][3]=K;
        
        KCount[0]=0;
        while(top!=-1)
        {
            int j=st[top][0],i=st[top][1],l=st[top][2],s=st[top][3];


            top--;
            if(j+1<=N)
            {
                top++;
                st[top][0]=j+1,st[top][1]=i,st[top][2]=l,st[top][3]=s;
            }
            if(ddegree[j]>=s-1)
            {

                dclique[l]=j;
                bool flag=true;
                for(int x=1;x<l+1;x++)
                {
                    for(int y=x+1;y<l+1;y++)
                    {
                        if(adjacencyMat[ 1ll*dclique[x]*1000000+dclique[y] ]==false)
                        {    
                            flag=false;
                            break;
                        }
                    }
                    if(!flag)
                        break;
                }
                if(flag)
                {
                    if(l<s)
                    {

                        top++;
                        st[top][0]=j+1,st[top][1]=j+1,st[top][2]=l+1,st[top][3]=s;
                    }
                    else
                    {
                        while(atomicCAS(&glock,0,1)) {}
                        __threadfence();   

                        KCount[0]++;

                        __threadfence();
                        atomicExch(&glock,0);

                    }
                }
            }
        }
    }
    __syncthreads();
}

int main()
{
    int k;
    string path;
   
    cin>>path;
    cin>>k;

    degree = (int*)malloc(sizeof(int)*1000000);
    cudaMallocManaged(&adjacencyMatrix, 1000000000000*sizeof(bool));
    
    ifstream MyReadFile(path);
    string myText;
    int n=0;
    while (getline (MyReadFile, myText)){
        int a,b,i=0;
        string t="";
        while(myText[i]!=' ')
        {
            t+=myText[i];
            i++;
        }
        a=stoi(t);
        b=stoi(myText.substr(i+1));

        n = max(n,max(a,b));

        adjacencyMatrix[1ll*a*1000000+b] = true;
        adjacencyMatrix[1ll*b*1000000+a] = true;
        degree[a]++;
        degree[b]++;
    }

    int *ddegree,*KCount;
    cudaMalloc((void**)&ddegree, 10000*sizeof(int));
    cudaMalloc((void**)&KCount, 1*sizeof(int));
    
    cudaMemcpy(ddegree, degree, 10000*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(KCount, Count, 1*sizeof(int), cudaMemcpyHostToDevice);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float gpu_time = 0.0f;
    
    cudaEventRecord(start, 0); 
    kclique<<<41,128>>>(ddegree,0,1,k,n,adjacencyMatrix,KCount);
    cudaEventRecord(stop, 0); 
    cudaEventSynchronize(stop);
    
    cudaEventElapsedTime(&gpu_time, start, stop);
    
    cudaMemcpy(Count, KCount, 1*sizeof(int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    
    cout<<Count[0]<<endl;
    cout<<"Execution Time: "<<gpu_time<<" ms"<<endl;
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    cudaFree(ddegree);
    cudaFree(KCount);

    free(degree);

    return 0;
}




