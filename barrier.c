#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <mpi.h>
#include <math.h>


void mpiBarrier(int,int);  // declare barrier function
//void doComputation(int,int);  // declare computation function


int main (int argc,char **argv){
    int rank;
    int numprocs;

    MPI_Init(&argc, &argv); 
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);

    printf( "This is rank %d before running mpibarrier \n",rank);
    sleep(5);
    mpiBarrier(rank,numprocs);
    sleep(5);
    printf( "This is rank %d after running mpibarrier \n",rank);

    MPI_Finalize();
    
    return 0;
}
 
//mpiBarrier function-tree Barrier
void mpiBarrier(int rank,int numprocs){

    int k = (int)log2(numprocs);  // log2P is the complexity constraint

    // aggregation phase
    int i=0;
    int j=0;
    int step = 0; 
    char msg_1[] = "reaches to the barrier";
    for(i=k-1;i>=0;i--){
        step += 1;
	    for(j=pow(2,i);j<pow(2,i+1);j++){
		    if(rank == j){
		        MPI_Send(msg_1, strlen(msg_1),MPI_CHAR, j-(int)pow(2,i), 99, MPI_COMM_WORLD);
			}
		    if(rank == j-(int)pow(2,i)){
		        MPI_Status status;
		        char recv[strlen(msg_1)];
		        MPI_Recv(&recv, strlen(msg_1),MPI_CHAR, j, 99, MPI_COMM_WORLD, &status);
		        printf("Rank %d receives message: '%s' from %d in step %d \n",j-(int)pow(2,i),msg_1, j, step);
			}
        }
    }


    // dissemination phase
    i=0;
    j=0;
    step = 0;
    char msg_2[] = "finish the barrier";
    for(i=0;i<=k-1;i++){
        step += 1;
	    for(j=0;j<=pow(2,i)-1;j++){
		    if(rank==j){
                printf("Rank %d sends message: '%s' to %d in step %d \n", j, msg_2, j+(int)pow(2,i), step);
		        MPI_Send(msg_2, strlen(msg_2),MPI_CHAR, j+(int)pow(2,i), 98, MPI_COMM_WORLD);
			}
	    	if(rank== j+(int)pow(2,i)){
		        MPI_Status status;
		        char recv[strlen(msg_2)];
		        MPI_Recv(recv, strlen(msg_2),MPI_CHAR, j, 98, MPI_COMM_WORLD, &status);
			 }
        }
    }

}
