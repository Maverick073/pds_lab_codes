# MPI (Message Passing Interface) Documentation

## Introduction to MPI

**MPI** (Message Passing Interface) is a standardized and portable message-passing system designed for parallel computing in distributed memory environments. It allows processes running on different nodes of a distributed system to communicate with each other by sending and receiving messages. MPI is widely used in high-performance computing (HPC) applications where large-scale parallelism is required, such as scientific simulations, data processing, and AI workloads.

### Why MPI is Used

- **Distributed Memory Systems**: MPI allows processes to communicate across nodes in a distributed memory system.
- **Scalability**: MPI is highly scalable, allowing it to work with a large number of processors.
- **Flexibility**: It provides fine-grained control over communication, allowing developers to optimize for performance in various hardware environments.
- **Portability**: MPI is supported on most parallel computing platforms.

### Use Cases
- **Scientific Computing**: Simulations and calculations across multiple processors, like weather simulations, molecular modeling, and particle physics simulations.
- **Data Processing**: MPI is used for large-scale data processing tasks where datasets are split across multiple machines.
- **Parallel Computing**: For parallel execution of tasks that require synchronization between multiple nodes or processors.

## MPI Functions

Here are the commonly used MPI functions and their explanations in the programs:

| **Function**                   | **Signature**                                                                  | **Description**                                                                                   | **Example**                                                                                                                                                                  |
|---------------------------------|---------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `MPI_Init`                      | `int MPI_Init(int *argc, char ***argv);`                                        | Initializes the MPI environment. It must be called before any other MPI functions are invoked.     | ```MPI_Init(&argc, &argv);```                                                                                                                                               |
| `MPI_Comm_size`                 | `int MPI_Comm_size(MPI_Comm comm, int *size);`                                  | Determines the number of processes in a communicator.                                              | ```MPI_Comm_size(MPI_COMM_WORLD, &num_processes);```                                                                                                                       |
| `MPI_Comm_rank`                 | `int MPI_Comm_rank(MPI_Comm comm, int *rank);`                                  | Determines the rank (ID) of the calling process within a communicator.                             | ```MPI_Comm_rank(MPI_COMM_WORLD, &process_rank);```                                                                                                                         |
| `MPI_Bcast`                     | `int MPI_Bcast(void *buffer, int count, MPI_Datatype datatype, int root, MPI_Comm comm);` | Broadcasts a message from the root process to all other processes in the communicator.             | ```MPI_Bcast(message, MAX_MESSAGE_LENGTH, MPI_CHAR, 0, MPI_COMM_WORLD);```                                                                                                 |
| `MPI_Send`                      | `int MPI_Send(void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm);` | Sends a message from the calling process to a destination process.                                  | ```MPI_Send(send_message, MAX_MESSAGE_LENGTH, MPI_CHAR, i, 0, MPI_COMM_WORLD);```                                                                                           |
| `MPI_Recv`                      | `int MPI_Recv(void *buf, int count, MPI_Datatype datatype, int source, int tag, MPI_Comm comm, MPI_Status *status);` | Receives a message sent from another process.                                                      | ```MPI_Recv(recv_message, MAX_MESSAGE_LENGTH, MPI_CHAR, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);```                                                                       |
| `MPI_Finalize`                  | `int MPI_Finalize(void);`                                                      | Terminates the MPI environment, releasing all resources. Should be the last MPI function called.  | ```MPI_Finalize();```                                                                                                                                                        |
| `MPI_Scatter`                   | `int MPI_Scatter(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, int recvcount, MPI_Datatype recvtype, int root, MPI_Comm comm);` | Distributes data from the root process to all other processes in the communicator.                  | ```MPI_Scatter(numbers, 1, MPI_INT, &recv_number, 1, MPI_INT, 0, MPI_COMM_WORLD);```                                                                                        |
| `MPI_Gather`                    | `int MPI_Gather(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, int recvcount, MPI_Datatype recvtype, int root, MPI_Comm comm);` | Collects data from all processes and sends it to the root process.                                 | ```MPI_Gather(&result, 1, MPI_INT, gathered_results, 1, MPI_INT, 0, MPI_COMM_WORLD);```                                                                                    |
| `MPI_Reduce`                    | `int MPI_Reduce(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, int root, MPI_Comm comm);` | Performs a reduction operation (e.g., sum, max) on data across all processes in the communicator.    | ```MPI_Reduce(&result, &sum_of_squares, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);```                                                                                         |
| `MPI_Barrier`                   | `int MPI_Barrier(MPI_Comm comm);`                                               | Synchronizes all processes in a communicator, blocking until all processes reach this function.    | ```MPI_Barrier(MPI_COMM_WORLD);```                                                                                                                                          |

## Examples of MPI Functions in Programs

- **MPI Initialization**: 
  - **Function**: `MPI_Init()`
  - **Example**:
    ```cpp
    MPI_Init(&argc, &argv);
    ```

- **Getting the Size and Rank**:
  - **Functions**: `MPI_Comm_size()`, `MPI_Comm_rank()`
  - **Example**:
    ```cpp
    MPI_Comm_size(MPI_COMM_WORLD, &num_processes);
    MPI_Comm_rank(MPI_COMM_WORLD, &process_rank);
    ```

- **Broadcasting a Message**:
  - **Function**: `MPI_Bcast()`
  - **Example**:
    ```cpp
    MPI_Bcast(message, MAX_MESSAGE_LENGTH, MPI_CHAR, 0, MPI_COMM_WORLD);
    ```

- **Sending and Receiving Messages**:
  - **Functions**: `MPI_Send()`, `MPI_Recv()`
  - **Example**:
    ```cpp
    MPI_Send(send_message, MAX_MESSAGE_LENGTH, MPI_CHAR, i, 0, MPI_COMM_WORLD);
    MPI_Recv(recv_message, MAX_MESSAGE_LENGTH, MPI_CHAR, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    ```

- **Performing a Reduce Operation**:
  - **Function**: `MPI_Reduce()`
  - **Example**:
    ```cpp
    MPI_Reduce(&result, &sum_of_squares, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    ```

---

### Experiment 7 - Chat Server Application Using MPI
```cpp
#include <mpi.h>
#include <stdio.h>
#include <string.h>

#define MAX_MESSAGE_LENGTH 1024

int main(int argc, char** argv) {
    int num_processes, process_rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &num_processes);
    MPI_Comm_rank(MPI_COMM_WORLD, &process_rank);

    if (process_rank == 0) {
        char message[MAX_MESSAGE_LENGTH];
        printf("Enter a message: ");
        fflush(stdout);
        fgets(message, MAX_MESSAGE_LENGTH, stdin);
        MPI_Bcast(message, MAX_MESSAGE_LENGTH, MPI_CHAR, 0, MPI_COMM_WORLD);
    } else {
        char message[MAX_MESSAGE_LENGTH];
        MPI_Bcast(message, MAX_MESSAGE_LENGTH, MPI_CHAR, 0, MPI_COMM_WORLD);
        printf("Received message from root process: %s", message);
    }

    for (int i = 0; i < num_processes; i++) {
        if (process_rank != i) {
            char send_message[MAX_MESSAGE_LENGTH];
            sprintf(send_message, "Hello from process %d!", process_rank);
            MPI_Send(send_message, MAX_MESSAGE_LENGTH, MPI_CHAR, i, 0, MPI_COMM_WORLD);

            char recv_message[MAX_MESSAGE_LENGTH];
            MPI_Recv(recv_message, MAX_MESSAGE_LENGTH, MPI_CHAR, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            printf("Received message from process %d: %s\n", i, recv_message);
        }
    }

    MPI_Finalize();
    return 0;
}
```
### Experiment 8 - Mutual Exclusion Using MPI
```cpp
#include <mpi.h>
#include <stdio.h>
#include <stdbool.h>
#include <unistd.h> // For sleep function

#define REQUEST 1
#define REPLY 2
#define RELEASE 3
#define MAX_PROCESSES 10

int timestamp = 0;
int num_processes, process_rank;
bool requesting = false;
bool in_critical_section = false;
int replies_count = 0;
bool deferred_reply[MAX_PROCESSES];

void send_message(int dest, int tag) {
    if (dest >= 0 && dest < num_processes) {
        MPI_Send(&timestamp, 1, MPI_INT, dest, tag, MPI_COMM_WORLD);
    }
}

void receive_message(int* recv_timestamp, int* source, int* tag) {
    MPI_Status status;
    MPI_Recv(recv_timestamp, 1, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
    *source = status.MPI_SOURCE;
    *tag = status.MPI_TAG;
}

void handle_request(int source, int recv_timestamp) {
    bool grant_permission = false;

    if (!in_critical_section && !requesting) {
        grant_permission = true;
    } else if (recv_timestamp < timestamp || (recv_timestamp == timestamp && source < process_rank)) {
        grant_permission = true;
    }

    if (grant_permission) {
        send_message(source, REPLY);
    } else {
        deferred_reply[source] = true;
    }
}

void handle_reply() {
    replies_count++;
    if (replies_count == num_processes - 1) {
        in_critical_section = true;
        printf("Process %d in critical section\n", process_rank);
        sleep(1); // Simulate time spent in the critical section
    }
}

void handle_release(int source) {
    printf("Process %d received RELEASE from process %d\n", process_rank, source);
}

void request_critical_section() {
    requesting = true;
    timestamp++;
    replies_count = 0;

    for (int i = 0; i < num_processes; i++) {
        if (i != process_rank) {
            send_message(i, REQUEST);
        }
    }
}

void release_critical_section() {
    printf("Process %d releasing critical section\n", process_rank);
    in_critical_section = false;
    requesting = false;

    for (int i = 0; i < num_processes; i++) {
        if (deferred_reply[i]) {
            send_message(i, REPLY);
            deferred_reply[i] = false;
        }
    }

    for (int i = 0; i < num_processes; i++) {
        if (i != process_rank) {
            send_message(i, RELEASE);
        }
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &num_processes);
    MPI_Comm_rank(MPI_COMM_WORLD, &process_rank);

    for (int i = 0; i < MAX_PROCESSES; i++) {
        deferred_reply[i] = false;
    }

    if (process_rank == 0) {
        sleep(1); // Process 0 starts first
    }
    request_critical_section();

    while (true) {
        int recv_timestamp, tag, source;
        receive_message(&recv_timestamp, &source, &tag);

        timestamp = (timestamp > recv_timestamp) ? timestamp + 1 : recv_timestamp + 1;

        switch (tag) {
            case REQUEST:
                printf("Process %d received REQUEST from process %d\n", process_rank, source);
                handle_request(source, recv_timestamp);
                break;

            case REPLY:
                printf("Process %d received OK from process %d\n", process_rank, source);
                handle_reply();
                break;

            case RELEASE:
                handle_release(source);
                break;
        }

        if (in_critical_section) {
            release_critical_section();
            break;
        }
    }

    MPI_Finalize();
    return 0;
}
```
### Experiment 9 -  Group Communication Using MPI
```cpp
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char** argv) {
    int num_processes, process_rank;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &num_processes);
    MPI_Comm_rank(MPI_COMM_WORLD, &process_rank);

    int array_size = num_processes; 
    char message[100];

    if (process_rank == 0) {
        printf("Enter a message to broadcast: ");
        fflush(stdout);
        fgets(message, 100, stdin);

        printf("Array size set to number of processes: %d\n", array_size);
        printf("Enter %d elements for the array:\n", array_size);
        fflush(stdout);
    }

    MPI_Bcast(&array_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
    int numbers[array_size];

    if (process_rank == 0) {
        for (int i = 0; i < array_size; i++) {
            scanf("%d", &numbers[i]);
        }
    }

    MPI_Bcast(message, 100, MPI_CHAR, 0, MPI_COMM_WORLD);
    printf("Process %d received message: %s\n", process_rank, message);

    int recv_number;
    MPI_Scatter(numbers, 1, MPI_INT, &recv_number, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int result = recv_number * recv_number;
    printf("Process %d received %d and computed its square: %d\n", process_rank, recv_number, result);

    int gathered_results[array_size];
    MPI_Gather(&result, 1, MPI_INT, gathered_results, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (process_rank == 0) {
        printf("Gathered results: ");
        for (int i = 0; i < array_size; i++) {
            printf("%d ", gathered_results[i]);
        }
        printf("\n");
    }

    int sum_of_squares;
    MPI_Reduce(&result, &sum_of_squares, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (process_rank == 0) {
        printf("Sum of squares: %d\n", sum_of_squares);
    }

    MPI_Finalize();
    return 0;
}
```
### Experiment 10 - Clock Synchronization Using MPI
```cpp
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define ROOT 0

int main(int argc, char** argv) {
    int num_processes, process_rank;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &num_processes);
    MPI_Comm_rank(MPI_COMM_WORLD, &process_rank);

    int local_time;
    if (process_rank == ROOT) {
        printf("You have %d processes. Enter the logical clock values for each process:\n", num_processes);
        for (int i = 0; i < num_processes; i++) {
            if (i == ROOT) {
                printf("Enter the logical clock value for process %d: ", ROOT);
                scanf("%d", &local_time);
            } else {
                int input_time;
                printf("Enter the logical clock value for process %d: ", i);
                scanf("%d", &input_time);
                MPI_Send(&input_time, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
            }
        }
    } else {
        MPI_Recv(&local_time, 1, MPI_INT, ROOT, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    printf("Process %d has local time: %d\n", process_rank, local_time);

    int* local_times = NULL;
    if (process_rank == ROOT) {
        local_times = (int*)malloc(num_processes * sizeof(int));
    }

    MPI_Gather(&local_time, 1, MPI_INT, local_times, 1, MPI_INT, ROOT, MPI_COMM_WORLD);

    if (process_rank == ROOT) {
        int sum = 0;
        for (int i = 0; i < num_processes; i++) {
            printf("Process %d time: %d\n", i, local_times[i]);
            sum += local_times[i];
        }

        int average_time = sum / num_processes;
        printf("Coordinator calculated the average time: %d\n", average_time);

        int* adjustments = (int*)malloc(num_processes * sizeof(int));
        for (int i = 0; i < num_processes; i++) {
            adjustments[i] = average_time - local_times[i];
            printf("Process %d should adjust its time by: %d\n", i, adjustments[i]);
        }

        for (int i = 1; i < num_processes; i++) {
            MPI_Send(&adjustments[i], 1, MPI_INT, i, 0, MPI_COMM_WORLD);
        }

        local_time += adjustments[ROOT];
        printf("Process %d adjusted its time by %d. New local time: %d\n", ROOT, adjustments[ROOT], local_time);
        free(local_times);
        free(adjustments);
    } else {
        int adjustment;
        MPI_Recv(&adjustment, 1, MPI_INT, ROOT, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        local_time += adjustment;
        printf("Process %d adjusted its time by %d. New local time: %d\n", process_rank, adjustment, local_time);
    }

    MPI_Finalize();
    return 0;
}
```

### Experiment 11 - Leader Election 
```cpp
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    int leader;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int* uid = (int*)malloc(size * sizeof(int));
    int* token = (int*)malloc(size * sizeof(int));

    uid[rank] = rank * 100 + rank;

    if (rank == 0) {
        token[rank] = uid[rank];
    }

    if (rank != 0) {
        MPI_Recv(token, size, MPI_INT, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("Process %d received token from process %d\n", rank, rank - 1);
        token[rank] = uid[rank];
    }

    MPI_Send(token, size, MPI_INT, (rank + 1) % size, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        MPI_Recv(token, size, MPI_INT, size - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("Process %d received token from process %d\n", rank, size - 1);

        int max = token[0];
        leader = 0;
        for (int i = 1; i < size; i++) {
            if (token[i] > max) {
                max = token[i];
                leader = i;
            }
        }
        MPI_Send(&leader, 1, MPI_INT, (rank + 1) % size, 1, MPI_COMM_WORLD);
    }

    if (rank != 0) {
        MPI_Recv(&leader, 1, MPI_INT, rank - 1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("Process %d received leader information from process %d, Leader is %d\n", rank, rank - 1, leader);
        MPI_Send(&leader, 1, MPI_INT, (rank + 1) % size, 1, MPI_COMM_WORLD);
    }

    if (rank == 0) {
        MPI_Recv(&leader, 1, MPI_INT, size - 1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("Process %d received leader information from process %d, Leader is %d\n", rank, size - 1, leader);
    }

    free(uid);
    free(token);
    MPI_Finalize();
    return 0;
}
```

### Commands to Compile and Run an MPI Program

 **Compile and Run the MPI Program**
   ```bash
   mpicc -o <output_filename> <source_filename>.c
   mpirun -np <number_of_processes> ./<output_filename>
```
