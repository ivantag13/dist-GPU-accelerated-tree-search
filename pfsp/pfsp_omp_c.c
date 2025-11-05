#define _GNU_SOURCE

/*
  Multi-GPU B&B to solve Taillard instances of the PFSP in C+OpenMP+CUDA.
*/

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <unistd.h>
#include <limits.h>
#include <sched.h>
#include <getopt.h>
#include <time.h>
#include <math.h>
#include <sched.h>
#include <omp.h>

#include "lib/c_bound_simple.h"
#include "lib/c_bound_johnson.h"
#include "lib/c_taillard.h"
#include "lib/PFSP_lib.h"
#include "lib/Pool_atom.h"
#include "lib/PFSP_statistic.h"
#include "../common/util.h"

void checkBest(int *best_l, int *best, _Atomic bool *bestLock)
{
    if (*best_l > *best)
        *best_l = *best;
    else if (*best_l < *best)
    {
        bool expected = false;
        while (1)
        {
            expected = false;
            if (atomic_compare_exchange_strong(bestLock, &expected, true))
            {
                if (*best_l < *best)
                    *best = *best_l;
                atomic_store(bestLock, false);
                break;
            }
        }
    }
    return;
}

/*******************************************************************************
Implementation of the parallel CPU PFSP search.
*******************************************************************************/
void pfsp_search(const int inst, const int lb, const int m, const int M, const int D, const double perc, int ws, int *best,
                 unsigned long long int *exploredTree, unsigned long long int *exploredSol, double *elapsedTime, unsigned long long int *expTreeGPU,
                 unsigned long long int *expSolGPU, unsigned long long int *genChildGPU, unsigned long long int *nbStealsGPU, unsigned long long int *nbSStealsGPU,
                 unsigned long long int *nbTerminationGPU, double *timeGpuCpy, double *timeGpuMalloc, double *timeGpuKer, double *timeGenChild,
                 double *timePoolOps, double *timeGpuIdle, double *timeTermination)
{
    // Initializing problem
    int jobs = taillard_get_nb_jobs(inst);
    int machines = taillard_get_nb_machines(inst);

    // Starting pool
    Node root;
    initRoot(&root, jobs);

    SinglePool_atom pool;
    initSinglePool_atom(&pool);

    pushBack(&pool, root);

    // Boolean variables for termination detection
    _Atomic bool bestLock = false;
    _Atomic bool allTasksIdleFlag = false;
    _Atomic bool eachTaskState[D]; // one task per GPU
    for (int i = 0; i < D; i++)
        atomic_store(&eachTaskState[i], BUSY);

    // Timer
    double startTime, endTime;
    startTime = omp_get_wtime();

    // Bounding data
    lb1_bound_data *lbound1;
    lbound1 = new_bound_data(jobs, machines);
    taillard_get_processing_times(lbound1->p_times, inst);
    fill_min_heads_tails(lbound1);

    lb2_bound_data *lbound2;
    lbound2 = new_johnson_bd_data(lbound1);
    fill_machine_pairs(lbound2 /*, LB2_FULL*/);
    fill_lags(lbound1->p_times, lbound2);
    fill_johnson_schedules(lbound1->p_times, lbound2);

    /*
      Step 1: We perform a partial breadth-first search on CPU in order to create
      a sufficiently large amount of work for GPU computation.
    */

    while (pool.size < D * m)
    {
        int hasWork = 0;
        Node parent = popFrontFree(&pool, &hasWork);
        if (!hasWork)
            break;
        decompose(jobs, lb, best, lbound1, lbound2, parent, exploredTree, exploredSol, &pool);
    }
    endTime = omp_get_wtime();
    double t1 = endTime - startTime;

    printf("\nInitial search on CPU completed\n");
    printf("Size of the explored tree: %llu\n", *exploredTree);
    printf("Number of explored solutions: %llu\n", *exploredSol);
    printf("Elapsed time: %f [s]\n", t1);

    /*
      Step 2: We continue the search on GPU in a depth-first manner, until there
      is not enough work.
    */

    SinglePool_atom multiPool[D];
    for (int i = 0; i < D; i++)
        initSinglePool_atom(&multiPool[i]);
    int best_l = *best;

    startTime = omp_get_wtime();
#pragma omp parallel num_threads(D) shared(bestLock, eachTaskState, allTasksIdleFlag, pool, multiPool,                                         \
                                               jobs, machines, lbound1, lbound2, lb, m, M, D, perc, ws, best, exploredTree, exploredSol,       \
                                               elapsedTime, expTreeGPU, expSolGPU, genChildGPU, nbStealsGPU, nbSStealsGPU, nbTerminationGPU,   \
                                               timeGpuCpy, timeGpuMalloc, timeGpuKer, timeGenChild, timePoolOps, timeGpuIdle, timeTermination) \
    reduction(min : best_l)
    {
        double startGpuMalloc, endGpuMalloc, startGpuKer, endGpuKer, startPoolOps, endPoolOps, startGpuIdle, endGpuIdle, startTermination, endTermination;
        // startGpuCpy, endGpuCpy, startGenChild, endGenChild,
        int cpuID = omp_get_thread_num();
        // int num_procs = omp_get_num_procs();
        // int cpu = sched_getcpu();
        // printf("Thread %d sees %d processors, & cpu is %d\n", cpuID, num_procs, cpu);

        unsigned long long int tree = 0, sol = 0;
        int nbSteals = 0, nbSSteals = 0;
        SinglePool_atom *pool_loc;
        pool_loc = &multiPool[cpuID];
        SinglePool_atom parentsPool, childrenPool;
        initSinglePool_atom(&parentsPool);
        initSinglePool_atom(&childrenPool);

        // int best_l = *best;
        bool taskState = BUSY;

        roundRobin_distribution(pool_loc, &pool, cpuID, D);
#pragma omp barrier
        pool.front = 0;
        pool.size = 0;

        startGpuMalloc = omp_get_wtime();
        int falseM = 20000;
        Node *parents = (Node *)malloc(falseM * sizeof(Node));
        Node *children = (Node *)malloc(jobs * falseM * sizeof(Node));
        Node *stolenNodes = (Node *)malloc(5 * M * sizeof(Node));
        endGpuMalloc = omp_get_wtime();
        timeGpuMalloc[cpuID] = endGpuMalloc - startGpuMalloc;

        while (1)
        {
            // Each task gets its parenst nodes from the pool
            startPoolOps = omp_get_wtime();
            int poolSize = popBackBulk(pool_loc, m, falseM, parents, 1);
            endPoolOps = omp_get_wtime();
            timePoolOps[cpuID] += endPoolOps - startPoolOps;

            if (poolSize > 0)
            {
                startPoolOps = omp_get_wtime();
                pushBackBulk(&parentsPool, parents, poolSize);
                endPoolOps = omp_get_wtime();
                timePoolOps[cpuID] += endPoolOps - startPoolOps;

                int hasWork = 1;
                if (best_l != *best)
                    checkBest(&best_l, best, &bestLock);

                startGpuKer = omp_get_wtime();
                while (hasWork)
                {
                    hasWork = 0;
                    Node parent = popBackFree(&parentsPool, &hasWork);
                    if (hasWork)
                    {
                        if (taskState == IDLE)
                        {
                            taskState = BUSY;
                            atomic_store(&eachTaskState[cpuID], BUSY);
                        }

                        decompose(jobs, lb, best, lbound1, lbound2, parent, &tree, &sol, &childrenPool);
                    }
                }
                endGpuKer = omp_get_wtime();
                timeGpuKer[cpuID] += endGpuKer - startGpuKer;

                if (childrenPool.size > 0)
                {
                    startPoolOps = omp_get_wtime();
                    int childrenSize = popBackBulkFree(&childrenPool, 1, childrenPool.size, children, 1);
                    pushBackBulk(pool_loc, children, childrenSize);
                    endPoolOps = omp_get_wtime();
                    timePoolOps[cpuID] += endPoolOps - startPoolOps;
                }
                if (best_l != *best)
                    checkBest(&best_l, best, &bestLock);
            }
            else
            {
                if (ws == 0)
                    break;
                else
                {
                    // Local work stealing
                    startGpuIdle = omp_get_wtime();
                    int tries = 0;
                    bool steal = false;
                    int victims[D];
                    permute(victims, D);
                    bool expected;

                    while (tries < D && steal == false)
                    { // WS0 loop
                        const int victimID = victims[tries];

                        if (victimID != cpuID)
                        { // if not me
                            SinglePool_atom *victim;
                            victim = &multiPool[victimID];
                            nbSteals++;
                            int nn = 0;
                            while (nn < 10)
                            { // WS1 loop
                                expected = false;
                                if (atomic_compare_exchange_strong(&(victim->lock), &expected, true))
                                { // get the lock
                                    int size = victim->size;

                                    if (size >= 2 * m)
                                    {
                                        //  Higher value for parameter M allows real application of steal-half strategy
                                        int stolenNodesSize = popBackBulkFree(victim, m, 5 * M, stolenNodes, 2); // ratio is 2

                                        if (stolenNodesSize == 0)
                                        {                                         // safety check
                                            atomic_store(&(victim->lock), false); // reset lock
                                            exit(-1);
                                        }

                                        startPoolOps = omp_get_wtime();
                                        pushBackBulk(pool_loc, stolenNodes, stolenNodesSize); // necessary lock operation
                                        endPoolOps = omp_get_wtime();
                                        timePoolOps[cpuID] += endPoolOps - startPoolOps;

                                        steal = true;
                                        nbSSteals++;
                                        atomic_store(&(victim->lock), false); // reset lock
                                        goto WS0;                             // Break out of WS0 loop
                                    }
                                    atomic_store(&(victim->lock), false); // reset lock
                                    break;                                // Break out of WS1 loop
                                }
                                nn++;
                            }
                        }
                        tries++;
                    }
                WS0:
                    endGpuIdle = omp_get_wtime();
                    timeGpuIdle[cpuID] += endGpuIdle - startGpuIdle;
                    if (steal == false) // Termination Detection
                    {
                        startTermination = omp_get_wtime();
                        nbTerminationGPU[cpuID]++;
                        if (taskState == BUSY)
                        {
                            taskState = IDLE;
                            atomic_store(&eachTaskState[cpuID], IDLE);
                        }
                        if (allIdle(eachTaskState, D, &allTasksIdleFlag))
                        {
                            endTermination = omp_get_wtime();
                            timeTermination[cpuID] += endTermination - startTermination;
                            break; // Break from GPU-accelerated bounding phase while
                        }
                        endTermination = omp_get_wtime();
                        timeTermination[cpuID] += endTermination - startTermination;
                        continue;
                    }
                    else
                    {
                        continue;
                    }
                }
            }
        }

        // Freeing variables from OpenMP environment
        free(parents);
        free(children);
        free(stolenNodes);

#pragma omp critical
        {
            // printf("Thread[%d] best_l[%d] best[%d]\n", cpuID, best_l, *best);
            nbStealsGPU[cpuID] = nbSteals;
            nbSStealsGPU[cpuID] = nbSSteals;
            expTreeGPU[cpuID] = tree;
            expSolGPU[cpuID] = sol;
            *exploredTree += expTreeGPU[cpuID];
            *exploredSol += expSolGPU[cpuID];
            //*best = MIN(*best, best_l);
            const int poolLocSize = pool_loc->size;
            for (int i = 0; i < poolLocSize; i++)
            {
                int hasWork = 0;
                pushBackFree(&pool, popBackFree(pool_loc, &hasWork));
                if (!hasWork)
                    break;
            }
        }
        deleteSinglePool_atom(pool_loc);
        deleteSinglePool_atom(&parentsPool);
        deleteSinglePool_atom(&childrenPool);
    } // End of parallel region

    *best = best_l;
    endTime = omp_get_wtime();
    double t2 = endTime - startTime;

    printf("\nSearch on Parallel CPU completed\n");
    printf("Size of the explored tree: %llu\n", *exploredTree);
    printf("Number of explored solutions: %llu\n", *exploredSol);
    printf("Elapsed time: %f [s]\n", t2);

    /*
      Step 3: We complete the depth-first search on CPU.
    */

    startTime = omp_get_wtime();
    while (1)
    {
        int hasWork = 0;
        Node parent = popBackFree(&pool, &hasWork);
        if (!hasWork)
            break;
        decompose(jobs, lb, best, lbound1, lbound2, parent, exploredTree, exploredSol, &pool);
    }

    // Freeing memory for structs common to all steps
    deleteSinglePool_atom(&pool);
    free_bound_data(lbound1);
    free_johnson_bd_data(lbound2);

    endTime = omp_get_wtime();
    double t3 = endTime - startTime;
    *elapsedTime = t1 + t2 + t3;
    printf("\nSearch on CPU completed\n");
    printf("Size of the explored tree: %llu\n", *exploredTree);
    printf("Number of explored solutions: %llu\n", *exploredSol);
    printf("Elapsed time: %f [s]\n", t3);

    printf("\nExploration terminated.\n");
}

int main(int argc, char *argv[])
{
    srand(time(NULL));
    int version = 2; // Multi-GPU version is code 2
    // Parallel Multi-core PFSP only uses: inst, lb, ub, m, T, C, ws
    int inst, lb, ub, m, M, T, D, C, ws, LB, commSize = 1; // commSize is an artificial variable here
    double perc;
    parse_parameters(argc, argv, &inst, &lb, &ub, &m, &M, &T, &D, &C, &ws, &LB, &perc);
    D = 0;

    int nb_proc = omp_get_num_procs();
    if (C > nb_proc)
    {
        printf("Execution Terminated. More processing units requested than the ones available\n");
        exit(1);
    }
    if (C == 0)
    {
        printf("No processing units requested. Please set C to at least 1\n");
        exit(1);
    }

    int jobs = taillard_get_nb_jobs(inst);
    int machines = taillard_get_nb_machines(inst);

    print_settings(inst, machines, jobs, ub, lb, D, C, ws, commSize, LB, version);

    int optimum = (ub == 1) ? taillard_get_best_ub(inst) : INT_MAX;
    unsigned long long int exploredTree = 0, exploredSol = 0;
    unsigned long long int expTreeCPU[C], expSolCPU[C], genChildCPU[C], nbStealsCPU[C], nbSStealsCPU[C], nbTerminationCPU[C];

    double elapsedTime = 0;
    double timeGpuCpy[C], timeGpuMalloc[C], timeCpuKer[C], timeGenChild[C], timePoolOps[C], timeCpuIdle[C], timeTermination[C];

    for (int i = 0; i < C; i++)
    {
        timeGpuCpy[i] = 0;
        timeGpuMalloc[i] = 0;
        timeCpuKer[i] = 0;
        timeGenChild[i] = 0;
        timePoolOps[i] = 0;
        timeCpuIdle[i] = 0;
        timeTermination[i] = 0;
        expTreeCPU[i] = 0;
        expSolCPU[i] = 0;
        genChildCPU[i] = 0;
        nbStealsCPU[i] = 0;
        nbSStealsCPU[i] = 0;
        nbTerminationCPU[i] = 0;
    }

    pfsp_search(inst, lb, m, T, C, perc, ws, &optimum, &exploredTree, &exploredSol, &elapsedTime,
                expTreeCPU, expSolCPU, genChildCPU, nbStealsCPU, nbSStealsCPU, nbTerminationCPU,
                timeGpuCpy, timeGpuMalloc, timeCpuKer, timeGenChild, timePoolOps, timeCpuIdle, timeTermination);

    print_results(optimum, exploredTree, exploredSol, elapsedTime);

    print_results_file_multi_gpu(inst, lb, D, C, ws, optimum, m, M, T, exploredTree, exploredSol, elapsedTime,
                                 expTreeCPU, expSolCPU, genChildCPU, nbStealsCPU, nbSStealsCPU, nbTerminationCPU,
                                 timeGpuCpy, timeGpuMalloc, timeCpuKer, timeGenChild, timePoolOps, timeCpuIdle, timeTermination);

    return 0;
}
