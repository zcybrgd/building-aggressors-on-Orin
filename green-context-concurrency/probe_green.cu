#include <cuda.h>
#include <stdio.h>

int main() {
    cuInit(0);
    CUdevice dev; cuDeviceGet(&dev, 0);
    CUcontext ctx; cuDevicePrimaryCtxRetain(&ctx, dev); cuCtxSetCurrent(ctx);

    CUdevResource full;
    cuDeviceGetDevResource(dev, &full, CU_DEV_RESOURCE_TYPE_SM);
    printf("Full pool: smCount=%u\n", full.sm.smCount);

    // Try every possible minCount from 1 to 8 with every nbGroups from 1 to 8
    printf("\n--- Probing all combinations ---\n");
    for (unsigned int min = 1; min <= 8; min++) {
        for (unsigned int nb = 1; nb <= 8; nb++) {
            CUdevResource groups[8], rem;
            unsigned int nb_out = nb;
            CUresult res = cuDevSmResourceSplitByCount(groups, &nb_out, &full, &rem, 0, min);
            if (res == CUDA_SUCCESS) {
                printf("minCount=%u nbGroups_requested=%u -> group[0].smCount=%u rem.smCount=%u nb_actual=%u\n",
                       min, nb, groups[0].sm.smCount, rem.sm.smCount, nb_out);
            }
        }
    }
    return 0;
}

