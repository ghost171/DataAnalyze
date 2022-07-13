#include <stddef.h>
#include <ctype.h>
#include <math.h>
#include <omp.h>

double pair_euclidean_distances(const double *pa, const double *pb, const size_t dim)
{
    double d = 0, t = 0;

    #pragma omp parallel for reduction(+:d)
    for (size_t k = 0; k < dim; k++)
    {
        t = pa[k] - pb[k];
        d += t * t;
    }
    return sqrt(d);
}

double * euclidean_distances(
    double *po,
    const double *pa,
    const double *pb,
    const size_t na,
    const size_t nb,
    const size_t dim
)
{
    #pragma omp parallel for
    for (size_t i = 0; i < na; i++)
    {
        const double *row_a = &pa[i * dim];

        #pragma omp parallel for
        for (size_t j = 0; j < nb; j++)
        {
            const double *row_b = &pb[j * dim];
            po[i * nb + j] = pair_euclidean_distances(row_a, row_b, dim);
        }
    }

    return po;
}
