#include <stddef.h>
#include <ctype.h>
#include <math.h>

double * euclidean_distances(
    double *pdistances,
    const double *pa,
    const double *pb,
    const size_t na,
    const size_t nb,
    const size_t dim
)
{
    for (size_t i = 0; i < na; i++)
    {
        const double *row_a = &pa[i * dim];
        
        for (size_t j = 0; j < nb; j++)
        {
            const double *row_b = &pb[j * dim];
            
            double d = 0, t = 0;
            for (size_t k = 0; k < dim; k++)
            {
                t = row_a[k] - row_b[k];
                d += t * t;
            }
            pdistances[i * nb + j] = sqrt(d);
        }
    }
    
    return pdistances;
}