#include <thrust/sort.h>
#include <thrust/device_ptr.h>

//---------------------------------------------------------------------------
// NVCC is not yet able to compile C++11 code.
// Hence the need to keep Thrust and VexCL code in separate files.
//---------------------------------------------------------------------------
template <typename Key, typename Val>
void thrust_sort_by_key(Key *key_begin, Key *key_end, Val *val_begin) {
    thrust::sort_by_key(
            thrust::device_pointer_cast(key_begin),
            thrust::device_pointer_cast(key_end),
            thrust::device_pointer_cast(val_begin)
            );
}

template <typename Key, typename Val>
std::pair<Key*, Val*> thrust_reduce_by_key(
        const Key *key_begin, const Key *key_end, const Val *val_begin,
        Key *key_output, Val *val_output
        )
{
    thrust::pair< thrust::device_ptr<Key>, thrust::device_ptr<Val> >
        end = thrust::reduce_by_key(
            thrust::device_pointer_cast(key_begin),
            thrust::device_pointer_cast(key_end),
            thrust::device_pointer_cast(val_begin),
            thrust::device_pointer_cast(key_output),
            thrust::device_pointer_cast(val_output)
            );

    return std::make_pair(
            thrust::raw_pointer_cast(end.first),
            thrust::raw_pointer_cast(end.second)
            );
}

//---------------------------------------------------------------------------
// Due to the code separation we also need to explicitly instantiate the
// necessary templates.
//---------------------------------------------------------------------------
#define VEXCL_INSTANTIATE_THRUST_SORT_BY_KEY(K, V)                             \
  template void thrust_sort_by_key<K, V>(                                      \
          K * key_begin, K * key_end, V * val_begin)

VEXCL_INSTANTIATE_THRUST_SORT_BY_KEY(int, double);

#undef VEXCL_INSTANTIATE_THRUST_SORT_BY_KEY

#define VEXCL_INSTANTIATE_THRUST_REDUCE_BY_KEY(K, V)                           \
  template std::pair<K *, V *> thrust_reduce_by_key<K, V>(                     \
      const K * key_begin, const K * key_end, const V * val_begin,             \
      K * key_output, V * val_output)

VEXCL_INSTANTIATE_THRUST_REDUCE_BY_KEY(int, double);

#undef VEXCL_INSTANTIATE_THRUST_REDUCE_BY_KEY
