#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <thrust/iterator/zip_iterator.h>

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
Val* thrust_reduce_by_key(
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

    return thrust::raw_pointer_cast(end.second);
}

//---------------------------------------------------------------------------
// Same thing, for a pair of keys
//---------------------------------------------------------------------------
template <typename Key1, typename Key2, typename Val>
void thrust_sort_by_key(
        Key1 *key1_begin, Key1 *key1_end, Key2 *key2_begin, Val *val_begin)
{
    thrust::sort_by_key(
            thrust::make_zip_iterator(
                thrust::make_tuple(
                    thrust::device_pointer_cast(key1_begin),
                    thrust::device_pointer_cast(key2_begin)
                    )
                ),
            thrust::make_zip_iterator(
                thrust::make_tuple(
                    thrust::device_pointer_cast(key1_end),
                    thrust::device_pointer_cast(key2_begin + (key1_end - key1_begin))
                    )
                ),
            thrust::device_pointer_cast(val_begin)
            );
}

template <typename Key1, typename Key2, typename Val>
Val* thrust_reduce_by_key(
        const Key1 *key1_begin, const Key1 *key1_end, const Key2 *key2_begin,
        const Val *val_begin,
        Key1 *key1_output, Key2 *key2_output, Val *val_output
        )
{
    thrust::pair<
        thrust::zip_iterator<
            thrust::tuple<
                thrust::device_ptr<Key1>,
                thrust::device_ptr<Key2>
            >
        >,
        thrust::device_ptr<Val>
        > end = thrust::reduce_by_key(
                thrust::make_zip_iterator(
                    thrust::make_tuple(
                        thrust::device_pointer_cast(key1_begin),
                        thrust::device_pointer_cast(key2_begin)
                        )
                    ),
                thrust::make_zip_iterator(
                    thrust::make_tuple(
                        thrust::device_pointer_cast(key1_end),
                        thrust::device_pointer_cast(key2_begin + (key1_end - key1_begin))
                        )
                    ),
                thrust::device_pointer_cast(val_begin),
                thrust::make_zip_iterator(
                    thrust::make_tuple(
                        thrust::device_pointer_cast(key1_output),
                        thrust::device_pointer_cast(key2_output)
                        )
                    ),
                thrust::device_pointer_cast(val_output)
            );

    return thrust::raw_pointer_cast(end.second);
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
  template V * thrust_reduce_by_key<K, V>(                                     \
      const K * key_begin, const K * key_end, const V * val_begin,             \
      K * key_output, V * val_output)

VEXCL_INSTANTIATE_THRUST_REDUCE_BY_KEY(int, double);

#undef VEXCL_INSTANTIATE_THRUST_REDUCE_BY_KEY

#define VEXCL_INSTANTIATE_THRUST_SORT_BY_KEY2(K1, K2, V)                       \
  template void thrust_sort_by_key<K1, K2, V>(                                 \
          K1 * key1_begin, K1 * key1_end, K2 * key2_begin, V * val_begin)

VEXCL_INSTANTIATE_THRUST_SORT_BY_KEY2(int, int, double);

#undef VEXCL_INSTANTIATE_THRUST_SORT_BY_KEY2

#define VEXCL_INSTANTIATE_THRUST_REDUCE_BY_KEY2(K1, K2, V)                     \
  template V *thrust_reduce_by_key<K1, K2, V>(                                 \
      const K1 * key1_begin, const K1 * key1_end, const K2 * key2_begin,       \
      const V * val_begin, K1 * key1_output, K2 * key2_output, V * val_output)

VEXCL_INSTANTIATE_THRUST_REDUCE_BY_KEY2(int, int, double);

#undef VEXCL_INSTANTIATE_THRUST_REDUCE_BY_KEY2
