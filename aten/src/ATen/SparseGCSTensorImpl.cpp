#include <ATen/ATen.h>
#include <ATen/SparseGCSTensorImpl.h>
#include <ATen/InitialTensorOptions.h>
#include <ATen/core/LegacyTypeDispatch.h>

namespace at {

namespace {
  DeviceType sparseGCSTensorSetToDeviceType(DispatchKeySet key_set) {
    if (key_set.has(DispatchKey::SparseGCS_CPU)) {
      return kCPU;
    } else if (key_set.has(DispatchKey::SparseGCS_CUDA)) {
      return kCUDA;
    } else {
      AT_ERROR("Cannot construct SparseTensor with non-sparse tensor type ID ", key_set);
    }
  }
}


SparseGCSTensorImpl::SparseGCSTensorImpl(at::DispatchKeySet key_set,
                                         const caffe2::TypeMeta& data_type)
  :   SparseGCSTensorImpl(key_set, data_type
      , at::empty({0}, at::initialTensorOptions().device(sparseGCSTensorSetToDeviceType(key_set)).dtype(ScalarType::Long))
      // indices in case of GCS tensor is always a 1D array so need to init size as {1,0}.
      , at::empty({0}, at::initialTensorOptions().device(sparseGCSTensorSetToDeviceType(key_set)).dtype(ScalarType::Long))
      , at::empty({0}, at::initialTensorOptions().device(sparseGCSTensorSetToDeviceType(key_set)).dtype(data_type))
      , at::empty({0}, at::initialTensorOptions().device(sparseGCSTensorSetToDeviceType(key_set)).dtype(ScalarType::Long))
      , Scalar()  ) {}

SparseGCSTensorImpl::SparseGCSTensorImpl(at::DispatchKeySet key_set,
                                         const caffe2::TypeMeta& data_type,
                                         at::Tensor pointers, at::Tensor indices, at::Tensor values,
                                         at::Tensor reduction, Scalar fill_value)
  : TensorImpl(key_set, data_type, values.device()),
    pointers_(std::move(pointers)),
    indices_(std::move(indices)),
    values_(std::move(values)),
    reduction_(std::move(reduction)),
    fill_value_(std::move(fill_value)) {}

void SparseGCSTensorImpl::resize_and_clear_(int64_t nnz_size, int64_t ptr_size, int64_t redux_size, ArrayRef<int64_t> size) {
  // TODO: perform error checking.

  // call pointers().options() here since the struct contructor calls the tensor constructor
  // with args for device specific init.
  auto empty_pointers = at::empty(ptr_size, pointers().options());
  auto empty_indices = at::empty(nnz_size, indices().options());
  auto empty_values = at::empty(nnz_size, values().options());
  auto empty_reduction = at::empty(redux_size, reduction().options());

  // directly set to the member variables. there should be lots of error checking here.
  pointers_ = empty_pointers;
  indices_ = empty_indices;
  values_ = empty_values;
  reduction_ = empty_reduction;
  sizes_ = size.vec();
}
  
void SparseGCSTensorImpl::set_member_tensors_unsafe(const Tensor& pointers, const Tensor& indices,
                                                      const Tensor& values, const Tensor& reduction,
                                                    const Scalar& fill_value) {
  // TODO: perform lots of error checking to check correct type and sizes of inputs. Check
  // SparseTensorImpl::set_indices_and_values_unsafe() for details
  pointers_ = pointers;
  indices_ = indices;
  values_ = values;
  reduction_ = reduction;
  fill_value_ = fill_value;

  AT_ASSERT(device() == values_.device());    
  AT_ASSERT(indices_.device() == values_.device());
  AT_ASSERT(reduction_.device() == values_.device());

  auto reduction_accessor = reduction_.accessor<int64_t, 1>();

  rsplit_dim_ = reduction_accessor[reduction_.size(0)-1];
    
  dims0_.resize(rsplit_dim_);
  strides0_.resize(1);
  
  dims1_.resize(sizes_.size() - rsplit_dim_);
  strides1_.resize(1);
  
  for (int i = 0; i < rsplit_dim_; ++i) { dims0_[i] = i; }
  for (int i = 0; i < sizes_.size() - rsplit_dim_; ++i) { dims1_[i] = i + rsplit_dim_; }

  make_strides(0, strides0_, dims0_);
  make_strides(rsplit_dim_, strides1_, dims1_);
}

void SparseGCSTensorImpl::make_strides(int shape_start, std::vector<int64_t>& strides, std::vector<int64_t>& dims) {
  int ndims = dims.size();
  strides[0] = 1;
  for (int i = 0; i < ndims-1; ++i) {
    strides.insert(strides.begin(), strides[0] * sizes_[shape_start + ndims - i - 1]);
  }
}
}
