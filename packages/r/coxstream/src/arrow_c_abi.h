// Arrow C Data Interface + C Stream Interface (ABI-stable struct definitions).
//
// Vendored verbatim from the Apache Arrow project (Apache-2.0). These are the
// complete, self-contained definitions of the three structs that make up the
// Arrow C data/stream interface. They contain NO Arrow code -- only plain-C
// struct layouts and function-pointer signatures -- so including this header
// requires neither libarrow headers nor linking against libarrow. The function
// pointers carried inside ArrowArrayStream point into whatever Arrow runtime
// produced the stream (here, the statically-linked arrow.so of the R `arrow`
// package), which is how we consume parquet row groups in C++ without any
// build-time Arrow dependency.
//
// Reference: https://arrow.apache.org/docs/format/CDataInterface.html

#ifndef COXSTREAM_ARROW_C_ABI_H
#define COXSTREAM_ARROW_C_ABI_H

#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

#ifndef ARROW_C_DATA_INTERFACE
#define ARROW_C_DATA_INTERFACE

#define ARROW_FLAG_DICTIONARY_ORDERED 1
#define ARROW_FLAG_NULLABLE 2
#define ARROW_FLAG_MAP_KEYS_SORTED 4

struct ArrowSchema {
    // Array type description
    const char* format;
    const char* name;
    const char* metadata;
    int64_t flags;
    int64_t n_children;
    struct ArrowSchema** children;
    struct ArrowSchema* dictionary;

    // Release callback
    void (*release)(struct ArrowSchema*);
    // Opaque producer-specific data
    void* private_data;
};

struct ArrowArray {
    // Array data description
    int64_t length;
    int64_t null_count;
    int64_t offset;
    int64_t n_buffers;
    int64_t n_children;
    const void** buffers;
    struct ArrowArray** children;
    struct ArrowArray* dictionary;

    // Release callback
    void (*release)(struct ArrowArray*);
    // Opaque producer-specific data
    void* private_data;
};

#endif  // ARROW_C_DATA_INTERFACE

#ifndef ARROW_C_STREAM_INTERFACE
#define ARROW_C_STREAM_INTERFACE

struct ArrowArrayStream {
    // Callback to get the stream type (will be the same for all arrays in the
    // stream). Returns 0 on success, an `errno`-compatible code otherwise.
    int (*get_schema)(struct ArrowArrayStream*, struct ArrowSchema* out);

    // Callback to get the next array. Returns 0 on success. A successful call
    // with `out->release == NULL` signals end of stream.
    int (*get_next)(struct ArrowArrayStream*, struct ArrowArray* out);

    // Callback to get optional detailed error information.
    const char* (*get_last_error)(struct ArrowArrayStream*);

    // Release callback
    void (*release)(struct ArrowArrayStream*);
    // Opaque producer-specific data
    void* private_data;
};

#endif  // ARROW_C_STREAM_INTERFACE

#ifdef __cplusplus
}
#endif

#endif  // COXSTREAM_ARROW_C_ABI_H
