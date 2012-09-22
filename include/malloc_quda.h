#ifndef _MALLOC_QUDA_H
#define _MALLOC_QUDA_H

#include <cstdlib>

/*
 * The following functions should not be called directly.  Use the
 * macros below instead.
 */
void *device_malloc_(const char *func, const char *file, int line, size_t size);
void *safe_malloc_(const char *func, const char *file, int line, size_t size);
void *pinned_malloc_(const char *func, const char *file, int line, size_t size);
void *mapped_malloc_(const char *func, const char *file, int line, size_t size);
void device_free_(const char *func, const char *file, int line, void *ptr);
void host_free_(const char *func, const char *file, int line, void *ptr);

#define device_malloc(size) device_malloc_(__func__, __FILE__, __LINE__, size)
#define safe_malloc(size) safe_malloc_(__func__, __FILE__, __LINE__, size)
#define pinned_malloc(size) pinned_malloc_(__func__, __FILE__, __LINE__, size)
#define mapped_malloc(size) mapped_malloc_(__func__, __FILE__, __LINE__, size)
#define device_free(ptr) device_free_(__func__, __FILE__, __LINE__, ptr)
#define host_free(ptr) host_free_(__func__, __FILE__, __LINE__, ptr)

#endif // _MALLOC_QUDA_H
