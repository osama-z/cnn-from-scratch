/**
 * DAY 7, STEP 2: THE MODEL LOADER
 * ================================
 *
 * Header-only, C99, and valid C++ too -- golden_c.c and golden_cpp.cpp both
 * include this file, so the two languages provably parse identical bytes.
 * If the loader had been written twice, "C and C++ agree" would partly be a
 * claim about two loaders rather than about two inference engines.
 *
 * THE FORMAT (see MODEL_FORMAT.md for the full spec)
 * ---------------------------------------------------
 *   header, 16 bytes:
 *     u32 magic     0x574E4E43   'C','N','N','W' little-endian
 *     u32 version   1
 *     u32 n_tensors
 *     u32 checksum  sum of every body byte, mod 2^32
 *
 *   then, per tensor, each field 4-byte aligned:
 *     u32 name_len
 *     u8  name[name_len]      + zero padding to a 4-byte boundary
 *     u32 dtype               0 = float32
 *     u32 ndim
 *     u32 dims[ndim]
 *     u32 n_bytes
 *     u8  data[n_bytes]       + padding
 *
 * ZERO-COPY
 * ---------
 * Every field before `data` is a multiple of 4 bytes, so `data` always begins
 * on a 4-byte boundary and a `const float*` can point straight into the loaded
 * buffer. No parsing pass, no second allocation.
 *
 * That alignment is not decoration. On x86 an unaligned float load is merely
 * slow; on several ARM cores it raises a fault. This is exactly why real
 * formats (TFLite via FlatBuffers, ONNX via protobuf + external data) are so
 * fussy about alignment -- they are designed to be mmap'd on embedded targets
 * and read in place.
 */

#ifndef MODEL_IO_H
#define MODEL_IO_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#define MODEL_MAGIC        0x574E4E43u
#define MODEL_VERSION      1u
#define MODEL_DTYPE_F32    0u
#define MODEL_MAX_TENSORS  16
#define MODEL_MAX_DIMS     4
#define MODEL_MAX_NAME     63

typedef struct {
    char         name[MODEL_MAX_NAME + 1];
    uint32_t     ndim;
    uint32_t     dims[MODEL_MAX_DIMS];
    uint32_t     n_elem;
    const float* data;          /* points INTO blob -- do not free separately */
} MTensor;

typedef struct {
    unsigned char* blob;        /* the whole file, owned */
    size_t         size;
    uint32_t       n_tensors;
    MTensor        tensors[MODEL_MAX_TENSORS];
} Model;

/* Little-endian u32 read. Explicit byte assembly rather than a cast, so the
 * result does not depend on the host's byte order -- the format says LE, and
 * this reads LE on any machine. */
static uint32_t model__u32(const unsigned char* p) {
    return (uint32_t)p[0]
         | ((uint32_t)p[1] << 8)
         | ((uint32_t)p[2] << 16)
         | ((uint32_t)p[3] << 24);
}

static uint32_t model__pad4(uint32_t n) { return ((uint32_t)(-(int32_t)n)) & 3u; }

/**
 * Returns 0 on success, non-zero on failure (message printed to stderr).
 *
 * Every read is bounds-checked against the file size. A truncated model is the
 * realistic embedded failure -- a flash write interrupted by a brownout -- and
 * it must be caught here, not discovered as a wrong prediction in flight.
 */
static int model_load(Model* m, const char* path) {
    memset(m, 0, sizeof(*m));

    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "model_load: cannot open '%s'\n", path); return 1; }

    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (sz < 16) { fprintf(stderr, "model_load: file too small\n"); fclose(f); return 1; }

    m->blob = (unsigned char*)malloc((size_t)sz);
    if (!m->blob) { fprintf(stderr, "model_load: out of memory\n"); fclose(f); return 1; }
    if (fread(m->blob, 1, (size_t)sz, f) != (size_t)sz) {
        fprintf(stderr, "model_load: short read\n");
        free(m->blob); m->blob = NULL; fclose(f); return 1;
    }
    fclose(f);
    m->size = (size_t)sz;

    /* ---- header ---- */
    uint32_t magic    = model__u32(m->blob + 0);
    uint32_t version  = model__u32(m->blob + 4);
    uint32_t ntensors = model__u32(m->blob + 8);
    uint32_t checksum = model__u32(m->blob + 12);

    if (magic != MODEL_MAGIC) {
        /* The magic number earns its keep here: without it, a JPEG or a
         * truncated file would be parsed as tensor metadata and produce
         * enormous bogus dimensions before anything looked wrong. */
        fprintf(stderr, "model_load: bad magic 0x%08X (expected 0x%08X)\n",
                magic, MODEL_MAGIC);
        goto fail;
    }
    if (version != MODEL_VERSION) {
        fprintf(stderr, "model_load: version %u, this loader speaks %u\n",
                version, MODEL_VERSION);
        goto fail;
    }
    if (ntensors > MODEL_MAX_TENSORS) {
        fprintf(stderr, "model_load: %u tensors exceeds limit %d\n",
                ntensors, MODEL_MAX_TENSORS);
        goto fail;
    }

    /* ---- integrity ---- */
    {
        uint32_t sum = 0;
        for (size_t i = 16; i < m->size; ++i) sum += m->blob[i];
        if (sum != checksum) {
            fprintf(stderr, "model_load: checksum 0x%08X, expected 0x%08X "
                            "(file truncated or corrupt)\n", sum, checksum);
            goto fail;
        }
    }

    /* ---- tensor records ---- */
    {
        size_t off = 16;
        for (uint32_t t = 0; t < ntensors; ++t) {
            MTensor* mt = &m->tensors[t];

            if (off + 4 > m->size) goto truncated;
            uint32_t name_len = model__u32(m->blob + off); off += 4;
            if (name_len > MODEL_MAX_NAME || off + name_len > m->size) goto truncated;
            memcpy(mt->name, m->blob + off, name_len);
            mt->name[name_len] = '\0';
            off += name_len + model__pad4(name_len);

            if (off + 8 > m->size) goto truncated;
            uint32_t dtype = model__u32(m->blob + off); off += 4;
            mt->ndim       = model__u32(m->blob + off); off += 4;
            if (dtype != MODEL_DTYPE_F32) {
                fprintf(stderr, "model_load: tensor '%s' has dtype %u, "
                                "only float32 supported\n", mt->name, dtype);
                goto fail;
            }
            if (mt->ndim > MODEL_MAX_DIMS) {
                fprintf(stderr, "model_load: tensor '%s' has %u dims, max %d\n",
                        mt->name, mt->ndim, MODEL_MAX_DIMS);
                goto fail;
            }

            if (off + 4u * mt->ndim > m->size) goto truncated;
            mt->n_elem = 1;
            for (uint32_t d = 0; d < mt->ndim; ++d) {
                mt->dims[d] = model__u32(m->blob + off); off += 4;
                mt->n_elem *= mt->dims[d];
            }

            if (off + 4 > m->size) goto truncated;
            uint32_t nbytes = model__u32(m->blob + off); off += 4;
            if (nbytes != mt->n_elem * 4u) {
                fprintf(stderr, "model_load: tensor '%s' claims %u bytes but "
                                "its shape needs %u\n", mt->name, nbytes, mt->n_elem * 4u);
                goto fail;
            }
            if (off + nbytes > m->size) goto truncated;

            /* Zero-copy: `off` is 4-aligned by construction, so this cast is
             * safe on every target the format targets. */
            mt->data = (const float*)(const void*)(m->blob + off);
            off += nbytes + model__pad4(nbytes);
        }
    }

    m->n_tensors = ntensors;
    return 0;

truncated:
    fprintf(stderr, "model_load: file truncated while reading tensor records\n");
fail:
    free(m->blob);
    m->blob = NULL;
    return 1;
}

static void model_free(Model* m) {
    free(m->blob);
    m->blob = NULL;
    m->n_tensors = 0;
}

/** Look a tensor up by name. Returns NULL if absent. */
static const MTensor* model_get(const Model* m, const char* name) {
    for (uint32_t i = 0; i < m->n_tensors; ++i)
        if (strcmp(m->tensors[i].name, name) == 0) return &m->tensors[i];
    return NULL;
}

/** Fetch by name or abort. Missing weights are a bug, not a runtime condition. */
static const MTensor* model_require(const Model* m, const char* name) {
    const MTensor* t = model_get(m, name);
    if (!t) {
        fprintf(stderr, "model_require: tensor '%s' not found in model\n", name);
        exit(1);
    }
    return t;
}

/* Optional debug helper: not every translation unit calls it, and that is
 * fine -- a header-only library should not force callers to use every symbol. */
#if defined(__GNUC__)
__attribute__((unused))
#endif
static void model_print(const Model* m) {
    printf("  model: %u tensors, %zu bytes\n", m->n_tensors, m->size);
    for (uint32_t i = 0; i < m->n_tensors; ++i) {
        const MTensor* t = &m->tensors[i];
        printf("    %-14s (", t->name);
        for (uint32_t d = 0; d < t->ndim; ++d)
            printf("%u%s", t->dims[d], d + 1 < t->ndim ? ", " : "");
        printf(")  %u values\n", t->n_elem);
    }
}

#endif /* MODEL_IO_H */
