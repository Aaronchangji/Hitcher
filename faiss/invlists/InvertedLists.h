/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#ifndef FAISS_INVERTEDLISTS_IVF_H
#define FAISS_INVERTEDLISTS_IVF_H

/**
 * Definition of inverted lists + a few common classes that implement
 * the interface.
 */

#include <faiss/MetricType.h>
#include <vector>

namespace faiss {

struct InvertedListsIterator {
    virtual ~InvertedListsIterator();
    virtual bool is_available() const = 0;
    virtual void next() = 0;
    virtual std::pair<idx_t, const uint8_t*> get_id_and_codes() = 0;
};

/** Table of inverted lists
 * multithreading rules:
 * - concurrent read accesses are allowed
 * - concurrent update accesses are allowed
 * - for resize and add_entries, only concurrent access to different lists
 *   are allowed
 */
struct InvertedLists {
    size_t nlist;     ///< number of possible key values
    size_t code_size; ///< code size per vector in bytes

    /// request to use iterator rather than get_codes / get_ids
    bool use_iterator = false;

    InvertedLists(size_t nlist, size_t code_size);

    virtual ~InvertedLists();

    /// used for BlockInvertedLists, where the codes are packed into groups
    /// and the individual code size is meaningless
    static const size_t INVALID_CODE_SIZE = static_cast<size_t>(-1);

    /*************************
     *  Read only functions */

    /// get the size of a list
    virtual size_t list_size(size_t list_no) const = 0;

    /** get the codes for an inverted list
     * must be released by release_codes
     *
     * @return codes    size list_size * code_size
     */
    virtual const uint8_t* get_codes(size_t list_no) const = 0;

    /** get the ids for an inverted list
     * must be released by release_ids
     *
     * @return ids      size list_size
     */
    virtual const idx_t* get_ids(size_t list_no) const = 0;

    /// release codes returned by get_codes (default implementation is nop
    virtual void release_codes(size_t list_no, const uint8_t* codes) const;

    /// release ids returned by get_ids
    virtual void release_ids(size_t list_no, const idx_t* ids) const;

    /// @return a single id in an inverted list
    virtual idx_t get_single_id(size_t list_no, size_t offset) const;

    /// @return a single code in an inverted list
    /// (should be deallocated with release_codes)
    virtual const uint8_t* get_single_code(size_t list_no, size_t offset) const;

    /// prepare the following lists (default does nothing)
    /// a list can be -1 hence the signed long
    virtual void prefetch_lists(const idx_t* list_nos, int nlist) const;

    /*****************************************
     * Iterator interface (with context)     */

    /// check if the list is empty
    virtual bool is_empty(size_t list_no, void* inverted_list_context = nullptr)
            const;

    /// get iterable for lists that use_iterator
    virtual InvertedListsIterator* get_iterator(
            size_t list_no,
            void* inverted_list_context = nullptr) const;

    /*************************
     * writing functions     */

    /// add one entry to an inverted list
    virtual size_t add_entry(
            size_t list_no,
            idx_t theid,
            const uint8_t* code,
            void* inverted_list_context = nullptr);

    virtual size_t add_entries(
            size_t list_no,
            size_t n_entry,
            const idx_t* ids,
            const uint8_t* code) = 0;

    virtual void update_entry(
            size_t list_no,
            size_t offset,
            idx_t id,
            const uint8_t* code);

    virtual void update_entries(
            size_t list_no,
            size_t offset,
            size_t n_entry,
            const idx_t* ids,
            const uint8_t* code) = 0;

    virtual void resize(size_t list_no, size_t new_size) = 0;

    virtual void reset();

    /*************************
     * high level functions  */

    /// move all entries from oivf (empty on output)
    void merge_from(InvertedLists* oivf, size_t add_id);

    // how to copy a subset of elements from the inverted lists
    // This depends on two integers, a1 and a2.
    enum subset_type_t : int {
        // depends on IDs
        SUBSET_TYPE_ID_RANGE = 0, // copies ids in [a1, a2)
        SUBSET_TYPE_ID_MOD = 1,   // copies ids if id % a1 == a2
        // depends on order within invlists
        SUBSET_TYPE_ELEMENT_RANGE =
                2, // copies fractions of invlists so that a1 elements are left
                   // before and a2 after
        SUBSET_TYPE_INVLIST_FRACTION =
                3, // take fraction a2 out of a1 from each invlist, 0 <= a2 < a1
        // copy only inverted lists a1:a2
        SUBSET_TYPE_INVLIST = 4
    };

    /** copy a subset of the entries index to the other index
     * @return number of entries copied
     */
    size_t copy_subset_to(
            InvertedLists& other,
            subset_type_t subset_type,
            idx_t a1,
            idx_t a2) const;

    /*************************
     * statistics            */

    /// 1= perfectly balanced, >1: imbalanced
    double imbalance_factor() const;

    /// display some stats about the inverted lists
    void print_stats() const;

    /// sum up list sizes
    size_t compute_ntotal() const;

    /**************************************
     * Scoped inverted lists (for automatic deallocation)
     *
     * instead of writing:
     *
     *     uint8_t * codes = invlists->get_codes (10);
     *     ... use codes
     *     invlists->release_codes(10, codes)
     *
     * write:
     *
     *    ScopedCodes codes (invlists, 10);
     *    ... use codes.get()
     *    // release called automatically when codes goes out of scope
     *
     * the following function call also works:
     *
     *    foo (123, ScopedCodes (invlists, 10).get(), 456);
     *
     */

    struct ScopedIds {
        const InvertedLists* il;
        const idx_t* ids;
        size_t list_no;

        ScopedIds(const InvertedLists* il, size_t list_no)
                : il(il), ids(il->get_ids(list_no)), list_no(list_no) {}

        const idx_t* get() {
            return ids;
        }

        idx_t operator[](size_t i) const {
            return ids[i];
        }

        ~ScopedIds() {
            il->release_ids(list_no, ids);
        }
    };

    struct ScopedCodes {
        const InvertedLists* il;
        const uint8_t* codes;
        size_t list_no;

        ScopedCodes(const InvertedLists* il, size_t list_no)
                : il(il), codes(il->get_codes(list_no)), list_no(list_no) {}

        ScopedCodes(const InvertedLists* il, size_t list_no, size_t offset)
                : il(il),
                  codes(il->get_single_code(list_no, offset)),
                  list_no(list_no) {}

        const uint8_t* get() {
            return codes;
        }

        ~ScopedCodes() {
            il->release_codes(list_no, codes);
        }
    };
};

/// simple (default) implementation as an array of inverted lists
struct ArrayInvertedLists : InvertedLists {
    int version;
    std::vector<size_t> sizes2;
    std::vector<uint8_t*> codes2; // binary codes, size nlist
    std::vector<idx_t*> ids2;     ///< Inverted lists for indexes
    std::vector<std::vector<uint8_t>> codes; // binary codes, size nlist
    std::vector<std::vector<idx_t>> ids;     ///< Inverted lists for indexes

    ArrayInvertedLists(size_t nlist, size_t code_size);

    size_t list_size(size_t list_no) const override;
    const uint8_t* get_codes(size_t list_no) const override;
    const idx_t* get_ids(size_t list_no) const override;

    size_t add_entries(
            size_t list_no,
            size_t n_entry,
            const idx_t* ids,
            const uint8_t* code) override;

    void update_entries(
            size_t list_no,
            size_t offset,
            size_t n_entry,
            const idx_t* ids,
            const uint8_t* code) override;

    void resize(size_t list_no, size_t new_size) override;

    /// permute the inverted lists, map maps new_id to old_id
    void permute_invlists(const idx_t* map);

    bool is_empty(size_t list_no, void* inverted_list_context = nullptr)
            const override;

    ~ArrayInvertedLists() override;
};

/*****************************************************************
 * Meta-inverted lists
 *
 * About terminology: the inverted lists are seen as a sparse matrix,
 * that can be stacked horizontally, vertically and sliced.
 *****************************************************************/

/// invlists that fail for all write functions
struct ReadOnlyInvertedLists : InvertedLists {
    ReadOnlyInvertedLists(size_t nlist, size_t code_size)
            : InvertedLists(nlist, code_size) {}

    size_t add_entries(
            size_t list_no,
            size_t n_entry,
            const idx_t* ids,
            const uint8_t* code) override;

    void update_entries(
            size_t list_no,
            size_t offset,
            size_t n_entry,
            const idx_t* ids,
            const uint8_t* code) override;

    void resize(size_t list_no, size_t new_size) override;
};

/// Horizontal stack of inverted lists
struct HStackInvertedLists : ReadOnlyInvertedLists {
    std::vector<const InvertedLists*> ils;

    /// build InvertedLists by concatenating nil of them
    HStackInvertedLists(int nil, const InvertedLists** ils);

    size_t list_size(size_t list_no) const override;
    const uint8_t* get_codes(size_t list_no) const override;
    const idx_t* get_ids(size_t list_no) const override;

    void prefetch_lists(const idx_t* list_nos, int nlist) const override;

    void release_codes(size_t list_no, const uint8_t* codes) const override;
    void release_ids(size_t list_no, const idx_t* ids) const override;

    idx_t get_single_id(size_t list_no, size_t offset) const override;

    const uint8_t* get_single_code(size_t list_no, size_t offset)
            const override;
};

using ConcatenatedInvertedLists = HStackInvertedLists;

/// vertical slice of indexes in another InvertedLists
struct SliceInvertedLists : ReadOnlyInvertedLists {
    const InvertedLists* il;
    idx_t i0, i1;

    SliceInvertedLists(const InvertedLists* il, idx_t i0, idx_t i1);

    size_t list_size(size_t list_no) const override;
    const uint8_t* get_codes(size_t list_no) const override;
    const idx_t* get_ids(size_t list_no) const override;

    void release_codes(size_t list_no, const uint8_t* codes) const override;
    void release_ids(size_t list_no, const idx_t* ids) const override;

    idx_t get_single_id(size_t list_no, size_t offset) const override;

    const uint8_t* get_single_code(size_t list_no, size_t offset)
            const override;

    void prefetch_lists(const idx_t* list_nos, int nlist) const override;
};

struct VStackInvertedLists : ReadOnlyInvertedLists {
    std::vector<const InvertedLists*> ils;
    std::vector<idx_t> cumsz;

    /// build InvertedLists by concatenating nil of them
    VStackInvertedLists(int nil, const InvertedLists** ils);

    size_t list_size(size_t list_no) const override;
    const uint8_t* get_codes(size_t list_no) const override;
    const idx_t* get_ids(size_t list_no) const override;

    void release_codes(size_t list_no, const uint8_t* codes) const override;
    void release_ids(size_t list_no, const idx_t* ids) const override;

    idx_t get_single_id(size_t list_no, size_t offset) const override;

    const uint8_t* get_single_code(size_t list_no, size_t offset)
            const override;

    void prefetch_lists(const idx_t* list_nos, int nlist) const override;
};

/** use the first inverted lists if they are non-empty otherwise use the second
 *
 * This is useful if il1 has a few inverted lists that are too long,
 * and that il0 has replacement lists for those, with empty lists for
 * the others. */
struct MaskedInvertedLists : ReadOnlyInvertedLists {
    const InvertedLists* il0;
    const InvertedLists* il1;

    MaskedInvertedLists(const InvertedLists* il0, const InvertedLists* il1);

    size_t list_size(size_t list_no) const override;
    const uint8_t* get_codes(size_t list_no) const override;
    const idx_t* get_ids(size_t list_no) const override;

    void release_codes(size_t list_no, const uint8_t* codes) const override;
    void release_ids(size_t list_no, const idx_t* ids) const override;

    idx_t get_single_id(size_t list_no, size_t offset) const override;

    const uint8_t* get_single_code(size_t list_no, size_t offset)
            const override;

    void prefetch_lists(const idx_t* list_nos, int nlist) const override;
};

/** if the inverted list in il is smaller than maxsize then return it,
 *  otherwise return an empty invlist */
struct StopWordsInvertedLists : ReadOnlyInvertedLists {
    const InvertedLists* il0;
    size_t maxsize;

    StopWordsInvertedLists(const InvertedLists* il, size_t maxsize);

    size_t list_size(size_t list_no) const override;
    const uint8_t* get_codes(size_t list_no) const override;
    const idx_t* get_ids(size_t list_no) const override;

    void release_codes(size_t list_no, const uint8_t* codes) const override;
    void release_ids(size_t list_no, const idx_t* ids) const override;

    idx_t get_single_id(size_t list_no, size_t offset) const override;

    const uint8_t* get_single_code(size_t list_no, size_t offset)
            const override;

    void prefetch_lists(const idx_t* list_nos, int nlist) const override;
};

} // namespace faiss

#endif
