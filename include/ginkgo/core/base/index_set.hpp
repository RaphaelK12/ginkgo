/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2020, the Ginkgo authors
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
contributors may be used to endorse or promote products derived from
this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************<GINKGO LICENSE>*******************************/

#ifndef GKO_CORE_BASE_INDEX_SET_HPP_
#define GKO_CORE_BASE_INDEX_SET_HPP_


#include <algorithm>
#include <mutex>
#include <vector>


#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/base/utils.hpp>


namespace gko {


template <typename IndexType = int32>
class IndexSet {
public:
    class ElementIterator;
    class IntervalIterator;

    using index_type = IndexType;

    /**
     * Default constructor.
     */
    IndexSet() noexcept
        : is_merged_(true),
          largest_subset_(invalid_unsigned_int),
          index_space_size_(0)
    {}

    /**
     * Constructor that also sets the overall size of the index range.
     */
    explicit IndexSet(const size_type size)
        : is_merged_(true),
          largest_subset_(invalid_unsigned_int),
          index_space_size_(size)
    {}

    /**
     * Copy constructor.
     */
    IndexSet(const IndexSet &) = default;

    /**
     * Copy assignment operator.
     */
    IndexSet &operator=(const IndexSet &) = default;

    /**
     * Move constructor. Create a new IndexSet by transferring the internal data
     * of the input set.
     */
    IndexSet(IndexSet &&other) noexcept;

    /**
     * Move assignment operator. Transfer the internal data of the input set
     * into the current one.
     */
    IndexSet &operator=(IndexSet &&is) noexcept;


    /**
     * Return the size of the index space of which this index set is a subset
     * of.
     *
     * Note that the result is not equal to the number of indices within this
     * set. The latter information is returned by get_num_elements().
     */
    size_type get_size() const { return index_space_size_; }

    /**
     * Set the maximal size of the indices upon which this object operates.
     *
     * This function can only be called if the index set does not yet contain
     * any elements.  This can be achieved by calling clear(), for example.
     */
    void set_size(const size_type input_size)
    {
        GKO_ASSERT(subsets_.empty());
        index_space_size_ = input_size;
        is_merged_ = true;
    }

    /**
     * Merge the internal representation by merging individual elements with
     * contiguous subsets, etc. This function does not have any external effect.
     */
    void merge() const;

    /**
     * Add the half-open subset $[\text{begin},\text{end})$ to the set of
     * indices represented by this class.
     * @param[in] begin The first element of the subset to be added.
     * @param[in] end The past-the-end element of the subset to be added.
     */
    void add_subset(const size_type begin, const size_type end);

    /**
     * Add an individual index to the set of indices.
     */
    void add_index(const size_type index);


    /**
     * Add the given IndexSet @p other to the current one, constructing the
     * union of *this and @p other.
     *
     * If the @p offset argument is nonzero, then every index in @p other is
     * shifted by @p offset before being added to the current index set. This
     * allows to construct, for example, one index set from several others that
     * are supposed to represent index sets corresponding to different subsets_
     * (e.g., when constructing the set of nonzero entries of a block vector
     * from the sets of nonzero elements of the individual blocks of a vector).
     *
     * This function will generate an exception if any of the (possibly shifted)
     * indices of the @p other index set lie outside the subset
     * <code>[0,size())</code> represented by the current object.
     */
    void add_indices(const IndexSet &other, const size_type offset);

    /**
     * Add a whole set of indices described by dereferencing every element of
     * the iterator subset <code>[begin,end)</code>.
     *
     * @param[in] begin Iterator to the first element of subset of indices to be
     * added
     * @param[in] end The past-the-end iterator for the subset of elements to be
     * added. @pre The condition <code>begin@<=end</code> needs to be satisfied.
     */
    template <typename ForwardIterator>
    void add_indices(const ForwardIterator &begin, const ForwardIterator &end);

    /**
     * Return whether the specified index is an element of the index set.
     */
    bool is_element(const size_type index) const;

    /**
     * Return whether the index set stored by this object defines a contiguous
     * subset. This is true also if no indices are stored at all.
     */
    bool is_contiguous() const;

    /**
     * Return whether the index set stored by this object contains no elements.
     * This is similar, but faster than checking <code>get_num_elements() ==
     * 0</code>.
     */
    bool is_empty() const { return subsets_.empty(); }

    /**
     * Return whether the IndexSets are ascending with respect to MPI process
     * number and 1:1, i.e., each index is contained in exactly one IndexSet
     * (among those stored on the different processes), each process stores
     * contiguous subset of indices, and the index set on process $p+1$ starts
     * at the index one larger than the last one stored on process $p$.
     * In case there is only one MPI process, this just means that the IndexSet
     * is complete.
     */
    bool is_ascending_and_one_to_one(const gko::Executor *exec) const;


    /**
     * Return the number of elements stored in this index set.
     */
    size_type get_num_elements() const;

    /**
     * Return the global index of the local index with number @p local_index
     * stored in this index set. @p local_index obviously needs to be less than
     * get_num_elements().
     */
    size_type get_global_index(const size_type local_index) const;

    /**
     * Return the how-manyth element of this set (counted in ascending order) @p
     * global_index is. @p global_index needs to be less than the size(). This
     * function returns numbers::invalid_dof_index if the index @p global_index
     * is not actually a member of this index set, i.e. if
     * is_element(global_index) is false.
     */
    size_type get_local_index(const size_type global_index) const;

    /**
     * Each index set can be represented as the union of a number of contiguous
     * intervals of indices, where if necessary intervals may only consist of
     * individual elements to represent isolated members of the index set.
     *
     * This function returns the minimal number of such intervals that are
     * needed to represent the index set under consideration.
     */
    unsigned int get_num_subsets() const;

    /**
     * This function returns the local index of the beginning of the largest
     * subset.
     *
     * In other words, the return value is superset_index(x), where x is the
     * first index of the largest contiguous subset of indices in the
     * IndexSet. The return value is therefore equal to the number of elements
     * in the set that come before the largest subset.
     *
     * This call assumes that the IndexSet is nonempty.
     */
    size_type largest_subset_starting_index() const;

    /**
     * Comparison for equality of index sets. This operation is only allowed if
     * the size of the two sets is the same (though of course they do not have
     * to have the same number of indices).
     */
    bool operator==(const IndexSet &other) const;

    /**
     * Comparison for inequality of index sets. This operation is only allowed
     * if the size of the two sets is the same (though of course they do not
     * have to have the same number of indices).
     */
    bool operator!=(const IndexSet &other) const;

    /**
     * Return the intersection of the current index set and the argument given,
     * i.e. a set of indices that are elements of both index sets. The two index
     * sets must have the same size (though of course they do not have to have
     * the same number of indices).
     */
    IndexSet operator&(const IndexSet &other) const;


    /**
     * Remove all elements contained in @p other from this set. In other words,
     * if $x$ is the current object and $o$ the argument, then we compute $x
     * \leftarrow x \backslash o$.
     */
    void subtract_set(const IndexSet &other);


    /**
     * Remove and return the last element of the last subset.
     * This function throws an exception if the IndexSet is empty.
     */
    size_type pop_back();

    /**
     * Remove and return the first element of the first subset.
     * This function throws an exception if the IndexSet is empty.
     */
    size_type pop_front();

    /**
     * Remove all indices from this index set. The index set retains its size,
     * however.
     */
    void clear()
    {
        // reset so that there are no indices in the set any more; however,
        // as documented, the index set retains its size
        subsets_.clear();
        is_merged_ = true;
        largest_subset_ = invalid_unsigned_int;
    }


    /**
     * Dereferencing an IntervalIterator will return a reference to an object of
     * this type. It allows access to a contiguous interval $[a,b[$ (also called
     * a subset) of the IndexSet being iterated over.
     * This class is a modified version of the class from the deal.ii library.
     */
    class IntervalAccessor {
    public:
        /**
         * Construct a valid accessor given an IndexSet and the index @p
         * subset_idx of the subset to point to.
         */
        IntervalAccessor(const IndexSet *idxset, const size_type subset_idx)
            : index_set_(idxset), subset_idx_(subset_idx)
        {}

        /**
         * Construct an invalid accessor for the IndexSet.
         */
        explicit IntervalAccessor(const IndexSet *idxset)
            : index_set_(idxset), subset_idx_(invalid_index)
        {}

        /**
         * Number of elements in this interval.
         */
        size_type get_num_elements() const
        {
            return index_set_->subsets_[subset_idx_].end_ -
                   index_set_->subsets_[subset_idx_].begin_;
        }

        /**
         * If true, we are pointing at a valid interval in the IndexSet.
         */
        bool is_valid() const
        {
            return index_set_ != nullptr &&
                   subset_idx_ < index_set_->get_num_elements();
        }

        /**
         * Return an iterator pointing at the first index in this interval.
         */
        ElementIterator begin() const
        {
            GKO_ASSERT(is_valid());
            return {index_set_, subset_idx_,
                    index_set_->subsets_[subset_idx_].begin_};
        }

        /**
         * Return an iterator pointing directly after the last index in this
         * interval.
         */
        ElementIterator end() const
        {
            GKO_ASSERT(is_valid());
            if (subset_idx_ < index_set_->subsets_.size() - 1)
                return {index_set_, subset_idx_ + 1,
                        index_set_->subsets_[subset_idx_ + 1].begin_};
            else
                return index_set_->end();
        }

        /**
         * Return the index of the last index in this interval.
         */
        size_type last() const
        {
            GKO_ASSERT(is_valid());
            return index_set_->subsets_[subset_idx_].end_ - 1;
        }

    private:
        /**
         * Private copy constructor.
         */
        IntervalAccessor(const IntervalAccessor &other)
            : index_set_(other.index_set_), subset_idx_(other.subset_idx_)
        {}

        /**
         * Private copy operator.
         */
        IntervalAccessor &operator=(const IntervalAccessor &other)
        {
            index_set_ = other.index_set_;
            subset_idx_ = other.subset_idx_;
            GKO_ASSERT(subset_idx_ == invalid_index || is_valid());
            return *this;
        }

        /**
         * Test for equality, used by IntervalIterator.
         */
        bool operator==(const IntervalAccessor &other) const
        {
            GKO_ASSERT(index_set_ == other.index_set_);
            return subset_idx_ == other.subset_idx_;
        }

        /**
         * Smaller-than operator, used by IntervalIterator.
         */
        bool operator<(const IntervalAccessor &other) const
        {
            GKO_ASSERT(index_set_ == other.index_set_);
            return subset_idx_ < other.subset_idx_;
        }

        /**
         * Advance this accessor to point to the next interval in the @p
         * index_set.
         */
        void advance()
        {
            GKO_ASSERT(is_valid());
            ++subset_idx_;

            // set ourselves to invalid if we walk off the end
            if (subset_idx_ >= index_set_->subsets_.size()) {
                subset_idx_ = invalid_index;
            }
        }

        /**
         * Reference to the IndexSet.
         */
        const IndexSet *index_set_;

        /**
         * Index into index_set.subsets[]. Set to numbers::invalid_dof_index if
         * invalid or the end iterator.
         */
        size_type subset_idx_;

        friend class IntervalIterator;
    };

    /**
     * Class that represents an iterator pointing to a contiguous interval
     * $[a,b[$ as returned by IndexSet::begininterval().
     * This class a modified version of the class from the deal.ii library.
     */
    class IntervalIterator {
    public:
        /**
         * Construct a valid iterator pointing to the interval with index @p
         * subset_idx.
         */
        IntervalIterator(const IndexSet *idxset, const size_type subset_idx)
            : accessor_(idxset, subset_idx)
        {}

        /**
         * Construct an invalid iterator (used as end()).
         */
        explicit IntervalIterator(const IndexSet *idxset) : accessor_(idxset) {}

        /**
         * Construct an empty iterator.
         */
        IntervalIterator() : accessor_(nullptr) {}

        /**
         * Copy constructor from @p other iterator.
         */
        IntervalIterator(const IntervalIterator &other) = default;

        /**
         * Assignment of another iterator.
         */
        IntervalIterator &operator=(const IntervalIterator &other) = default;

        /**
         * Prefix increment.
         */
        IntervalIterator &operator++()
        {
            accessor_.advance();
            return *this;
        }

        /**
         * Postfix increment.
         */
        IntervalIterator operator++(int)
        {
            const IndexSet::IntervalIterator iter = *this;
            accessor_.advance();
            return iter;
        }

        /**
         * Dereferencing operator, returns an IntervalAccessor.
         */
        const IntervalAccessor &operator*() const { return accessor_; }

        /**
         * Dereferencing operator, returns a pointer to an IntervalAccessor.
         */
        const IntervalAccessor *operator->() const { return &accessor_; }

        /**
         * Comparison.
         */
        bool operator==(const IntervalIterator &other) const
        {
            return accessor_ == other.accessor_;
        }

        /**
         * Inverse of <tt>==</tt>.
         */
        bool operator!=(const IntervalIterator &other) const
        {
            return !(*this == other);
        }

        /**
         * Comparison operator.
         */
        bool operator<(const IntervalIterator &other) const
        {
            return accessor_ < other.accessor_;
        }

        /**
         * Return the distance between the current iterator and the argument.
         * The distance is given by how many times one has to apply operator++
         * to the current iterator to get the argument (for a positive return
         * value), or operator-- (for a negative return value).
         */
        int operator-(const IntervalIterator &other) const
        {
            GKO_ASSERT(accessor_.index_set_ == other.accessor_.index_set_);

            const size_type lhs = (accessor_.subset_idx_ == invalid_index)
                                      ? accessor_.index_set_->subsets_.size()
                                      : accessor_.subset_idx_;
            const size_type rhs = (other.accessor_.subset_idx_ == invalid_index)
                                      ? accessor_.index_set_->subsets_.size()
                                      : other.accessor_.subset_idx_;

            if (lhs > rhs)
                return static_cast<int>(lhs - rhs);
            else
                return -static_cast<int>(rhs - lhs);
        }

        /**
         * Mark the class as forward iterator and declare some alias which are
         * standard for iterators and are used by algorithms to enquire about
         * the specifics of the iterators they work on.
         */
        using iterator_category = std::forward_iterator_tag;
        using value_type = IntervalAccessor;
        using difference_type = std::ptrdiff_t;
        using pointer = IntervalAccessor *;
        using reference = IntervalAccessor &;

    private:
        /**
         * Accessor that contains what IndexSet and interval we are pointing at.
         */
        IntervalAccessor accessor_;
    };

    /**
     * Class that represents an iterator pointing to a single element in the
     * IndexSet as returned by IndexSet::begin().
     */
    class ElementIterator {
    public:
        /**
         * Construct an iterator pointing to the global index @p index in the
         * interval @p subset_index
         */
        ElementIterator(const IndexSet *indexset, const size_type subset_index,
                        const size_type index)
            : index_set_(indexset), subset_index_(subset_index), index_(index)
        {
            GKO_ASSERT(subset_index_ < index_set_->subsets_.size());
            GKO_ASSERT(index_ >= index_set_->subsets_[subset_index_].begin_ &&
                       index_ < index_set->subsets_[subset_index_].end_);
        }

        /**
         * Construct an iterator pointing to the end of the IndexSet.
         */
        explicit ElementIterator(const IndexSet *index_set)
            : index_set_(index_set),
              subset_index_(invalid_index),
              index_(invalid_index)
        {}

        /**
         * Does this iterator point to an existing element?
         */
        bool is_valid() const
        {
            GKO_ASSERT(
                (subset_index_ == invalid_index && index_ == invalid_index) ||
                (subset_index_ < index_set_->subsets_.size() &&
                 index_ < index_set_->subsets_[subset_index_].end_));

            return (subset_index_ < index_set_->subsets_.size() &&
                    index_ < index_set_->subsets_[subset_index_].end_);
        }

        /**
         * Dereferencing operator. The returned value is the index of the
         * element inside the IndexSet.
         */
        size_type operator*() const
        {
            GKO_ASSERT(is_valid());
            return index_;
        }

        /**
         * Prefix increment.
         */
        ElementIterator &operator++()
        {
            advance();
            return *this;
        }

        /**
         * Postfix increment.
         */
        ElementIterator operator++(int)
        {
            const IndexSet::ElementIterator it = *this;
            advance();
            return it;
        }

        /**
         * Comparison.
         */
        bool operator==(const ElementIterator &other) const
        {
            GKO_ASSERT(index_set_ == other.index_set_);
            return subset_index_ == other.subset_index_ &&
                   index_ == other.index_;
        }

        /**
         * Inverse of <tt>==</tt>.
         */
        bool operator!=(const ElementIterator &other) const
        {
            return !(*this == other);
        }

        /**
         * Comparison operator.
         */
        bool operator<(const ElementIterator &other) const
        {
            GKO_ASSERT(index_set_ == other.index_set_);
            return subset_index_ < other.subset_index_ ||
                   (subset_index_ == other.subset_index_ &&
                    index_ < other.index_);
        }

        /**
         * Return the distance between the current iterator and the argument. In
         * the expression <code>it_left-it_right</code> the distance is given by
         * how many times one has to apply operator++ to the right operand @p
         * it_right to get the left operand @p it_left (for a positive return
         * value), or to @p it_left to get the @p it_right (for a negative
         * return value).
         */
        std::ptrdiff_t operator-(const ElementIterator &other) const
        {
            GKO_ASSERT(index_set_ == other.index_set_);
            if (*this == other) return 0;
            if (!(*this < other)) return -(other - *this);

            // only other can be equal to end() because of the checks above.
            GKO_ASSERT(is_valid());

            // Note: we now compute how far advance *this in "*this < other" to
            // get other, so we need to return -c at the end.

            // first finish the current subset:
            std::ptrdiff_t c =
                index_set_->subsets_[subset_index_].end_ - index_;

            // now walk in steps of subsets_ (need to start one behind our
            // current one):
            for (size_type subset = subset_index_ + 1;
                 subset < index_set_->subsets_.size() &&
                 subset <= other.subset_index_;
                 ++subset)
                c += index_set_->subsets_[subset].end_ -
                     index_set_->subsets_[subset].begin_;

            GKO_ASSERT(other.subset_index_ < index_set_->subsets_.size() ||
                       other.subset_index_ == invalid_index);

            // We might have walked too far because we went until the end of
            // other.subset_index, so walk backwards to other.index:
            if (other.subset_index_ != invalid_index)
                c -= index_set_->subsets_[other.subset_index_].end_ -
                     other.index_;

            return -c;
        }

        /**
         * Mark the class as forward iterator and declare some alias which are
         * standard for iterators and are used by algorithms to enquire about
         * the specifics of the iterators they work on.
         */
        using iterator_category = std::forward_iterator_tag;
        using value_type = size_type;
        using difference_type = std::ptrdiff_t;
        using pointer = size_type *;
        using reference = size_type &;

    private:
        /**
         * Advance iterator by one.
         */
        void advance()
        {
            GKO_ASSERT(is_valid());
            if (index_ < index_set_->subsets_[subset_index_].end_) ++index_;
            // end of this subset?
            if (index_ == index_set_->subsets_[subset_index_].end_) {
                // point to first element in next interval if possible
                if (subset_index_ < index_set_->subsets_.size() - 1) {
                    ++subset_index_;
                    index_ = index_set_->subsets_[subset_index_].begin_;
                } else {
                    // we just fell off the end, set to invalid:
                    subset_index_ = invalid_index;
                    index_ = invalid_index;
                }
            }
        }

        /**
         * The parent IndexSet.
         */
        const IndexSet *index_set_;

        /**
         * Index into set
         */
        size_type subset_index_;

        /**
         * The global index this iterator is pointing at.
         */
        size_type index_;
    };

    /**
     * Return an iterator that points at the first index that is contained in
     * this IndexSet.
     */
    ElementIterator begin() const;

    /**
     * Return an element iterator pointing to the element with global index
     * @p global_index or the next larger element if the index is not in the
     * set. This is equivalent to
     * @code
     * auto p = begin();
     * while (*p<global_index)
     *   ++p;
     * return p;
     * @endcode
     *
     * If there is no element in this IndexSet at or behind @p global_index,
     * this method will return end().
     */
    ElementIterator at(const size_type global_index) const;

    /**
     * Return an iterator that points one after the last index that is contained
     * in this IndexSet.
     */
    ElementIterator end() const;

    /**
     * Return an Iterator that points at the first interval of this IndexSet.
     */
    IntervalIterator first_interval() const;

    /**
     * Return an Iterator that points one after the last interval of this
     * IndexSet.
     */
    IntervalIterator last_interval() const;

private:
    struct subset {
        subset() = delete;
        subset(const size_type begin, const size_type end)
            : begin_(begin), end_(end), superset_index_(invalid_index)
        {}

        friend inline bool operator<(const subset &subset1,
                                     const subset &subset2)
        {
            return ((subset1.begin_ < subset2.begin_) ||
                    ((subset1.begin_ == subset2.begin_) &&
                     (subset1.end_ < subset2.end_)));
        }

        static bool compare_end(const IndexSet::subset &x,
                                const IndexSet::subset &y)
        {
            return x.end_ < y.end_;
        }

        static bool superset_index_compare(const IndexSet::subset &x,
                                           const IndexSet::subset &y)
        {
            return (x.superset_index_ + (x.end_ - x.begin_) <
                    y.superset_index_ + (y.end_ - y.begin_));
        }

        friend inline bool operator==(const subset &subset1,
                                      const subset &subset2)
        {
            return ((subset1.begin_ == subset2.begin_) &&
                    (subset1.end_ == subset2.end_));
        }


        size_type begin_;
        size_type end_;
        size_type superset_index_;
    };

    void merge_impl() const;

    mutable bool is_merged_;
    mutable std::vector<subset> subsets_;
    mutable size_type largest_subset_;
    std::mutex merge_mutex_;
    size_type index_space_size_;
};


}  // namespace gko


#endif  // GKO_CORE_BASE_INDEX_SET_HPP_
