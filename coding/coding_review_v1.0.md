>   "@@@" - tech problem; "!!!" - cannot solve at the first time; "+++" - some useful trick; "///" - try later.

# Data Structure

## Array - Double Pointers

### Binary Search

#### 34. Find First and Last Position of Element in Sorted Array

*   ![image-20220628135534186](coding_review_v1.0.assets/image-20220628135534186.png)

*   Thoughts:

    *   The nums is sorted in non-decreading order, so the brute force methord is traversing.
    *   If we traverse the nums, the time complexity is $O(n)$. To achieve $O(\log n)$ for a list, adopt binary search.
    *   Also, we need to run binary search twice to find the start and end. Cauze if we only find the start index and traverse to find the end index, the worst time complexity will be $O(n)$
    *   For binary search, we first decide the interval is open or closed. Closed interval: $[left, right]$
    *   $mid = left + ((right - left) >> 2)$
    *   For the closed interval and we want to return when we find the target, the condition for `while` should be `left <= right`
    *   the equation we used to narrow the search range leads to start and end index
    *   To find the start index: even the nums[mid] = target, we still need to narrow the range to the left side

*   Solution

    *   ```python
        class Solution:
            def searchRange(self, nums: List[int], target: int) -> List[int]:
                size = len(nums)
                # first we find start index
                left, right = 0, size - 1
                while left <= right:
                    mid = left + ((right - left) >> 2)
                    if nums[mid] > target:
                        right = mid - 1
                    elif nums[mid] < target:
                        left = mid + 1
                    elif nums[mid] == target:
                        right = mid - 1
                start = left
                if start < 0 or start >= size or nums[start] != target:
                    return [-1, -1]
                
                # then we find end index
                left, right = 0, size - 1
                while left <= right:
                    mid = left + ((right - left) >> 2)
                    if nums[mid] > target:
                        right = mid - 1
                    elif nums[mid] < target:
                        left = mid + 1
                    elif nums[mid] == target:
                        left = mid + 1
                end = right
                return [start, end]
        
        ```

    

#### 704. Binary Search

*   Basic one, no thought

*   Solution:

    *   ```python
        class Solution:
            def search(self, nums: List[int], target: int) -> int:
                left, right = 0, len(nums) - 1
                while left <= right:
                    mid = left + ((right - left) >> 2)
                    if nums[mid] < target:
                        left = mid + 1
                    elif nums[mid] > target:
                        right = mid - 1
                    elif nums[mid] == target:
                        return mid
                return -1
            
        ```

    *   

#### 35. Search Insert Position

*   Thought:

    *   Apply open interval cause we want the insert position which is exactly one index.

*   Solution:

    *   ```python
        class Solution:
            def searchInsert(self, nums: List[int], target: int) -> int:
                # open interval [left, right)
                left, right = 0, len(nums)
                while left < right:
                    mid = left + ((right - left) >> 2)
                    if nums[mid] < target:
                        left = mid + 1
                    elif nums[mid] > target:
                        right = mid
                    elif nums[mid] == target:
                        # return the target position
                        return mid
                # return the insert position
                return left
                
        ```

    *   

#### !!! 354. Russian Doll Envelopes

*   ![image-20220628150031466](coding_review_v1.0.assets/image-20220628150031466.png)
*   Thoughts:
    *   Firstly, I'd like to sort the envelopes `envelopes.sort(key=lambda x:(x[0], -x[1]))` this coust $O(n\log n)$ in the worst case.
    *   Brute force search will cost $O(2^n)$.
    *   After sorting, this becomes dp problem. for each envelop, we give the maximum number it can contain. We can see that the subproblems are independent with each other. 
    *   dp: State "current_envelop"; selection: select the last envelop?
    *   ...
*   Solution
    *   The general dp is $O(n^2)$ exceeds the time limit.

#### 392.

#### 793. 

#### 875. 

#### 1011. 

### Sliding Window

### Else

## Linked List - Double Pointers



