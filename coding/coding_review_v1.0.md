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

#### 392. Is Subsequence

*   ![image-20220629121017387](coding_review_v1.0.assets/image-20220629121017387.png)

*   Thoughts:

    *   At least traveser the father sequence. So the time complexity is $O(n)$

*   Solution:

    *   ```python
        class Solution:
            def isSubsequence(self, s: str, t: str) -> bool:
                s_pointer, t_pointer = 0, 0
                while s_pointer < len(s) and t_pointer < len(t):
                    s_char, t_char = s[s_pointer], t[t_pointer]
                    if s_char == t_char:
                        s_pointer += 1
                    t_pointer += 1
                if s_pointer == len(s):
                    return True
                return False
        ```

    *   

#### 793. Preimage Size of Factorial Zeroes Function

*   ![image-20220629150255859](coding_review_v1.0.assets/image-20220629150255859.png)

*   Thoughts:

    *   1 zero -> 10 -> 2\*5 -> the \# of 2 is always bigger than the \# of 5.
    *   So we turn to find the \# of 5
    *   0!, 1!, 2!, 3!, 4!, has no 5
    *   5!, 6!, 7!, 8!, 9!, has one 5 
    *   10! 11!, 12!, 13!, 14! has two 5
    *   ...
    *   when it turn to 25!, 26!, ... it increases 2 at this time.
    *   for each k, the output is 0 or 5.
    *   if we increase count +=1 the time complexity is $O(k)$.
    *   How to make it better?
    *   Construct f(n)
    *   and binary search

*   Solution:

    *   time limit exceed:

    *   ```python
        class Solution:
            def preimageSizeFZF(self, k: int) -> int:
                multi = 0
                count = 0
                while count < k:
                    cur = multi * 5
                    while cur > 5 and cur % 5 == 0:
                        count += 1
                        cur = int(cur / 5)
                    multi += 1
                return 5 if count == k else 0
            
        ```

    *   !!!

    *   ```python
        class Solution:
            def f_n(self, n):
                multi = 5
                count = 0
                while multi <= n:
                    count += (n // multi)
                    multi = multi * 5
                return count
                
            def preimageSizeFZF(self, k: int) -> int:
                # if find the target, return 5; otherwise 0
                # [left_n, right_n)
                left_n = 0
                right_n = 5 * (k + 1)
                while left_n < right_n:
                    mid_n = left_n + ((right_n - left_n) >> 2)
                    cur_k = self.f_n(mid_n)
                    if cur_k == k:
                        return 5
                    elif cur_k < k:
                        left_n = mid_n + 1
                    elif cur_k > k:
                        right_n = mid_n
                return 0
        
        ```

    *   

#### 875. Koko Eating Bananas

*   ![image-20220629155806352](coding_review_v1.0.assets/image-20220629155806352.png)

*   Thoughts:

    *   given the speed k, cost $O(n)$ to compute the h
    *   Can we try some reasonable k to meet h
    *   to find the minimum k -> binary search the left boundary.
    *   Can we give a search range at the very beginning? Yes, the greatest speed is the max(piles), this cost $O(n)$, at least speed k >= 1.
    *   how many k we need to check? not sure for now.

*   Solution:

    *   ```python
        class Solution:
            def compute_hours(self, piles: List[int], k: int) -> int:
                hour = 0
                for pile in piles:
                    hour += int(pile // k)
                    if pile % k != 0:
                        hour += 1
                return hour
                    
            def minEatingSpeed(self, piles: List[int], h: int) -> int:
                upper = max(piles) + 1
                # round down
                lower = 1
                # [lower, upper)
                # Note that the bigger k, the less hours
                while lower < upper:
                    mid = lower + ((upper - lower) >> 2)
                    hours = self.compute_hours(piles, mid)
                    # try to be sloer
                    if hours == h:
                        upper = mid
                    # could be slower
                    elif hours < h:
                        upper = mid
                    elif hours > h:
                        lower = mid + 1
                return lower
        
        ```

    *   

#### 1011. Capacity To Ship Packages Within D Days

*   ![image-20220629170504200](coding_review_v1.0.assets/image-20220629170504200.png)![image-20220629170638769](coding_review_v1.0.assets/image-20220629170638769.png)

*   Thoughts:

    *   The days is inversely related to load: more we load the less days we spend
    *   Given the target day, find the mim load.
    *   Min days: 1; max load: sum(weights); $O(n)$
    *   Max days: len(weights); min load: max(weights) $O(n)$
    *   We need a func given the load, return days
    *   **We search for the most right days**
    *   we search for the min load given the days
    *   Follow the binary search principles I mentioned in data_structure.md

*   Solution:

    *   ```python
        class Solution:
            def compute_days(self, weights: List[int], load: int) -> int:
                days = 0
                ship = 0
                for weight in weights:
                    ship += weight
                    if ship > load:
                        days += 1
                        ship = weight
                days += 1
                return days
                
            def shipWithinDays(self, weights: List[int], days: int) -> int:
                # we search for the most right days
                # we search for the min load given the days
                # out of loads
                lower_load = max(weights)
                upper_load = sum(weights)
                # [lower_load, upper_load] closed interval
                while lower_load <= upper_load:
                    mid_load = lower_load + ((upper_load - lower_load) >> 2)
                    expected_days = self.compute_days(weights, mid_load)
                    # should decrease the load
                    if expected_days < days:
                        upper_load = mid_load - 1
                    # should increase the load
                    elif expected_days > days:
                        lower_load = mid_load + 1
                    # try to decrease the load
                    elif expected_days == days:
                        upper_load = mid_load - 1
                # loop ends when lower_load == upper_load - 1 && upper_load = mid_load - 1 and we want mid_load
                return upper_load + 1
            
        ```

    *   

### Sliding Window

*   
*   Thoughts:
*   Solution:

#### 3. 

#### 76. 

#### 438. 

#### 567. 

#### 239. 

### Else

## Linked List - Double Pointers



