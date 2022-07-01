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

#### 3. Longest Substring Without Repeating Characters

*   ![image-20220630110716755](coding_review_v1.0.assets/image-20220630110716755.png)

*   Thoughts:

    *   We need to traverse the string $O(n)$
    *   can we traverse only once?
    *   we need a hashset window, whose find func cost $O(1)$
    *   we need left and right pointers /marks.

*   Solution:

    *   ```python
        class Solution:
            def lengthOfLongestSubstring(self, s: str) -> int:
                window = set()
                left = right = 0
                size = len(s)
                result = 0
                while right < size:
                    cur_char = s[right]
                    if cur_char not in window:
                        window.add(cur_char)
                        right += 1
                        continue
                    # cur_char is the repeating one
                    result = max(result, (right - 1) - left + 1)
                    while s[left] != cur_char:
                        window.remove(s[left])
                        left += 1
                    window.remove(s[left])
                    left += 1
                result = max(result, right - left)
                return result
        
        ```

    *   

#### 76. Minimum Window Substring

*   ![image-20220630112352177](coding_review_v1.0.assets/image-20220630112352177.png)![image-20220630121740123](coding_review_v1.0.assets/image-20220630121740123.png)

*   Thoughts: 

    *   We need a dictionary of t
    *   So we first build a hash map (char: number)
    *   How can we find the min substring by travesing only once
    *   left and right pointers, create a window to record current string in s.
    *   We need a valid num to meet the number of char.
    *   The diff is once we meet the req, we need to shrink the left then we can get the result.
    *   $O(n+m)$ for n, we visit each char in s twice at most.

*   !!! Solution:

    *   ```python
        class Solution:
            def add_to_map(self, char: str, req: dict):
                if char not in req:
                    req[char] = 1
                elif char in req:
                    req[char] += 1
                
            def minWindow(self, s: str, t: str) -> str:
                req = dict()
                for char in t:
                    self.add_to_map(char, req)
                req_size = len(req)
                    
                window = dict()
                left_mark, size_mark = 0, len(s) + 1
                left = right = 0
                valid = 0
                
                while right < len(s):
                    char_right = s[right]
                    if char_right in req:
                        self.add_to_map(char_right, window)
                        if window[char_right] == req[char_right]:
                            valid += 1
                    # left need shrink?
                    while valid == req_size:
                        # update result
                        # will be removed
                        char_left = s[left]
                        if char_left in req:
                            if (window[char_left] == req[char_left]):
                                if (right - left + 1) < size_mark:
                                    left_mark = left
                                    size_mark = right - left + 1
                                valid -= 1
                            window[char_left] -= 1
                        left += 1
                    right += 1
                    
                if size_mark > len(s):
                    return ""
                
                return s[left_mark: left_mark + size_mark]
            
        ```

    *   

#### 438. Find All Anagrams in a String

*   ![image-20220630154455151](coding_review_v1.0.assets/image-20220630154455151.png)

*   Thoughts:

    *   the order of p's letters is not important, so we store them in a hash map (letter: number)
    *   we traverse s with left and right pointers. The right one keep moving, and shrink the left one when the requirements meet.
    *   !!! **Note this one is different. the length of window is fixed.** 

*   Solution:

    *   ```python
        class Solution:
            def hash_add(self, hashmap: dict, key: str):
                if key not in hashmap:
                    hashmap[key] = 1
                else:
                    hashmap[key] += 1
                
            def findAnagrams(self, s: str, p: str) -> List[int]:
                need = dict()
                for char in p:
                    self.hash_add(need, char)
                
                left = right = 0
                window = dict()
                result = []
                need_num = len(need)
                valid = 0
                
                while right < len(s):
                    # for right pointer
                    char_right = s[right]
                    if char_right in need:
                        self.hash_add(window, char_right)
                        if window[char_right] == need[char_right]:
                            valid += 1
                    # for left pointer
                    while right - left + 1 >= len(p):
                        char_left = s[left]
                        if valid == need_num:
                            result.append(left)
                        if char_left in need:
                            if window[char_left] == need[char_left]:
                                valid -= 1
                            window[char_left] -= 1
                        left += 1
                    right += 1
                return result
                
        ```

    *   

#### 567. Permutation in String

*   ![image-20220630161435208](coding_review_v1.0.assets/image-20220630161435208.png)

*   Thoughts:

    *   Permutation, in other words, the order is not important, so I will create a hashmap "need" for s1
    *   left and right pointers traverse s2 with a window, find out whether window meets need.
    *   the size of window is Len(s1)
    *   I will write a func for add letters to hashmap.

*   Solution:

    *   ```python
        class Solution:
            def hash_add(self, hashmap: dict, key: str):
                if key in hashmap:
                    hashmap[key] += 1
                else:
                    hashmap[key] = 1
                
            def checkInclusion(self, s1: str, s2: str) -> bool:
                need = dict()
                for char in s1:
                    self.hash_add(need, char)
                
                left = right = 0
                valid = 0
                need_num = len(need)
                window = dict()
                
                while right < len(s2):
                    char_right = s2[right]
                    if char_right in need:
                        self.hash_add(window, char_right)
                        if window[char_right] == need[char_right]:
                            valid += 1
                    
                    if right - left + 1 == len(s1):
                        char_left = s2[left]
                        if valid == need_num:
                            return True
                        if char_left in need:
                            if window[char_left] == need[char_left]:
                                valid -= 1
                            window[char_left] -= 1
                        left += 1
                    right += 1
                
                return False
            
        ```

    *   

#### 239. Sliding Window Maximum

*   ![image-20220630172018030](coding_review_v1.0.assets/image-20220630172018030.png)

*   Thoughts:

    *   If we simplely move the window and find the max for each window, it cost $O(k * (n - k + 1))$. the worst one will be $O(n^2)$
    *   the window moves only one position, try to figure out how to reuse the infomation from the last movement.
    *   From the example, it seems that we need to know the descending order.
    *   A fact: a big item will overwhelm all smaller elements in the front of it. So we can crash them.
    *   A queue can meet the requirements. 
    *   Here is a problem, some great numbers should get removed if they are not in the window: remove the first item if it is in the queue before we move the window.
    *   **Note we cant smash the same val**
    *   *this is called monotonic stack*

*   /// Solution:

    *   ```python
        from collections import deque
        class Solution:
            def queue_add(self, queue: deque, val: int):
                # Note we cant smash the same val
                # it should be remove in line 19 and 20.
                while queue and queue[-1] < val:
                    queue.pop()
                queue.append(val)
                
            def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
                queue = deque()
                # init
                for i in range(k - 1):
                    self.queue_add(queue, nums[i])
                result = []
                for i in range(k - 1, len(nums)):
                    self.queue_add(queue, nums[i])
                    result.append(queue[0])
                    if queue and queue[0] == nums[i + 1 - k]:
                        queue.popleft()
                    
                return result
            

### Else

#### 16. 

#### 26. 

#### 27. 

#### 283. 

#### 1099. 

#### 11. 

#### 42. 

#### 986. 

#### 15.

#### 18. 

#### 259.



## Linked List - Double Pointers

### 2. Add Two Numbers

*   ![image-20220630230951963](coding_review_v1.0.assets/image-20220630230951963.png)

*   Thoughts:

    *   Traverse the two linked list at the same time. Since the digits are stored in reverse order, we can add each node and store the carry.
    *   And we create node for result in the same time.
    *   They may have different length. join the rest of it and dont forget the carry reminded.
    *   ![image-20220630232139692](coding_review_v1.0.assets/image-20220630232139692.png)
    *   Note that after traversing l1 and l2, carry could be 1.

*   Solution:

    *   ```python
        # Definition for singly-linked list.
        # class ListNode:
        #     def __init__(self, val=0, next=None):
        #         self.val = val
        #         self.next = next
        class Solution:
            def addTwoNumbers(self, l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
                carry = 0
                head = ListNode()
                dummy = head
                while l1 and l2:
                    val = l1.val + l2.val + carry
                    carry = 1 if val >= 10 else 0
                    head.next = ListNode(val % 10)
                    head = head.next
                    l1 = l1.next
                    l2 = l2.next
                while l1:
                    if carry != 0:
                        val = l1.val + carry
                        carry = 1 if val >= 10 else 0
                        head.next = ListNode(val % 10)
                        head = head.next
                        l1 = l1.next
                    elif carry == 0:
                        head.next = l1
                        break
                while l2:
                    if carry != 0:
                        val = l2.val + carry
                        carry = 1 if val >= 10 else 0
                        head.next = ListNode(val % 10)
                        head = head.next
                        l2 = l2.next
                    elif carry == 0:
                        head.next = l2
                        break
                if carry == 1:
                    while head.next:
                        head = head.next
                    head.next = ListNode(1)
                
                return dummy.next
                
        ```

    *   

### 19. Remove Nth Node From End of List

*   ![image-20220630232401337](coding_review_v1.0.assets/image-20220630232401337.png)
*   Thoughts:
*   Solution:

### 21. 

*   
*   Thoughts:
*   Solution:

### 23. 

*   
*   Thoughts:
*   Solution:

### 141. 

*   
*   Thoughts:
*   Solution:

### 142. 

*   
*   Thoughts:
*   Solution:

### 160. 

*   
*   Thoughts:
*   Solution:

### 876. 

*   
*   Thoughts:
*   Solution:

### 25. 

*   
*   Thoughts:
*   Solution:

### 83. 

*   
*   Thoughts:
*   Solution:

### 92. 

*   
*   Thoughts:
*   Solution:

### 234.

*   
*   Thoughts:
*   Solution:

## Prefix Sum

## Differential Array

## Queue / Stack

## Binary Heap

## Data Structure Design



# Tree and Graph

## Binary Tree

## Binary Search Tree

## Graph



# Search

## Backtrack

## DFS

## BFS



# Dynamic Programming

## 1D DP

## 2D DP

## Knapsack Problem



# Else

## Math

## Interval





