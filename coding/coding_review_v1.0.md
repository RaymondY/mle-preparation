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

    *   At least we need to traverse the linked list, but we cant go back I think, it is time consuming.
    *   We can build two pointers slow and fast, and let the fast n nodes away from slow.
    *   Since we need to link father node and child node, we better let the slow.next be the one should be deleted.

*   Solution:

    *   ```python
        # Definition for singly-linked list.
        # class ListNode:
        #     def __init__(self, val=0, next=None):
        #         self.val = val
        #         self.next = next
        class Solution:
            def removeNthFromEnd(self, head: Optional[ListNode], n: int) -> Optional[ListNode]:
                dummy = ListNode()
                dummy.next = head
                slow = fast = dummy
                for i in range(n):
                    fast = fast.next
                while fast.next:
                    slow = slow.next
                    fast = fast.next
                # we need to delete slow.next
                child = slow.next.next
                slow.next = child
                
                return dummy.next
        
        ```

    *   

### 21. Merge Two Sorted Lists

*   ![image-20220701120127125](coding_review_v1.0.assets/image-20220701120127125.png)

*   Thoughts: 

    *   cause they are both sorted. build a pointer for each of them and compare the pointers. move the smaller one.
    *   when one pointer reach the end, joint the other one to the end of result list.

*   Solution:

    *   ```python
        # Definition for singly-linked list.
        # class ListNode:
        #     def __init__(self, val=0, next=None):
        #         self.val = val
        #         self.next = next
        class Solution:
            def mergeTwoLists(self, list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:
                pointer1 = list1
                pointer2 = list2
                dummy = head = ListNode()
                while pointer1 and pointer2:
                    val1 = pointer1.val
                    val2 = pointer2.val
                    if val1 <= val2:
                        head.next = pointer1
                        pointer1 = pointer1.next
                    elif val2 < val1:
                        head.next = pointer2
                        pointer2 = pointer2.next
                    head = head.next
                if pointer1:
                    head.next = pointer1
                if pointer2:
                    head.next = pointer2
                    
                return dummy.next
            
        ```

    *   !!! Recursive one

    *   ```python
        class Solution:
            def mergeTwoLists(self, l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
                if l1 is None:
                    return l2
                if l2 is None:
                    return l1
                if l1.val < l2.val:
                    l1.next = self.mergeTwoLists(l1.next, l2)
                    return l1
                else:
                    l2.next = self.mergeTwoLists(l1, l2.next)
                    return l2
                
        ```

    *   

### 23. Merge k Sorted Lists

*   ![image-20220701140712667](coding_review_v1.0.assets/image-20220701140712667.png)

*   Thoughts:

    *   Similar to merge 2 sorted lists
    *   The point is how we can compare k pointers fast.
    *   !!! utilize the data structure priority queue / the min heap

*   Solution:

    *   Time $O(n\log k)$

    *   ```python
        # Definition for singly-linked list.
        # class ListNode:
        #     def __init__(self, val=0, next=None):
        #         self.val = val
        #         self.next = next
        from queue import PriorityQueue
        
        class Solution:
            def mergeKLists(self, lists: List[Optional[ListNode]]) -> Optional[ListNode]:
                # !!!
                setattr(ListNode, "__lt__", lambda self, other: self.val <= other.val)
                
                dummy = head = ListNode()
                pq = PriorityQueue()
                
                for lst in lists:
                    if lst != None:
                        pq.put(lst)
                
                while not pq.empty():
                    head.next = pq.get()
                    head = head.next
                    if head.next != None:
                        pq.put(head.next)
                
                return dummy.next
                
        ```

    *   I think we can also apply divide and conquer strategy. it is also $O(n \log k)$

### 141. Linked List Cycle

*   **Follow up:** Can you solve it using `O(1)` (i.e. constant) memory?

*   Solution:

    *   ```python
        class Solution:
            def hasCycle(self, head: Optional[ListNode]) -> bool:
                slow = fast = head
                while fast and fast.next :
                    slow = slow.next
                    fast = fast.next.next
                    if slow == fast:
                        return True
                return False
                
        ```

    *   

### 142. Linked List Cycle II

*   ![image-20220701150728459](coding_review_v1.0.assets/image-20220701150728459.png)

*   Thoughts:

    *   With $O(1)$ space complexity to find whether there is a cycle, we use fast and slow pointers.
    *   To find the position, we need to dig into step they move.
    *   In the most cases, they won't meet at the right position
    *   After first meeting, put one pointer at the start node, move k - m steps at the same pace until they meet again.

*   Solution:

    *   ```python
        class Solution:
            def detectCycle(self, head: Optional[ListNode]) -> Optional[ListNode]:
                slow = fast = head
                has_cycle = False
                while fast and fast.next:
                    slow = slow.next
                    fast = fast.next.next
                    if slow == fast:
                        has_cycle = True
                        break
                if not has_cycle:
                    return None
                slow = head
                while slow != fast:
                    slow = slow.next
                    fast = fast.next
                return slow
                
        ```

    *   

### 160. Intersection of Two Linked Lists

*   ![image-20220701160004579](coding_review_v1.0.assets/image-20220701160004579.png)

*   Thoughts:

    *   The difficulty is that they may not has the same size.
    *   Try to let two pointers meet after moving same steps.
    *   If we move m + n steps and still not meet, they won't meet.

*   Solution:

    *   Note the condition order is important

    *   ```python
        # Definition for singly-linked list.
        # class ListNode:
        #     def __init__(self, x):
        #         self.val = x
        #         self.next = None
        
        class Solution:
            def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> Optional[ListNode]:
                pointerA = headA
                pointerB = headB
                # -1 means non-visited
                sizeA = sizeB = -1
                step = 1
                while True:
                    if not pointerA and sizeA == -1:
                        pointerA = headB
                        sizeA = step - 1
                    if not pointerB and sizeB == -1:
                        pointerB = headA
                        sizeB = step - 1
                    if pointerA == pointerB and pointerA:
                        return pointerA
                    if (step - 1) == (sizeA + sizeB):
                        return None
                    pointerA = pointerA.next
                    pointerB = pointerB.next
                    step += 1
                    
        ```

    *   /// this one is much clear

    *   ```python
        # Definition for singly-linked list.
        # class ListNode:
        #     def __init__(self, x):
        #         self.val = x
        #         self.next = None
        
        class Solution:
            def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> Optional[ListNode]:
                pointerA = headA
                pointerB = headB
                while pointerA != pointerB:
                    if not pointerA:
                        pointerA = headB
                    else:
                        pointerA = pointerA.next
                    if not pointerB:
                        pointerB = headA
                    else:
                        pointerB = pointerB.next
                return pointerA
        
        ```

    *   

### 876. Middle of the Linked List

*   ![image-20220701162441944](coding_review_v1.0.assets/image-20220701162441944.png)

*   Thoughts:

    *   Since for even \# of nodes we need the second middle node, it decide the end condition.

*   Solution:

    *   ```python
        class Solution:
            def middleNode(self, head: Optional[ListNode]) -> Optional[ListNode]:
                slow = fast = head
                # not fast: even
                # not fast.next: ood
                while fast and fast.next:
                    slow = slow.next
                    fast = fast.next.next
                return slow
                
        ```

    *   

### 25. Reverse Nodes in k-Group

*   ![image-20220704151939800](coding_review_v1.0.assets/image-20220704151939800.png)

*   Thoughts:

    *   Find a way with $O(1)$ space complexity
    *   Think about the case with k > 2
    *   How can I know the \# of nodes is not a multiple of k?
        *   Record the start node, once traverse k nodes, reverse the list.
    *   <img src="coding_review_v1.0.assets/IMG_A8DEE08DB5E4-1.jpeg" alt="IMG_A8DEE08DB5E4-1" style="zoom: 33%;" />

*   Solution:

    *   ```python
        # Definition for singly-linked list.
        # class ListNode:
        #     def __init__(self, val=0, next=None):
        #         self.val = val
        #         self.next = next
        class Solution:
            def reverse_list(self, start: ListNode, end: ListNode, k: int):
                pre = start.next
                start.next = end
                
                node = pre.next
                pre.next = end.next
                start = pre
                
                for i in range(1, k):
                    temp = node.next
                    node.next = pre
                    pre = node
                    node = temp
                return start
                
            def reverseKGroup(self, head: Optional[ListNode], k: int) -> Optional[ListNode]:
                dummy = ListNode()
                dummy.next = head
                count = 0
                start = dummy
                end = None
                while head:
                    count += 1
                    if count == k:
                        end = head
                        head = start = self.reverse_list(start, end, k)
                        count = 0
                    head = head.next
                return dummy.next
                
        ```

*   Other: !!! Recursive

    *   ```python
        # Definition for singly-linked list.
        # class ListNode:
        #     def __init__(self, val=0, next=None):
        #         self.val = val
        #         self.next = next
        class Solution:
            def reverse_list(self, start: ListNode, k: int):
                pre = None
                cur = nxt = start
                for i in range(k):
                    nxt = cur.next
                    cur.next = pre
                    pre = cur
                    cur = nxt
                return pre
                
            def reverseKGroup(self, head: Optional[ListNode], k: int) -> Optional[ListNode]:
                if not head:
                    return None
                start = end = head
                for i in range(k):
                    if not end:
                        return start
                    end = end.next
                new_head = self.reverse_list(start, k)
                start.next = self.reverseKGroup(end, k)
                return new_head
                
        ```

    *   

### 83. Remove Duplicates from Sorted List

*   ![image-20220704163437104](coding_review_v1.0.assets/image-20220704163437104.png)

*   Thoughts:

    *   The link is sorted, so once we find a new val, delete all following nodes with same val.
    *   `-100 <= Node.val <= 100` init val_mark = -1000.
    *   and we can continue find the .next until we find the one different, then connect them.
    *   note the last node_mark

*   Solution:

    *   ```python
        # Definition for singly-linked list.
        # class ListNode:
        #     def __init__(self, val=0, next=None):
        #         self.val = val
        #         self.next = next
        class Solution:
            def deleteDuplicates(self, head: Optional[ListNode]) -> Optional[ListNode]:
                if not head:
                    return None
                dummy = head
                value = head.val
                node_mark = head
                while head:
                    if head.val != value:
                        node_mark.next = head
                        node_mark = head
                        value = head.val
                    head = head.next
                # note the end
                node_mark.next = None
                return dummy
                
        ```

    *   

### 92. Reverse Linked List II

*   ![image-20220704220341063](coding_review_v1.0.assets/image-20220704220341063.png)
*   Thoughts:

    *   First we find the node in the front of left

*   Solution:

    *   ```python
        # Definition for singly-linked list.
        # class ListNode:
        #     def __init__(self, val=0, next=None):
        #         self.val = val
        #         self.next = next
        class Solution:
            def reverseBetween(self, head: Optional[ListNode], left: int, right: int) -> Optional[ListNode]:
                dummy = ListNode()
                dummy.next = head
                pointer = dummy
                for i in range(left - 1):
                    pointer = pointer.next
                # now the pointer is the father of left
                before = pointer
                pointer = pointer.next
                new_end = pointer
                pre = None
                cur = nxt = pointer
                for i in range(left, right + 1):
                    nxt = cur.next
                    cur.next = pre
                    pre = cur
                    cur = nxt
                behind = cur
                new_start = pre
                # connect them
                before.next = new_start
                new_end.next = behind
                
                return dummy.next
                
        ```

    *   


### 234. Palindrome Linked List

*   <img src="coding_review_v1.0.assets/image-20220704222156840.png" alt="image-20220704222156840" style="zoom:25%;" />

*   Thoughts:

    *   To decide whether a whole linked list, we utilize stack, FILO
    *   How can we know when I or O. the point is we dont know the size, and the we can't decide it by stack.top()
    *   Stack is not good
    *   Use !!! fast-slow pointers to find the mid point, then reverse the right side list
    *   Note odd or even cases
    *   

*   Solution:

    *   ```python
        # Definition for singly-linked list.
        # class ListNode:
        #     def __init__(self, val=0, next=None):
        #         self.val = val
        #         self.next = next
        class Solution:
            def isPalindrome(self, head: Optional[ListNode]) -> bool:
                if not head.next:
                    return True
                # move the first step
                slow = head
                fast = head.next
                while fast and fast.next:
                    slow = slow.next
                    fast = fast.next.next
                # mark the mid pos
                mid = slow
                # reverse the list behind from mid
                pre = None
                cur = nxt = mid.next
                while cur:
                    nxt = cur.next
                    cur.next = pre
                    pre = cur
                    cur = nxt
                # pre is the new start
                first = head
                second = pre
                while second:
                    if first.val != second.val:
                        return False
                    first = first.next
                    second = second.next
                return True
                
        ```

    *   


## Prefix Sum

### 303. Range Sum Query - Immutable

*   Solution: 

    *   ```python
        class NumArray:
        
            def __init__(self, nums: List[int]):
                self.pre_sum = []
                self.pre_sum.append(nums[0])
                for i in range(1, len(nums)):
                    self.pre_sum.append(nums[i] + self.pre_sum[i-1])
                
        
            def sumRange(self, left: int, right: int) -> int:
                sum_val = self.pre_sum[right]
                if left > 0:
                    sum_val -= self.pre_sum[left - 1]
                return sum_val
        
        
        # Your NumArray object will be instantiated and called as such:
        # obj = NumArray(nums)
        # param_1 = obj.sumRange(left,right)
        
        ```

    *   

### /// 304. Range Sum Query 2D - Immutable

*   
*   Thoughts: 
    *   
    *   
*   Solution: 

### !!!327. Count of Range Sum

*   !!! 315 do this first

*   ![image-20220705010933407](coding_review_v1.0.assets/image-20220705010933407.png)
*   Thoughts: 
    *   There are $n(n-1)/2$ possible ranges.
    *   So if we compute each range, the time will be at least $O(n^2)$
    *   How can we utilize info better? Sort? no, we need to keep the order
*   Solution: 

### 1352. 

*   
*   Thoughts: 
*   Solution: 

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





