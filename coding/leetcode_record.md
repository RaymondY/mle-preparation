>   "@@@" - tech problem; "!!!" - cannot solve at the first time; "+++" - some useful trick; "///" - try later.

## 05.31

### 303. Range Sum Query - Immutable

>   Prefix sum **array**.

*   Time Complexity: $O(n)$

*   Space Complexity: $O(1)$

*   ```python
    class NumArray:
    
        def __init__(self, nums: List[int]):
            self.pre_sum = nums  # pass by pointer
        
            for i in range(len(nums)-1):
                self.pre_sum[i+1] += self.pre_sum[i]
    
        def sumRange(self, left: int, right: int) -> int:
            if left == 0:
                return self.pre_sum[right]
            
            return self.pre_sum[right] - self.pre_sum[left-1] 
    ```

### 304. Range Sum Query 2D - Immutable

>   Prefix sum **Matrix**.

*   Time Complexity: $O(m*n)$
*   Space Complexity: $O(1)$



###  1109. Corporate Flight Bookings

>   Diff Array.

*   Time Complexity: $O(n)$

*   Space Complexity: $O(n)$

*   ```python
    class Solution:
        def __init__(self):
            self.diff = []
            
        def book_flight(self, begin, end, num):
            self.diff[begin] += num
            if end + 1 < len(self.diff):
                self.diff[end + 1] -= num
            
        def corpFlightBookings(self, bookings: List[List[int]], n: int) -> List[int]:
            # construct diff array
            self.diff = [0 for i in range(n)]
            res = [0 for i in range(n)]
            
            # Note that the index starts from 1 in the description.
            for book in bookings:
                self.book_flight(book[0]-1, book[1]-1, book[2])
                
            res[0] = self.diff[0]
            for i in range(1, n):
                res[i] = res[i-1] + self.diff[i]
                
            return res
                
    ```



### 1094. Car Pooling

>   Diff Array.

*   Time Complexity: $O(n)$

*   Space Complexity: $O(m)$ (`0 <= from_i < to_i <= 1000`,  $m = 1001$ here)

*   ```python
    class Solution:
        def carPooling(self, trips: List[List[int]], capacity: int) -> bool:
            # 0 <= from_i < to_i <= 1000
            diff = [0 for i in range(1001)]
            
            for trip in trips:
                diff[trip[1]] += trip[0]
                diff[trip[2]] -= trip[0]
                
            count = 0
                
            for item in diff:
                count += item
                if count > capacity:
                    return False
            
            return True
            
    ```

### 21. Merge Two Sorted Lists

>   Two pointers
>
>   **Dummy** head is useful.

*   Time Complexity: $O(n)$
*   Space Complexity: $O(1)$



### @@@ 23. Merge k Sorted Lists

*   `queue.PriorityQueue` has the same $T(n), S(n)$ as `heapq` in python3 but it is synchronized.
*   Time Complexity: $O(k*n)$ (it can be $O(nlogk)$ if we "Divide and Conquer ")
*   Space Complexity: $O(1)$

*   ```python
    # Definition for singly-linked list.
    # class ListNode:
    #     def __init__(self, val=0, next=None):
    #         self.val = val
    #         self.next = next
    
    from queue import PriorityQueue
    
    class Solution:
        def mergeKLists(self, lists: List[Optional[ListNode]]) -> Optional[ListNode]:
            
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

### 19. Remove Nth Node From End of List

>   Find kth node from the end.

*   `dummy.next = head` avoids error when we deleting the first node.

*   Time Complexity: $O(sz)$ (The number of nodes in the list is `sz`)

*   Space Complexity: $O(1)$

*   ```python
    # Definition for singly-linked list.
    # class ListNode:
    #     def __init__(self, val=0, next=None):
    #         self.val = val
    #         self.next = next
    class Solution:
        def removeNthFromEnd(self, head: Optional[ListNode], n: int) -> Optional[ListNode]:
            # Delete slow.next
            dummy = ListNode()
            dummy.next = head
            fast = slow = dummy
            for i in range(n + 1):
                fast = fast.next
            
            while fast != None:
                fast = fast.next
                slow = slow.next
            
            slow.next = slow.next.next
            
            return dummy.next
    
    ```



### 876. Middle of the Linked List

>   Fast & Slow pointers

*   Time Complexity: $O(sz)$ (The number of nodes in the list is `sz`)
*   Space Complexity: $O(1)$



### 141. Linked List Cycle

>   Fast & Slow pointers

*   Time Complexity: $O(sz)$ (The number of nodes in the list is `sz`)
*   Space Complexity: $O(1)$



### 142. Linked List Cycle II

>   Fast & Slow pointers

*   Time Complexity: $O(sz)$ (The number of nodes in the list is `sz`)

*   Space Complexity: $O(1)$

*   ```python
    # Definition for singly-linked list.
    # class ListNode:
    #     def __init__(self, x):
    #         self.val = x
    #         self.next = None
    
    class Solution:
        def detectCycle(self, head: Optional[ListNode]) -> Optional[ListNode]:
            fast = slow = head
            while fast != None and fast.next != None:
                fast = fast.next.next
                slow = slow.next
                if fast == slow:
                    slow = head
                    while fast != slow:
                        fast = fast.next
                        slow = slow.next
                    return slow
            
            return None
            
    ```

*   The length of cycle is $step(fast)-step(slow)=step(slow)=k$. After another $k-m$ steps, the pointer arrives at the very node.



### 160. Intersection of Two Linked Lists

>   How to synchronize two pointer?

*   Time Complexity: $O(m+n)$ 

    *   The number of nodes of `listA` is in the `m`.
    *   The number of nodes of `listB` is in the `n`.

*   Space Complexity: $O(1)$

*   ```python
    # Definition for singly-linked list.
    # class ListNode:
    #     def __init__(self, x):
    #         self.val = x
    #         self.next = None
    
    class Solution:
        def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> Optional[ListNode]:
            pointA = headA
            pointB = headB
    
            while pointA != pointB:
                if pointA != None:
                    pointA = pointA.next
                else:
                    pointA = headB
                if pointB != None:
                    pointB = pointB.next
                else:
                    pointB = headA
            
            return pointA
    
    ```



### 26. Remove Duplicates from Sorted Array

>   Two pointers in array

*   Time Complexity: $O(n)$ 

*   Space Complexity: $O(1)$

*   ```python
    class Solution:
        def removeDuplicates(self, nums: List[int]) -> int:
            slow = 0
            for num in nums:
                if nums[slow] != num:
                    slow += 1
                    nums[slow] = num
            return slow + 1
            
    ```



### 83. Remove Duplicates from Sorted List

>   LinkedList version of the last question.

*   Time Complexity: $O(n)$ 

*   Space Complexity: $O(1)$

*   ```python
    # Definition for singly-linked list.
    # class ListNode:
    #     def __init__(self, val=0, next=None):
    #         self.val = val
    #         self.next = next
    class Solution:
        def deleteDuplicates(self, head: Optional[ListNode]) -> Optional[ListNode]:
            if head == None:
                return None
            slow = fast = head
            while fast != None:
                if fast.val != slow.val:
                    slow.next = fast
                    slow = slow.next
                fast = fast.next
            slow.next = None
            return head
                
    ```



## 06.01

### 27. Remove Element

>   Similar

*   Time Complexity: $O(n)$ 
*   Space Complexity: $O(1)$

### 283. Move Zeroes

>   #27

*   Time Complexity: $O(n)$ 

*   Space Complexity: $O(1)$

*   ```python
    class Solution:
        def moveZeroes(self, nums: List[int]) -> None:
            """
            Do not return anything, modify nums in-place instead.
            """
            slow = 0
            for fast in range(len(nums)):
                if nums[fast] != 0:
                    nums[slow] = nums[fast]
                    slow += 1
            
            while slow < len(nums):
                nums[slow] = 0
                slow += 1       
            
    ```

### 1. Two Sum

*   Traversal:
    *   Time Complexity: $O(n^2)$ 
    *   Space Complexity: $O(1)$
    *   Runtime: 4657 ms
*   Hash Set:
    *   Time Complexity: $O(n)$ 
    *   Space Complexity: $O(n)$
    *   Runtime: 66 ms

### 167. Two Sum II - Input Array Is **Sorted**

>   Sorted -> Left & Right Pointers

*   Time Complexity: $O(n)$ 
*   Space Complexity: $O(1)$

### 344. Reverse String

*   Time Complexity: $O(n)$ 
*   Space Complexity: $O(1)$

### !!! 5. Longest Palindromic Substring

*   Time Complexity: $O(n^2)$ 
*   Space Complexity: $O(1)$

*   ```python
    class Solution:
        def get_palindormic(self, s: str, left: int, right: int):
            while left >= 0 and right < len(s) and s[left] == s[right]:
                left -= 1
                right += 1
            return left + 1, right - 1
        
        def longestPalindrome(self, s: str) -> str:
            left = len(s)
            right = 0
            for i in range(len(s)):
                odd_left, odd_right = self.get_palindormic(s, i, i)
                even_left, even_right = self.get_palindormic(s, i, i+1)
                if (odd_right - odd_left) > (right - left):
                    left = odd_left
                    right = odd_right
                if (even_right - even_left) > (right - left):
                    left = even_left
                    right = even_right
                    
            return s[left:right+1]
                
    ```



### !!! 76. Minimum Window Substring

>   Sliding Window

*   Time Complexity: $O(n)$ 
*   Space Complexity: $O(1)$

*   ```python
    class Solution:
        def minWindow(self, s: str, t: str) -> str:
            target = {}
            window = {}
            for letter in t:
                target[letter] = 1 if letter not in target else target[letter] + 1
            
            record_left = 0
            record_size = len(s) + 1
            
            # when not_valid == 0 -> valid
            not_valid = len(target)
            
            left = right = 0
            
            # [left, right)
            while right < len(s):
                # move right
                right_letter = s[right]
                # right is ready for the next check
                right += 1
                if right_letter in target:
                    window[right_letter] = 1 if right_letter not in window else window[right_letter] + 1
                    if window[right_letter] == target[right_letter]:
                        not_valid -= 1
                # move left
                while not_valid == 0:
                    left_letter = s[left]
                    if left_letter in window:
                        if window[left_letter] == target[left_letter]:
                            if right - left < record_size:
                                record_left = left
                                record_size = right - left
                            not_valid += 1
                        window[left_letter] -= 1
                    # left is ready for the next check
                    left += 1
            
            return "" if record_size > len(s) else s[record_left: record_left+record_size]
    
    ```



### 567. Permutation in String

*   Time Complexity: $O(n)$
*   Space Complexity: $O(1)$

*   ```python
    class Solution:
        def checkInclusion(self, s1: str, s2: str) -> bool:
            target = {}
            window = {}
            
            for letter in s1:
                target[letter] = 1 if letter not in target else target[letter] + 1
            
            # [left, right)
            left = right = 0
            valid = 0
            
            while right < len(s2):
                r_letter = s2[right]
                right += 1
                if r_letter in target:
                    window[r_letter] = 1 if r_letter not in window else window[r_letter] + 1
                    if window[r_letter] == target[r_letter]:
                        valid += 1
                
                while (right - left) >= len(s1):
                    if valid == len(target):
                        return True
                    l_letter = s2[left]
                    left += 1
                    if l_letter in target:
                        if window[l_letter] == target[l_letter]:
                            valid -= 1
                        window[l_letter] -= 1
    
            return False
    
    ```

### 438. Find All Anagrams in a String

>   Same as the last problem

*   Time Complexity: $O(n)$
*   Space Complexity: $O(1)$

*   ```python
    class Solution:
        def findAnagrams(self, s: str, p: str) -> List[int]:
            target = {}
            window = {}
            
            for letter in p:
                target[letter] = 1 if letter not in target else target[letter] + 1
            
            # [left, right)
            left = right = 0
            valid = 0
            result = []
            
            while right < len(s):
                r_letter = s[right]
                right += 1
                if r_letter in target:
                    window[r_letter] = 1 if r_letter not in window else window[r_letter] + 1
                    if window[r_letter] == target[r_letter]:
                        valid += 1
                
                
                if right - left== len(p):
                    if valid == len(target):
                        result.append(left)
                    l_letter = s[left]
                    left += 1
                    if l_letter in window:
                        if window[l_letter] == target[l_letter]:
                            valid -= 1
                        window[l_letter] -= 1
                    
            return result
                    
    ```



## 06.02

### 3. Longest Substring Without Repeating Characters

*   Time Complexity: $O(n)$

*   Space Complexity: $O(1)$

*   ```python
    class Solution:
        def lengthOfLongestSubstring(self, s: str) -> int:
            window = {}
            
            # [left right)
            left = right = 0
            length = 0
            
            while right < len(s):
                r_letter = s[right]
                window[r_letter] = 1 if r_letter not in window else window[r_letter] + 1
                while r_letter in window and window[r_letter] > 1:
                    if (right - left) > length:
                        length = right - left
                    l_letter = s[left]
                    left += 1
                    window[l_letter] -= 1
                    
                right += 1
                
            if (right - left) > length:
                    length = right - left
                    
            return length      
    ```

*   \* The left pointer can jump faster. (Still $O(n)$)



### 704. Binary Search

>   For sorted array.

*   Time Complexity: $O(logn)$

*   Space Complexity: $O(1)$

*   ```python
    class Solution:
        def search(self, nums: List[int], target: int) -> int:
            # SORTED
            # To find one targer, we apply closed interval
            left = 0
            right = len(nums) - 1
            while left <= right:
                mid = left + ((right - left) >> 1)
                if nums[mid] == target:
                    return mid
                elif nums[mid] < target:
                    left = mid + 1
                elif nums[mid] > target:
                    right = mid - 1
            return -1
            
    ```

*   Details for BS:
    *   `while` determines when it ends.
    *   `left = mid + 1 or mid` (`right = mid - 1 or mid`) depend on the search interval.
    *   Closed Interval or Open Interval?
    *   `mid = left + ((right - left) >> 1)`: prevent int overflow.

### !!!34. Find First and Last Position of Element in Sorted Array

>   Remember to scale the interval when we find the target.
>
>   Search for left & right boundaries.

*   Time Complexity: $O(logn)$

*   Space Complexity: $O(1)$

*   ```python
    class Solution:
        def searchRange(self, nums: List[int], target: int) -> List[int]:
            # closed interval
            
            # search for left
            left = 0
            right = len(nums) - 1
            
            while left <= right:
                mid = left + ((right - left) >> 1)
                if nums[mid] == target:
                    # scale the interval
                    right = mid - 1
                elif nums[mid] < target:
                    left = mid + 1
                elif nums[mid] > target:
                    right = mid - 1
            
            if left >= len(nums) or nums[left] != target:
                return [-1, -1]
            
            # the right index crosses the boundary here
            target_left = left
            
            # search for right
            left = 0
            right = len(nums) - 1
            
            while left <= right:
                mid = left + ((right - left) >> 1)
                if nums[mid] == target:
                    # scale the interval
                    left = mid + 1
                elif nums[mid] < target:
                    left = mid + 1
                elif nums[mid] > target:
                    right = mid - 1
            
            # the left index crosses the boundary here
            target_right = right
            
            return [target_left, target_right]
    
    ```



### 875. Koko Eating Bananas

>   Same as the regular one except for the scaling conditions

*   Time Complexity: $O(nlogm)$ (*There are `n` piles of bananas*, `m` is the maximum value of piles.)

*   Space Complexity: $O(1)$

*   ```python
    class Solution:
        def convert_target(self, piles, k):
            expected_h = 0
            for pile in piles:
                expected_h += pile // k if pile % k == 0 else pile // k + 1
            return expected_h
        
        def minEatingSpeed(self, piles: List[int], h: int) -> int:
            # find the left(min) boundary
            # [left, right]
            left = sum(piles) // h if sum(piles) % h == 0 else sum(piles) // h + 1
            right = max(piles)
            
            while left <= right:
                mid = left + ((right - left) >> 2)
                expected_h = self.convert_target(piles, mid)
                if expected_h == h:
                    right = mid - 1
                # Note the condition here is opposite to the regular one.
                elif expected_h > h:
                    left = mid + 1
                elif expected_h < h:
                    right = mid - 1
            
            return left
            
    ```



## 06.03

### 206. Reverse Linked List

*   Recursion

    *   Time Complexity: $O(n)$

    *   Space Complexity: $O(n)$

    *   ```python
        class Solution:
            def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
                # recursion
                if head == None or head.next == None:
                    return head
                
                # the head of the reversed list
                last = self.reverseList(head.next)
                head.next.next = head
                head.next = None
                
                return last
            
        ```

*   Iteration

    *   Time Complexity: $O(n)$

    *   Space Complexity: $O(1)$

    *   <img src="leetcode_record.assets/image-20220603123344908.png" alt="image-20220603123344908" style="zoom: 25%;" />

    *   ```python
        class Solution:
            def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
                
                target = None
                cur = head
                
                while cur != None:
                    saved_node = cur.next
                    cur.next = target
                    target = cur
                    cur = saved_node
                    
                return target
        ```

### 92. Reverse Linked List II

>   Same as the last problem except for recording the left, right related nodes and connecting them later.

*   Iteration

    *   Time Complexity: $O(n)$

    *   Space Complexity: $O(1)$

    *   <img src="leetcode_record.assets/image-20220603130232556.png" alt="image-20220603130232556" style="zoom:25%;" />

    *   

    *   ```python
        class Solution:
            def reverseBetween(self, head: Optional[ListNode], left: int, right: int) -> Optional[ListNode]:
                dummy = ListNode()
                dummy.next = head
                head = dummy
                count = 0
                
                while count != left - 1:
                    head = head.next
                    count += 1
                    
                left_node = head
                r_left_node = head.next
                head = head.next
                count += 1
                
                target = None
                while count != right + 1:
                    saved_node = head.next
                    head.next = target
                    target = head
                    head = saved_node
                    count += 1
                    
                right_node = head
                r_right_node = target
                
                left_node.next = r_right_node
                r_left_node.next = right_node
                
                return dummy.next
        ```

    *   

*   !!!Recursion

    *   Time Complexity: $O(n)$

    *   Space Complexity: $O(n)$

    *   ```python
        class Solution:
            def __init__(self):
                self.saved_node = None
            
            def reverse_pre_n(self, head: Optional[ListNode], n: int):
                if n == 1:
                    self.saved_node = head.next
                    return head
        
                # the head of the reversed list
                last = self.reverse_pre_n(head.next, n-1)
                
                head.next.next = head
                head.next = self.saved_node
                return last
                
            def reverseBetween(self, head: Optional[ListNode], left: int, right: int) -> Optional[ListNode]:
                if left == 1:
                    return self.reverse_pre_n(head, right - left + 1)
                head.next = self.reverseBetween(head.next, left - 1, right - 1)
                return head
                
        ```

### 20. Valid Parentheses

>   Stack

*   Time Complexity:$O(n)$
*   Space Complexity: $O(n)$

*   ```python
    class Solution:
        def isValid(self, s: str) -> bool:
            stack = []
            match = {'(': ')', '{': '}', '[': ']'}
            
            for letter in s:
                if letter in match:
                    stack.append(letter)
                elif len(stack) == 0 or match[stack.pop()] != letter:
                    return False
            
            return len(stack) == 0
           
    ```

### 921. Minimum Add to Make Parentheses Valid

>   Stack

*   Time Complexity:$O(n)$

*   Space Complexity: $O(1)$

*   Actually, "left" is a hidden stack.

*   ```python
    class Solution:
        def minAddToMakeValid(self, s: str) -> int:
            left = right = 0
            stack = []
            for letter in s:
                if letter == '(':
                    left += 1
                elif letter == ')' and left == 0:
                    right += 1
                elif letter == ')' and left != 0:
                    left -= 1
            return left + right
    
    ```

*   !!! Left & right strategy is not universal. Request & need is better.

*   Based on '(', *request* records the '(' we already inserted, and *need* records the num of ')' needed.

*   ```python
    class Solution:
        def minAddToMakeValid(self, s: str) -> int:
            request = need = 0
            for letter in s:
                if letter == '(':
                    need += 1
                elif letter == ')':
                    need -= 1
                    if need == -1:
                        request += 1
                        need = 0
            
            return request + need
        
    ```

*   

### !!!1541. Minimum Insertions to Balance a Parentheses String

*   Time Complexity:$O(n)$

*   Space Complexity: $O(1)$

*   ```python
    class Solution:
        def minInsertions(self, s: str) -> int:
            request = need = 0
            for letter in s:
                if letter == '(':
                    need += 2
                    if need % 2 == 1:
                        request += 1
                        need -= 1
                        
                if letter == ')':
                    need -= 1
                    if need == -1:
                        request += 1
                        need = 1
                        
            return request + need
            
    ```

### 496. Next Greater Element I

>   Monotonic Stack

*   Time Complexity: $O(n)$

*   Space Complexity: $O(n)$

*   ```python
    class Solution:
        def nextGreaterElement(self, nums1: List[int], nums2: List[int]) -> List[int]:
            hash_result = {}
            # Monotonic stack
            stack = []
            for i in range(1, len(nums2) + 1):
                num = nums2[-i]
                
                # remove blocked number
                while len(stack) != 0 and stack[-1] <= num:
                    stack.pop()
                    
                hash_result[num] = -1 if len(stack) == 0 else stack[-1]
                stack.append(num)
    
            return [hash_result[i] for i in nums1]
            
    ```

### ///739. Daily Temperatures



## 06.05

### @@@239. Sliding Window Maximum

>   Monotonic Queue

>   `Queue.Queue` and `collections.deque` serve different purposes. Queue.Queue is intended for allowing different threads to communicate using queued messages/data, whereas `collections.deque` is simply intended as a datastructure. That's why `Queue.Queue` has methods like `put_nowait()`, `get_nowait()`, and `join()`, whereas `collections.deque` doesn't. `Queue.Queue` isn't intended to be used as a collection, which is why it lacks the likes of the `in` operator.
>
>   It boils down to this: if you have multiple threads and you want them to be able to communicate without the need for locks, you're looking for `Queue.Queue`; if you just want a queue or a double-ended queue as a datastructure, use `collections.deque`.
>
>   Finally, accessing and manipulating the internal deque of a `Queue.Queue` is playing with fire - you really don't want to be doing that. [python - Queue.Queue vs. collections.deque - Stack Overflow](https://stackoverflow.com/questions/717148/queue-queue-vs-collections-deque)

*   Time Complexity: $O(n)$
    *   Although the queue operation in the loop is not $O(1)$, each element `pop()` and `append()`  once in the worst situation.
*   Space Complexity: $O(k)$

*   ```python
    from collections import deque
    
    class Solution:
        def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
            mono_queue = deque()
            result = []
            
            for i in range(k - 1):
                cur = nums[i]
                while mono_queue and mono_queue[-1] < cur:
                    mono_queue.pop()
                
                mono_queue.append(cur)
            
            # when adding a new item, delete all smaller items in queue.
            for i in range(k - 1, len(nums)):
                cur = nums[i]
                    
                while mono_queue and mono_queue[-1] < cur:
                    mono_queue.pop()    
                mono_queue.append(cur)
                
                result.append(mono_queue[0])
                
                if mono_queue and mono_queue[0] == nums[i-k+1]:
                    mono_queue.popleft()
            
            return result
            
    ```

### ///316. Remove Duplicate Letters

>   Kind of greedy problem.



## 06.06

### 226. Invert Binary Tree

*   DFS & Recursion: Any order is okay, but I prefer postorder. It's like after you flipped left and right sub-trees, flip the left and right tree of root.

    *   Time: $O(n)$

    *   Space: $O(\log n)$ (I think)

    *   ```python
        class Solution:
            def invertTree(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
                if root == None:
                    return None
                
                left = self.invertTree(root.left)
                right = self.invertTree(root.right)
                
                root.left = right
                root.right = left
                
                return root
                
        ```

*   This can be solved by BFS and DFS & Iteration



### 116. Populating Next Right Pointers in Each Node

*   BFS with $O(1)$ Space.

*   Time: $O(n)$ (We visit each node once.)

*   ```python
    class Solution:
        def connect(self, root: 'Optional[Node]') -> 'Optional[Node]':
            # We need the info from father nodes.
            if not root:
                return root
            record = root
            while record.left != None:
                node = record
                while node.next != None:
                    node.left.next = node.right
                    node.right.next = node.next.left
                    node = node.next
                node.left.next = node.right
                
                record = record.left
            
            return root
        
    ```

*   !!! Recursion (I didn't come up with this at first, but I think there are overlaps in it. Also, resursion need extra space for stack.)

    *   Time: Overlap occurs from level 4 to the end.

    *   ```python
        class Solution:
            def traverse(self, node1, node2):
                if node1 == None:
                    return
                node1.next = node2
                self.traverse(node1.left, node1.right)
                self.traverse(node1.right, node2.left)
                self.traverse(node2.left, node2.right)
                
            def connect(self, root: 'Optional[Node]') -> 'Optional[Node]':
                if not root:
                    return root
                
                self.traverse(root.left, root.right)
                
                return root
                
        ```

### 114. Flatten Binary Tree to Linked List

*   Recursive:

    *   Time: $O(n)$

    *   Space: $O(\log n)$ (stack)

    *   ```python
        class Solution:
            def flatten(self, root: Optional[TreeNode]) -> None:
                """
                Do not return anything, modify root in-place instead.
                """
                # Preorder
                if not root:
                    return
                
                self.flatten(root.left)
                self.flatten(root.right)
                
                left = root.left
                right = root.right
                
                root.left = None
                root.right = left
                
                while root.right != None:
                    root = root.right
                
                root.right = right
        
        ```

*   !!!**Follow up:** Can you flatten the tree in-place (with `O(1)` extra space)?

    *   set the right node behind the left's most right

    *   ```python
        class Solution:
            def flatten(self, root: Optional[TreeNode]) -> None:
                """
                Do not return anything, modify root in-place instead.
                """
                while root:
                    if root.left:
                        node = root.left
                        
                        while node.right:
                            node = node.right
                        node.right = root.right
        
                        root.right = root.left
                        root.left = None
                    
                    root = root.right
                
        ```

### 654. Maximum Binary Tree

>   Create tree: sub-problem

*   Recursive

    *   Time: $O(n^2)$ (`list.index()` -- $O(n)$)

    *   Space: $O(n^2)$ (`list[start:end]` -- $O(n)$)

    *   ```python
        class Solution:
            def constructMaximumBinaryTree(self, nums: List[int]) -> Optional[TreeNode]:
                if not nums:
                    return None
                
                node_value = max(nums)
                node_index = nums.index(node_value)
                
                left = self.constructMaximumBinaryTree(nums[: node_index])
                right = self.constructMaximumBinaryTree(nums[node_index + 1 :])
                
                return TreeNode(node_value, left, right)
                
        ```

    *   

### 105. Construct Binary Tree from Preorder and Inorder Traversal

*   Recursive

*   Time: $O(n^2)$ (`list.index()` -- $O(n)$)
*   Space:  It better to write a helper function with left & right indexes to visit array. The slice array leads to extra space.

*   ```python
    class Solution:
        def buildTree(self, preorder: List[int], inorder: List[int]) -> Optional[TreeNode]:
            if not preorder:
                return None
            
            node_value = preorder[0]
            node_index = inorder.index(node_value)
            
            left_inorder = inorder[:node_index]
            right_inorder = inorder[node_index + 1:]
            
            left_preorder = preorder[1:node_index+1]
            right_preorder = preorder[node_index+1:]
            
            left = self.buildTree(left_preorder, left_inorder)
            right = self.buildTree(right_preorder, right_inorder)
            
            return TreeNode(node_value, left, right)
            
    ```

### 106. Construct Binary Tree from Inorder and Postorder Traversal

*   This is much better than what we did in the last problem. However, `list.index()` is an $O(n)$ opt.

*   ```python
    class Solution:
        def build(self, inorder, inorder_left, inorder_right, postorder, postorder_left, postorder_right):
            if inorder_left > inorder_right:
                return None
            
            node_value = postorder[postorder_right]
            node_index = inorder.index(node_value)
            
            left = self.build(inorder, inorder_left, node_index - 1, postorder, postorder_left, postorder_left + node_index - inorder_left - 1)
            right = self.build(inorder, node_index + 1, inorder_right, postorder, postorder_right - inorder_right + node_index, postorder_right - 1)
            
            return TreeNode(node_value, left, right)
            
        def buildTree(self, inorder: List[int], postorder: List[int]) -> Optional[TreeNode]:
            num = len(inorder)
            return self.build(inorder, 0, num-1, postorder, 0, num-1)
        
    ```

*   !!! Here is a better one:

*   >   "`inorder` and `postorder` consist of **unique** values." This leads us to apply a hashmap.

*   !!! We must return right before left cuz it is postorder array.

*   Time: $O(n)$ (`list.pop()` -- $O(1)$)

*   Space: $O(n)$ (Hash map: we use $O(n)$ Extra space to avoid $O(n^2)$ time )

*   ```python
    class Solution:
        def __init__(self):
            self.hashmap = {}
            self.inorder = []
            self.postorder = []
            
        def build(self, start, end):
            if start > end:
                return None
            node_value = self.postorder.pop()
            node_index = self.hashmap[node_value]
            
            # right before left!
            right = self.build(node_index + 1, end)
            left = self.build(start, node_index - 1)
            
            return TreeNode(node_value, left, right)
            
        def buildTree(self, inorder: List[int], postorder: List[int]) -> Optional[TreeNode]:
            self.inorder, self.postorder = inorder, postorder
            self.hashmap = {}  
            for index, value in enumerate(inorder):
                self.hashmap[value] = index
                
            return self.build(0, len(self.inorder) - 1)
            
    ```

## 06.07

### 889. Construct Binary Tree from Preorder and Postorder Traversal

*   ```python
    class Solution:
        def __init__(self):
            self.post_hash = {}
            self.preorder = []
            self.postorder = []
        
        def build(self, pre_start, pre_end, post_start, post_end):
            if pre_start > pre_end:
                return None
            
            node_value = self.preorder[pre_start]
            
            if pre_start == pre_end:
                return TreeNode(node_value)
            
            left_value = self.preorder[pre_start + 1]
            left_size = self.post_hash[left_value] - post_start + 1
            right_size = post_end - post_start - left_size
            
            left = self.build(pre_start + 1, pre_start + left_size, post_start, post_start + left_size - 1)
            right = self.build(pre_end - right_size + 1, pre_end, post_end - right_size, post_end - 1)
            
            return TreeNode(node_value, left, right)
            
        def constructFromPrePost(self, preorder: List[int], postorder: List[int]) -> Optional[TreeNode]:
            self.preorder = preorder
            self.postorder = postorder
            
            for i, val in enumerate(postorder):
                self.post_hash[val] = i
            
            num = len(self.preorder)
            return self.build(0, num - 1, 0, num - 1)
            
    ```



### !!! 297. Serialize and Deserialize Binary Tree

*   Iterative

    *   Time: $O(n)$

    *   ```python
        class Codec:
            def serialize(self, root):
                """Encodes a tree to a single string.
                
                :type root: TreeNode
                :rtype: str
                """
                json = []
                stack = []
                stack.append(root)
                while stack:
                    node = stack.pop()
                    if node:
                        json.append(str(node.val))
                        stack.append(node.right)
                        stack.append(node.left)
                    elif not node:
                        json.append('#')
                
                return ','.join(json)
                
            def deserialize(self, data):
                """Decodes your encoded data to tree.
                
                :type data: str
                :rtype: TreeNode
                """
                if data == "#":
                    return None
                json = data.split(',')
                stack = []
                dummy = root = TreeNode(int(json[0]))
                stack.append(root)
                i = 1
                direct = 'left'
                while i < len(json):
                    node_val = json[i]
                    i += 1
                    if node_val != '#':
                        node = TreeNode(int(node_val))
                        if direct == 'left':
                            root.left = node
                        elif direct == 'right':
                            root.right = node
                            direct = 'left'
                        stack.append(node)
                        root = node
                    elif node_val == '#':
                        if stack:
                            root = stack.pop()
                            direct = 'right'
                            
                return dummy
        ```

    *   

*   Recursive

    *   Time: $O(n)$

    *   ```python
        from collections import deque
        
        class Codec:        
            def serialize_helper(self, root, json):
                if not root:
                    json.append('#')
                    return
                json.append(str(root.val))
                self.serialize_helper(root.left, json)
                self.serialize_helper(root.right, json)
                
            def serialize(self, root):
                """Encodes a tree to a single string.
                
                :type root: TreeNode
                :rtype: str
                """
                json = []
                self.serialize_helper(root, json)
                return ','.join(json)
                
            def deserialize_helper(self, json):
                root_val = json.popleft()
                if root_val == '#':
                    return None
                root = TreeNode(int(root_val))
                root.left = self.deserialize_helper(json)
                root.right = self.deserialize_helper(json)
                
                return root
        
            def deserialize(self, data):
                """Decodes your encoded data to tree.
                
                :type data: str
                :rtype: TreeNode
                """
                json = deque(data.split(','))
                return self.deserialize_helper(json)
                
        ```

    *   >   deque.popleft() is faster than list.pop(0), because the deque has been optimized to do popleft() approximately in O(1), while list.pop(0) takes O(n) .



### +++ 652. Find Duplicate Subtrees

>   Serialize sub tree in **postorder**!

*   **Time: $O(n^2)$** (String concatenation in python takes $O(n)$? )

*   ```python
    lass Solution:
        def __init__(self):
            self.result = []
            self.hashmap = {}
            
        def find_helper(self, root) -> string:
            if not root:
                return "#"
            
            left = self.find_helper(root.left)
            right = self.find_helper(root.right)
            
            sub_tree = left + ',' + right + ',' + str(root.val)
            
            self.hashmap[sub_tree] = 1 if sub_tree not in self.hashmap else self.hashmap[sub_tree] + 1
            
            if self.hashmap[sub_tree] == 2:
                self.result.append(root)
                
            return sub_tree
                
        def findDuplicateSubtrees(self, root: Optional[TreeNode]) -> List[Optional[TreeNode]]:
            self.find_helper(root)
            return self.result
            
    ```



### (Daily) 88. Merge Sorted Array

*   Time: $O(n)$ | Space: $O(1)$

*   ```python
    class Solution:
        def merge(self, nums1: List[int], m: int, nums2: List[int], n: int) -> None:
            """
            Do not return anything, modify nums1 in-place instead.
            """
            pointer1 = m - 1
            pointer2 = n - 1
            
            while pointer1 >= 0 and pointer2 >= 0:
                if nums1[pointer1] >= nums2[pointer2]:
                    nums1[pointer1 + pointer2 + 1] = nums1[pointer1]
                    pointer1 -= 1
                elif nums1[pointer1] < nums2[pointer2]:
                    nums1[pointer1 + pointer2 + 1] = nums2[pointer2]
                    pointer2 -= 1
                    
            while pointer2 >= 0:
                nums1[pointer2] = nums2[pointer2]
                pointer2 -= 1
                
    ```

### 912. Sort an Array

>   Mergesort here: Postorder traverse

*   Time: $O(n\log n)$

*   ```python
    class Solution:
        def __init__(self):
            self.temp = []
            
        def merge(self, nums, left, mid, right):
            self.temp[left:right + 1] = nums[left:right + 1]
            pointer1, pointer2 = left, mid + 1
            index = left
            while pointer1 <= mid and pointer2 <= right:
                if self.temp[pointer1] <= self.temp[pointer2]:
                    nums[index] = self.temp[pointer1]
                    pointer1 += 1
                elif self.temp[pointer1] > self.temp[pointer2]:
                    nums[index] = self.temp[pointer2]
                    pointer2 += 1
                index += 1
                
            while pointer1 <= mid:
                nums[index] = self.temp[pointer1]
                pointer1 += 1
                index += 1
                
            while pointer2 <= right:
                nums[index] = self.temp[pointer2]
                pointer2 += 1
                index += 1
                
            
        def merge_sort(self, nums, left, right):
            if left == right:
                return
            mid = left + ((right - left) >> 1)
            
            self.merge_sort(nums, left, mid)
            self.merge_sort(nums, mid + 1, right)
            self.merge(nums, left, mid, right)
    
        def sortArray(self, nums: List[int]) -> List[int]:
            # Merge sort
            self.temp = [0] * len(nums)
            self.merge_sort(nums, 0, len(nums) - 1)
            return nums
    
    ```

*   



## 06.08

### (Daily) 1332. Remove Palindromic Subsequences

>   TRASH

### /// !!! (too hard) 315. Count of Smaller Numbers After Self

>   Mergesort

*   

### /// !!! (too hard) 327. Count of Range Sum



### 230. Kth Smallest Element in a BST

*   Traverse: inorder traverse of BST is sorted.

    *   Time: $O(n)$

    *   Space: $O(n)$ (include recursive stack)

    *   ```python
        class Solution:
            def __init__(self):
                self.count = 0
                self.val = 0
            
            def traverse(self, root, k):
                if not root:
                    return
                self.traverse(root.left, k)
                self.count += 1
                if self.count == k:
                    self.val = root.val
                    return
                self.traverse(root.right, k)
                
            def kthSmallest(self, root: Optional[TreeNode], k: int) -> int:
                self.traverse(root, k)
                return self.val
                
        ```

*   Follow up

*   >   **Follow up:** If the BST is modified often (i.e., we can do insert and delete operations) and you need to find the kth smallest frequently, how would you optimize?

    *   If we know the size of trees for each root, we can achieve $O(\log n)$ time complexity.



### 538. Convert BST to Greater Tree

>   the prop of BST' inorder traverse

*   Time: $O(n)$

*   worst Space: $O(n)$ (include recursive stack)

*   ```python
    class Solution:
        def __init__(self):
            self.summary = 0
            
        def traverse(self, root):
            if not root:
                return
            self.traverse(root.right)
            
            cur_val = root.val
            root.val += self.summary
            self.summary += cur_val
            
            self.traverse(root.left)
            
        def convertBST(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
            self.traverse(root)
            return root
        
    ```

### 98. Validate Binary Search Tree

*   Time: $O(n)$

*   worst Space: $O(n)$ (include recursive stack)

*   !!! This solution is not universal for traversing a BST.

*   ```python
    class Solution:
        def helper(self, root, min_root, max_root):
            if not root:
                return True
            if min_root and min_root.val >= root.val or max_root and max_root.val <= root.val:
                return False
            
            return self.helper(root.left, min_root, root) and self.helper(root.right, root, max_root)
            
        def isValidBST(self, root: Optional[TreeNode]) -> bool:
            return self.helper(root, None, None)
            
    ```

*   Here is universal traverse:

    *   Iterative:

    *   ```python
        class Solution:
            def isValidBST(self, root: Optional[TreeNode]) -> bool:
                if not root:
                    return True
                
                pre_root = None
                stack = []
                while root or stack:
                    while root:
                        stack.append(root)
                        root = root.left
                    root = stack.pop()
                    if pre_root and pre_root.val >= root.val:
                        return False
                    pre_root = root
                    root = root.right
                    
                return True
        ```

    *   Recursive

    *   ```python
        class Solution:
            def __init__(self):
                self.pre_root = None
                self.is_valid = True
                
            def traverse(self, root):
                if not root:
                    return
                
                self.traverse(root.left)
                
                # early stop
                if not self.is_valid:
                    return
                
                if self.pre_root and self.pre_root.val >= root.val:
                    self.is_valid = False
                    return
                self.pre_root = root
                
                self.traverse(root.right)
                
            def isValidBST(self, root: Optional[TreeNode]) -> bool:
                self.traverse(root)
                return self.is_valid
            
        ```

    *   

### 700. Search in a Binary Search Tree

*   Time: $O(\log n)$

*   Space: $O(1)$

*   ```python
    class Solution:
        def searchBST(self, root: Optional[TreeNode], val: int) -> Optional[TreeNode]:
            while root:
                if root.val == val:
                    return root
                elif root.val < val:
                    root = root.right
                elif root.val > val:
                    root = root.left
                
            return None
            
    ```

*   Universal solution (recursive):

*   ```python
    class Solution:
        def searchBST(self, root: Optional[TreeNode], val: int) -> Optional[TreeNode]:
            if not root:
                return None
            
            if root.val > val:
                return self.searchBST(root.left, val)
                
            elif root.val < val:
                return self.searchBST(root.right, val)
                
            return root
            
    ```



### !!! 450. Delete Node in a BST

*   !!! Recursive (brilliant idea for `if root.left and root.right:`)

*   ```python
    class Solution:
        def get_min(self, root):
            while root.left:
                root = root.left
            return root
                
            
        def deleteNode(self, root: Optional[TreeNode], key: int) -> Optional[TreeNode]:
            if not root:
                return root
            
            if root.val == key:
                if not root.left and not root.right:
                    root = None
                elif root.left and not root.right:
                    root = root.left
                elif not root.left and root.right:
                    root = root.right
                # we can find the biggest key in the left sub tree or the smallest one in the right sub tree
                else:
                    min_node = self.get_min(root.right)
                    root.right = self.deleteNode(root.right, min_node.val)
                    min_node.left = root.left
                    min_node.right = root.right
                    root = min_node
                    
            elif root.val < key:
                root.right = self.deleteNode(root.right, key)
                
            elif root.val > key:
                root.left = self.deleteNode(root.left, key)
                
            return root
    
    ```

*   /// Iterative : 



### !!! 701. Insert into a Binary Search Tree

*   Time: $O(n)$
*   Space: $O(h)$

*   ```python
    class Solution:
        def insertIntoBST(self, root: Optional[TreeNode], val: int) -> Optional[TreeNode]:
            if not root:
                return TreeNode(val)
            if root.val < val:
                root.right = self.insertIntoBST(root.right, val)
            elif root.val > val:
                root.left = self.insertIntoBST(root.left, val)
            return root
        
    ```

*   /// Iterative:



## 06.09

### 96. Unique Binary Search Trees

>   DP; Catalan Number

*   Time: $O(n^2)$

*   Space: $O(n)$

*   ```python
    class Solution:
        def numTrees(self, n: int) -> int:
            memory = [0] * (n + 1)
            memory[0] = 1
            memory[1] = 1
            
            for num in range(2, n + 1):
                for left in range(num):
                    right = num - 1 - left
                    memory[num] += memory[left] * memory[right]
            
            return memory[-1]
            
    ```

### !!! 95. Unique Binary Search Trees II

*   Time Space: IDK

*   ```python
    class Solution:
        def build(self, start: int, end: int) -> List[Optional[TreeNode]]:
            result = []
            if start > end:
                result.append(None)
                return result
            
            for key in range(start, end + 1):
                left_result = self.build(start, key - 1)
                right_result = self.build(key + 1, end)
                for left in left_result:
                    for right in right_result:
                        # must create root here instead of the outer loop
                        root = TreeNode(key)
                        root.left = left
                        root.right = right
                        result.append(root)
            
            return result
            
        def generateTrees(self, n: int) -> List[Optional[TreeNode]]:
            return self.build(1, n)
            
    ```



### !!! 1373. Maximum Sum BST in Binary Tree

*   **Always remember we need left_max and right_min to valid BST.**

*   Thought

    *   We need results from left and right subtrees, so choose postorder traverse.
    *   *Note that the root.val could be negative.*
    *   For each node:
        *   We need left_max, right_min; left_sum, right_sum; ~~left_max~~, right_max
        *   So we return valid, min, max, sum

*   Time: $O(n)$

*   Space: $O(h)$

*   ```python
    INT_MAX = 4 * pow(10, 4) + 1
    INT_MIN = -INT_MAX
    
    class Solution:
        def __init__(self):
            self.max_val = 0
            
        def traverse(self, root):
            # We need left_max, right_min; left_sum, right_sum; ~~left_max~~, right_max
            # So we return valid, min, max, sum
            if not root:
                return True, INT_MAX, INT_MIN, 0
            left_valid, left_min, left_max, left_sum = self.traverse(root.left)
            right_valid, right_min, right_max, right_sum = self.traverse(root.right)
            # postorder here
            
            if left_valid and right_valid and left_max < root.val < right_min:
                root_min = min(left_min, right_min, root.val)
                root_max = max(left_max, right_max, root.val)
                root_sum = left_sum + right_sum + root.val
                self.max_val = max(self.max_val, root_sum)
                return True, root_min, root_max, root_sum
            else:
                # won't be used anymore
                return False, 0, 0, 0
                
        def maxSumBST(self, root: Optional[TreeNode]) -> int:
            self.traverse(root)
            return self.max_val
        
    ```

### /// 912. Sort an Array

>   Quick sort here: Preorder traverse / Create a BST



## 06.10

### 797. All Paths From Source to Target

>   Traverse a graph

*   Recursive:

    *   Time: idk

    *   ```python
        class Solution:
            def __init__(self):
                self.result = []
                
            def traverse(self, graph, key, path):
                n = len(graph)
                # reach the end
                if key == n - 1:
                    self.result.append(path + [key])
                    return
                else:
                    for target in graph[key]:
                        self.traverse(graph, target, path + [key])
                
            def allPathsSourceTarget(self, graph: List[List[int]]) -> List[List[int]]:
                # start from graph[0], to graph[n-1]
                self.traverse(graph, 0, [])
                
                return self.result
                
        ```

*   Iterative

    *   Time: idk

    *   ```python
        class Solution:
            def allPathsSourceTarget(self, graph: List[List[int]]) -> List[List[int]]:
                stack = []
                result = []
                stack.append((0, [0]))
                while stack:
                    cur, path = stack.pop()
                    if cur == len(graph) - 1:
                        result.append(path)
                    else:
                        for target in graph[cur]:
                            stack.append((target, path + [target]))
                            
                return result
        ```



### !!! 207. Course Schedule

>   Detect cycle in graph

*   DFS & Recursive:

    *   Time: $O(V+E)$?

    *   `self.on_path[cur] = False` is important

    *   ```python
        class Solution:
            def __init__(self):
                self.on_path = []
                self.visited = []
                self.has_cycle = False
                
            def traverse(self, graph, cur):
                if self.on_path[cur]:
                    self.has_cycle = True
                
                if self.visited[cur] or self.has_cycle:
                    return
                
                self.visited[cur] = True
                self.on_path[cur] = True
                for target in graph[cur]:
                    self.traverse(graph, target)
                
                self.on_path[cur] = False
                
            def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
                # DFS
                # Create adjacency list
                self.on_path = [False] * numCourses
                self.visited = [False] * numCourses
                graph = [[] for i in range(numCourses)]
                for target, prereq in prerequisites:
                    graph[prereq].append(target)
                    
                # acyclic means True
                for start in range(numCourses):
                    self.traverse(graph, start)
                    
                return not self.has_cycle
        
        ```

*   BFS:

    *   Time: O(V+E)

    *   ```python
        from collections import deque
        
        class Solution:
            def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
                # BFS
                # Create adjacency list
                graph = [[] for i in range(numCourses)]
                indegree = [0] * numCourses
                for target, prereq in prerequisites:
                    graph[prereq].append(target)
                    indegree[target] += 1
                    
                queue = deque()
                count = 0
                
                for start in range(numCourses):
                    if indegree[start] == 0:
                        queue.append(start)
                
                while queue:
                    cur = queue.popleft()
                    count += 1
                    for target in graph[cur]:
                        indegree[target] -= 1
                        # next level
                        if indegree[target] == 0:
                            queue.append(target)
                    
                return count == numCourses
        
        ```



## 06.14

### 210. Course Schedule II

>   Topological

*   DFS

*   Time: $O(V+E)$

*   Space: $O(V)$

*   ```python
    # DFS
    class Solution:
        def __init__(self):
            self.has_cycle = False
            # size: numCourses; type: Boolean
            # Cauze "in" is O(n) Opt for a list.
            # Extra space for lower time complexity.
            self.visited = []
            self.on_path = []
            self.topo = []
        
        def traverse(self, graph, cur):
            if self.on_path[cur]:
                self.has_cycle = True
            
            if self.visited[cur] or self.has_cycle:
                return
            
            self.on_path[cur] = True
            self.visited[cur] = True
            
            for target in graph[cur]:
                self.traverse(graph, target)
            
            # postoder
            self.topo.append(cur)
            self.on_path[cur] = False
            
            
        def findOrder(self, numCourses: int, prerequisites: List[List[int]]) -> List[int]:
            self.visited = [False] * numCourses
            self.on_path = [False] * numCourses
            graph = [[] for i in range(numCourses)]
            for target, prereq in prerequisites:
                graph[prereq].append(target)
            
            for start in range(numCourses):
                self.traverse(graph, start)
            
            if self.has_cycle:
                return []
            else:
                return self.topo[::-1]
    
    ```

*   BFS

*   Time: $O(V+E)$

*   Space: $O(V)$

*   ```python
    # BFS
    from collections import deque
    
    class Solution:
        def findOrder(self, numCourses: int, prerequisites: List[List[int]]) -> List[int]:
            indegree = [0] * numCourses
            graph = [[] for i in range(numCourses)]
            for target, prereq in prerequisites:
                graph[prereq].append(target)
                indegree[target] += 1
            
            queue = deque()
            count = 0
            result = []
            for start in range(numCourses):
                if indegree[start] == 0:
                    queue.append(start)
            
            while queue:
                cur = queue.popleft()
                count += 1
                result.append(cur)
                for target in graph[cur]:
                    indegree[target] -= 1
                    if indegree[target] == 0:
                        queue.append(target)
                    # early stop
                    elif indegree[target] < 0:
                        return []
            if count == numCourses:
                return result
            else:
                return []
                    
    ```

## 06.15

### 46. Permutations

*   Time: $O(n!)$ I think

*   Space: $O(n)$

*   ```python
    class Solution:
        def __init__(self):
            self.result = []
        def backtrack(self, nums: List[int], track: List[int], used: List[bool]):
            if (len(track) == len(nums)):
                self.result.append(track[:])
                return
            for i, num in enumerate(nums):
                if not used[i]:
                    used[i] = True
                    track.append(num)
                    self.backtrack(nums, track, used)
                    used[i] = False
                    track.pop()
                
        def permute(self, nums: List[int]) -> List[List[int]]:
            used = [False] * len(nums)
            self.backtrack(nums, [], used)
            return self.result
    ```

### 51. N-Queens

*   Time: $--$

*   Space: $--$

*   ```python
    
    class Solution:
        def __init__(self):
            self.result = []
            self.num = 0
        
        def convert_chess(self, track):
            chess = []
            for row, col in enumerate(track):
                chess_row = ['.'] * self.num
                chess_row[col] = 'Q'
                chess.append(''.join(chess_row))
            return chess
        
        def is_valid(self, track: List[int], cur_col: int) -> bool:
            cur_row = len(track) + 1 - 1
            for row, col in enumerate(track):
                if abs(cur_row - row) == abs(cur_col - col) or col == cur_col:
                    return False
            return True
            
        def backtrack(self, track: List[int]):
            if len(track) == self.num:
                self.result.append(self.convert_chess(track))
                return
            for cur_col in range(self.num):
                if self.is_valid(track, cur_col):
                    track.append(cur_col)
                    self.backtrack(track)
                    track.pop()
            
        def solveNQueens(self, n: int) -> List[List[str]]:
            self.num = n
            self.backtrack([])
            return self.result
            
    ```

### !!! 698. Partition to K Equal Sum Subsets

*   Time Limit Exceeded



### 78. Subsets

>   unique, non-duplicate

*   Time: $O(2^n)?$

*   Space: 

*   ```python
    class Solution:
        def __init__(self):
            self.result = []
            
        def backtrack(self, track: List[int], nums: List[int], start: int):
            self.result.append(track[:])
            
            if len(track) == len(nums):
                return
            
            for i in range(start, len(nums)):
                val = nums[i]
                track.append(val)
                self.backtrack(track, nums, i + 1)
                track.pop()
            
        def subsets(self, nums: List[int]) -> List[List[int]]:
            self.backtrack([], nums, 0)
            return self.result
            
    ```



### 77. Combinations

*   ```python
    class Solution:
        def __init__(self):
            self.result = []
            
        def backtrack(self, track, k, n, start):
            if len(track) == k:
                self.result.append(track[:])
                return
            for i in range(start, n):
                val = i + 1 
                track.append(val)
                self.backtrack(track, k, n, i + 1)
                track.pop()
        
        def combine(self, n: int, k: int) -> List[List[int]]:
            self.backtrack([], k, n, 0)
            return self.result
            
    ```



### 90. Subsets II

*   ```python
    class Solution:
        def __init__(self):
            self.result = []
            
        def backtrack(self, track: List[int], nums: List[int], start: int):
            self.result.append(track[:])
            
            if len(track) == len(nums):
                return
            
            for i in range(start, len(nums)):
                val = nums[i]
                if i > start and val == nums[i - 1]:
                    continue
                track.append(val)
                self.backtrack(track, nums, i + 1)
                track.pop()
            
        def subsetsWithDup(self, nums: List[int]) -> List[List[int]]:
            nums.sort()
            self.backtrack([], nums, 0)
            return self.result
            
    ```

### 40. Combination Sum II

*   ```python
    class Solution:
        def __init__(self):
            self.result = []
            
        def backtrack(self, track, cur_sum, target, candidates, start):
            if cur_sum == target:
                return self.result.append(track[:])
    
            for i in range(start, len(candidates)):
                val = candidates[i]
                if cur_sum + val <= target and not (i > start and val == candidates[i - 1]):
                    track.append(val)
                    self.backtrack(track, cur_sum + val, target, candidates, i + 1)
                    track.pop()
            
        def combinationSum2(self, candidates: List[int], target: int) -> List[List[int]]:
            candidates.sort(reverse=True)
            self.backtrack([], 0, target, candidates, 0)
            return self.result
            
    ```



### 47. Permutations II

*   class Solution:
        def __init__(self):
            self.result = []
            

    ```python
    def backtrack(self, track, nums, visited):
        if len(track) == len(nums):
            self.result.append(track[:])
            return
        for i, val in enumerate(nums):
            if visited[i] or (i > 0 and nums[i] == nums[i - 1] and not visited[i-1]):
                continue
            track.append(val)
            visited[i] = True
            self.backtrack(track, nums, visited)
            track.pop()
            visited[i] = False
    
    def permuteUnique(self, nums: List[int]) -> List[List[int]]:
        nums.sort(reverse=True)
        self.backtrack([], nums, [False] * len(nums))
        return self.result
    
    ```



### 39. Combination Sum

*   ```python
    class Solution:
        def __init__(self):
            self.result = []
            
        def backtrack(self, track, cur_sum, target, candidates, start):
            if cur_sum == target:
                self.result.append(track[:])
            for i in range(start, len(candidates)):
                val = candidates[i]
                if cur_sum + val <= target:
                    track.append(val)
                    self.backtrack(track, cur_sum + val, target, candidates, i)
                    track.pop()
        
        def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
            self.backtrack([], 0, target, candidates, 0)
            return self.result
        
    ```



## 06.16

### 200. Number of Islands

>   FloodFill

*   ```python
    # must apply DFS / BFS
    class Solution:   
        def dfs(self, grid, row, col):
            row_length = len(grid)
            col_length = len(grid[0])
            if row == -1 or row == row_length or col == -1 or col == col_length:
                return
            if grid[row][col] == "0":
                return
            grid[row][col] = "0"
            # check up
            self.dfs(grid, row - 1, col)
            # check down
            self.dfs(grid, row + 1, col)
            # check left
            self.dfs(grid, row, col - 1)
            # check right
            self.dfs(grid, row, col + 1)
            
        
        def numIslands(self, grid: List[List[str]]) -> int:
            if not grid or not grid[0]:
                return 0
            row_length = len(grid)
            col_length = len(grid[0])
            count = 0
            for row in range(row_length):
                for col in range(col_length):
                    if grid[row][col] == "1":
                        count += 1
                        # FloodFill
                        self.dfs(grid, row, col)
            
            return count
            
    ```



### 1254. Number of Closed Islands

>   Delete all islands next to edges
>
>   /// union find? BFS?

*   ```python
    class Solution:
        def flood_fill(self, grid, row, col):
            row_length = len(grid)
            col_length = len(grid[0])
            if row < 0 or row >= row_length or col < 0 or col >= col_length:
                return
            if grid[row][col] == 1:
                return
            
            grid[row][col] = 1
            self.flood_fill(grid, row - 1, col)
            self.flood_fill(grid, row + 1, col)
            self.flood_fill(grid, row, col - 1)
            self.flood_fill(grid, row, col + 1)
            
        def closedIsland(self, grid: List[List[int]]) -> int:
            if not grid or not grid[0]:
                return 0
            row_length = len(grid)
            col_length = len(grid[0])
            
            # here is the point
            for row in range(row_length):
                self.flood_fill(grid, row, 0)
                self.flood_fill(grid, row, col_length - 1)
            for col in range(col_length):
                self.flood_fill(grid, 0, col)
                self.flood_fill(grid, row_length - 1, col)
            
            count = 0
            for row in range(row_length):
                for col in range(col_length):
                    if grid[row][col] == 0:
                        count += 1
                        self.flood_fill(grid, row, col)
            return count
            
    ```



### 1020. Number of Enclaves

*   ```python
    class Solution:
        def __init__(self):
            self.count = 0
            
        def flood_fill(self, grid, row, col, is_count=True):
            row_length = len(grid)
            col_length = len(grid[0])
            if row < 0 or row > row_length - 1 or col < 0 or col > col_length - 1:
                return
            if grid[row][col] == 0:
                return
            if is_count:
                self.count += 1
            grid[row][col] = 0
            self.flood_fill(grid, row - 1, col, is_count)
            self.flood_fill(grid, row + 1, col, is_count)
            self.flood_fill(grid, row, col - 1, is_count)
            self.flood_fill(grid, row, col + 1, is_count)
            
        def numEnclaves(self, grid: List[List[int]]) -> int:
            if not grid or not grid[0]:
                return 0
            row_length = len(grid)
            col_length = len(grid[0])
            for row in range(row_length):
                self.flood_fill(grid, row, 0, is_count=False)
                self.flood_fill(grid, row, col_length - 1, is_count=False)
            for col in range(col_length):
                self.flood_fill(grid, 0, col, is_count=False)
                self.flood_fill(grid, row_length - 1, col, is_count=False)   
            for row in range(row_length):
                for col in range(col_length):
                    if grid[row][col] == 1:
                        self.flood_fill(grid, row, col)
            return self.count
            
    ```

### 695. Max Area of Island

*   ```python
    class Solution:
        def flood_fill(self, grid, row, col):
            row_length = len(grid)
            col_length = len(grid[0])
            if row < 0 or row > row_length - 1 or col < 0 or col > col_length - 1:
                return 0
            if grid[row][col] == 0:
                return 0
            grid[row][col] = 0
            count = 1
            count += self.flood_fill(grid, row - 1, col)
            count += self.flood_fill(grid, row + 1, col)
            count += self.flood_fill(grid, row, col - 1)
            count += self.flood_fill(grid, row, col + 1)
            
            return count
            
        def maxAreaOfIsland(self, grid: List[List[int]]) -> int:
            if not grid or not grid[0]:
                return 0
            row_length = len(grid)
            col_length = len(grid[0])
            result = 0
            for row in range(row_length):
                for col in range(col_length):
                    if grid[row][col] == 1:
                        count = self.flood_fill(grid, row, col)
                        if count > result:
                            result = count
            return result
            
    ```

### !!! 1905. Count Sub Islands

>   similar to 1254

*   ```python
    class Solution:
        def flood_fill(self, grid, row, col):
            row_length = len(grid)
            col_length = len(grid[0])
            if row < 0 or row >= row_length or col < 0 or col >= col_length:
                return
            if grid[row][col] == 0:
                return
            
            grid[row][col] = 0
            self.flood_fill(grid, row - 1, col)
            self.flood_fill(grid, row + 1, col)
            self.flood_fill(grid, row, col - 1)
            self.flood_fill(grid, row, col + 1)
            
        def countSubIslands(self, grid1: List[List[int]], grid2: List[List[int]]) -> int:
            if not grid2 or not grid2[0]:
                return 0
            row_length = len(grid2)
            col_length = len(grid2[0])
            count = 0
            for row in range(row_length):
                for col in range(col_length):
                    if grid1[row][col] == 0 and grid2[row][col] == 1:
                        self.flood_fill(grid2, row, col)
            for row in range(row_length):
                for col in range(col_length):
                    if grid2[row][col] == 1:
                        count += 1
                        self.flood_fill(grid2, row, col)
            return count
        
    ```



### 694. Number of Distinct Islands

*   Premium only...



### 111. Minimum Depth of Binary Tree

>   BFS is better thant DFS cause DFS need to visit all nodes but not for BFS.

*   Time: $O(n)$ 

*   Space: $O(n)$

*   ```python
    from collections import deque
    
    class Solution:
        def minDepth(self, root: Optional[TreeNode]) -> int:
            if not root:
                return 0
            queue = deque()
            queue.append(root)
            depth = 0
            while queue:
                depth += 1
                size = len(queue)
                for i in range(size):
                    cur = queue.popleft()
                    if not cur.left and not cur.right:
                        return depth
                    if cur.left:
                        queue.append(cur.left)
                    if cur.right:
                        queue.append(cur.right)
            return depth
        
    ```



### !!! 752. Open the Lock

*   ```python
    from collections import deque
    
    class Solution:
        def plus_wheel(self, cur, pos):
            temp = int(cur[pos])
            if temp == 9:
                temp = 0
            else:
                temp += 1
            return cur[:pos] + str(temp) + cur[pos + 1:]
        
        def minus_wheel(self, cur, pos):
            temp = int(cur[pos])
            if temp == 0:
                temp = 9
            else:
                temp -= 1
            return cur[:pos] + str(temp) + cur[pos + 1:]
        
        def openLock(self, deadends: List[str], target: str) -> int:
            visited = set(deadends)
            if "0000" in visited:
                return -1
            queue = deque()
            queue.append("0000")
            step = 0
            while queue:
                size = len(queue)
                for i in range(size):
                    cur = queue.popleft()
                    if cur == target:
                        return step
                    for pos in range(4):
                        plus = self.plus_wheel(cur, pos)
                        minus = self.minus_wheel(cur, pos)
                        if plus not in visited:
                            queue.append(plus)
                            visited.add(plus)
                        if minus not in visited:
                            queue.append(minus)
                            visited.add(minus)
                step += 1
                
            return -1
    
    ```

*   /// 2 ends

## 06.17

### 773. Sliding Puzzle

*   from collections import deque

    class Solution:
        def __init__(self):
            self.neighbor = [[1, 3], [0, 2, 4], [1, 5], [0, 4], [1, 3, 5], [2, 4]]
            
    ```python
    def move(self, board, zero, tar):
        new_board = list(board)
        new_board[zero] = board[tar]
        new_board[tar] = '0'
        return ''.join(new_board)
        
    def slidingPuzzle(self, board: List[List[int]]) -> int:
        init_board_list = []
        init_zero = 0
        for row in range(2):
            for col in range(3):
                init_board_list.append(str(board[row][col]))
                if board[row][col] == 0:
                    init_zero = row * 3 + col
        init_board = ''.join(init_board_list)
        visited = set()
        visited.add(init_board)
        target = "123450"
        step = 0
        queue = deque()
        queue.append((init_board, init_zero))
        while queue:
            size = len(queue)
            for i in range(size):
                cur_board, zero = queue.popleft()
                if cur_board == target:
                    return step
                for nei in self.neighbor[zero]:
                    new_board = self.move(cur_board, zero, nei)
                    if new_board in visited:
                        continue
                    queue.append((new_board, nei))
                    visited.add(new_board)
            step += 1
        return -1
        
    ```

### 509. Fibonacci Number

*   Time: $O(n)$; Space: $O(n)$.

*   ```python
    class Solution:
        def fib(self, n: int) -> int:
            if n <= 1:
                return n
            f = [0] * (n + 1)
            f[1] = 1
            for i in range(2, n + 1):
                f[i] = f[i - 1] + f[i - 2]
            return f[n]
    
    ```

*   Actually we only need `f[i-1]` and `f[i-2]`

*   Time: $O(n)$; Space: $O(1)$.

*   ```python
    class Solution:
        def fib(self, n: int) -> int:
            if n <= 1:
                return n
            f_1 = 1
            f_2 = 0
            for i in range(2, n + 1):
                result = f_1 + f_2
                f_2 = f_1
                f_1 = result
            return f_1
            
    ```

### 322. Coin Change

*   Top-down

    *   ```python
        class Solution:
            # dp[n] = -1, n < 0
            # dp[n] = 0, n = 0
            # dp[n] = min(dp(n - coint) + 1 | for coin in coins), n > 0
            def __init__(self):
                self.amount = 0
                self.memo = []
                
            def dp(self, coins, amount):
                if amount == 0:
                    return 0
                if amount < 0:
                    return -1
                if self.memo[amount] != self.amount + 1:
                    return self.memo[amount]
                
                result = self.amount + 1
                for coin in coins:
                    sub_problem = self.dp(coins, amount - coin)
                    if sub_problem == -1:
                        continue
                    result = min(result, sub_problem + 1)
                
                self.memo[amount] = result if result < self.amount + 1 else -1
                return self.memo[amount]
            
            def coinChange(self, coins: List[int], amount: int) -> int:
                self.amount = amount
                self.memo = [amount + 1] * (amount + 1)
                return self.dp(coins, amount)
                
        ```

## 06.20

### 322. Coin Change

*   Top-down

*   ```python
    # Top-down
    class Solution:
        # memo[amount] = num of coins
        def __init__(self):
            self.memo = []
            
        def dp(self, coins, amount):
            if amount == 0:
                return 0
            if amount < 0:
                return -1
            if self.memo[amount] != -2:
                return self.memo[amount]
            result = amount + 1
            for coin in coins:
                # if amount - coin < 0:
                #     continue
                sub_problem = self.dp(coins, amount - coin)
                if sub_problem != -1:
                    result = min(result, sub_problem + 1)
            self.memo[amount] = result if result < amount + 1 else -1
            return self.memo[amount]
            
        def coinChange(self, coins: List[int], amount: int) -> int:
            # -2 means non-visited
            self.memo = [-2] * (amount + 1)
            return self.dp(coins, amount)
    
    ```

*   !!! Bottom-up

*   ```python
    # Bottom up
    class Solution:
        def coinChange(self, coins: List[int], amount: int) -> int:
            dp = [amount + 1] * (amount + 1)
            # base state
            dp[0] = 0
            for state in range(amount + 1):
                for coin in coins:
                    # no solution
                    if state - coin < 0:
                        continue
                    dp[state] = min(dp[state], dp[state - coin] + 1)
            
            return dp[amount] if dp[amount] != amount + 1 else -1
    
    ```



### 931. Minimum Falling Path Sum

*   ```python
    class Solution:
        def minFallingPathSum(self, matrix: List[List[int]]) -> int:
            size = len(matrix)
            dp_pre = matrix[0]
            dp_cur = dp_pre[:]
            for i in range(1, size):
                for j in range(size):
                    if j == 0:
                        dp_cur[j] = min(dp_pre[j], dp_pre[j + 1]) + matrix[i][j]
                    elif j == size - 1:
                        dp_cur[j] = min(dp_pre[j - 1], dp_pre[j]) + matrix[i][j]
                    else:
                        dp_cur[j] = min(dp_pre[j - 1], dp_pre[j], dp_pre[j + 1])+ matrix[i][j]
                dp_pre = dp_cur[:]
            return min(dp_cur)
                    
    ```

### !!! 300. Longest Increasing Subsequence

*   DP:

    *   Time: $O(n^2)$

    *   ```python
        # dp[i]: the longest subsequence ending at i
        class Solution:
            def lengthOfLIS(self, nums: List[int]) -> int:
                size = len(nums)
                dp = [1] * size
                for i in range(1, size):
                    for j in range(i):
                        if nums[i] > nums[j]:
                            dp[i] = max(dp[i], dp[j] + 1)
                return max(dp)
                
        ```

    *   

*   /// Binary Search

    *   Time: $O(n\log n)$

### /// !!! 354. Russian Doll Envelopes

*   DP (Time Limit Exceeded):

    *   Time: $O(n^2)$

    *   ```python
        class Solution:
            def maxEnvelopes(self, envelopes: List[List[int]]) -> int:
                size = len(envelopes)
                # !!!
                envelopes = sorted(envelopes, key = lambda x: [x[0], -x[1]])
                dp = [1] * size
                for i in range(1, size):
                    cur_h = envelopes[i][1]
                    for j in range(i):
                        pre_h = envelopes[j][1]
                        if cur_h > pre_h:
                            dp[i] = max(dp[i], dp[j] + 1)
                return max(dp)
                
        ```

    *   

*   /// Binary Search

    *   Time: $O(n\log n)$

## 06.21

### !!! 53. Maximum Subarray

*   DP

    *   Time: $O(n)$

    *   ```python
        class Solution:
            def maxSubArray(self, nums: List[int]) -> int:
                dp_0 = nums[0]
                dp_1 = 0
                result = dp_0
                for i in range(1, len(nums)):
                    dp_1 = max(nums[i], dp_0 + nums[i])
                    dp_0 = dp_1
                    result = max(result, dp_1)
                return result
        
        ```

    *   

*   /// divide and conquer



### !!! 72. Edit Distance

*   ```python
    class Solution:
        def minDistance(self, word1: str, word2: str) -> int:
            size_1 = len(word1)
            size_2 = len(word2)
            dp = [[0] * (size_2 + 1) for i in range(size_1 + 1)]
            # base case
            for i in range(size_1 + 1):
                dp[i][0] = i
            for j in range(size_2 + 1):
                dp[0][j] = j
            i = j = 1
            for i in range(1, size_1 + 1):
                for j in range(1, size_2 + 1):
                    if word1[i-1] == word2[j-1]:
                        dp[i][j] = dp[i-1][j-1]
                    else:
                        dp[i][j] = min(dp[i][j-1] + 1, dp[i-1][j] + 1, dp[i-1][j-1] + 1)
            return dp[size_1][size_2]
                    
    ```

### 1143. Longest Common Subsequence

*   ```python
    class Solution:
        def longestCommonSubsequence(self, text1: str, text2: str) -> int:
            size_1 = len(text1)
            size_2 = len(text2)
            dp = [[0] * (size_2 + 1) for i in range(size_1 + 1)]
            for i in range(1, size_1 + 1):
                for j in range(1, size_2 + 1):
                    if text1[i - 1] == text2[j - 1]:
                        dp[i][j] = dp[i - 1][j - 1] + 1
                    else:
                        dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
            return dp[size_1][size_2]
            
    ```

### !!! /// 10. Regular Expression Matching



### 518. Coin Change 2

*   Note the permutation except the same coin leads to wrong num. (e.g., 1, 1, 2 and 2, 1, 1 and 1, 2, 1 should be treat as one result.)

*   Time: $O(n * amount)$

*   Space: $O(amount)$

*   ```python
    class Solution:
        def change(self, amount: int, coins: List[int]) -> int:
            dp = [0] * (amount + 1)
            dp[0] = 1
            # !!! the loop order is different to coin change 1
            # We remove duplicate cases in this way.
            for coin in coins:
                for state in range(1, amount + 1):
                    if state - coin < 0:
                        continue
                    dp[state] += dp[state-coin]
            return dp[amount]
            
    ```



### !!! 416. Partition Equal Subset Sum

*   !!! Similar to the last question, we fix the order of selections, but they can get selected only once. So we reverse the state iteration order to aviod reusing. (e.g., [1, 2, 5], then 1 will be reused to set all dp to be True)

*   ```python
    class Solution:
        def canPartition(self, nums: List[int]) -> bool:
            sum_val = sum(nums)
            if sum_val % 2 != 0:
                return False
            target = int(sum_val / 2)
            dp = [False] * (target + 1)
            dp[0] = True
            for select in nums:
                for state in range(target, 0, -1):
                    if state - select < 0:
                        continue
                    dp[state] = dp[state - select] or dp[state]
            return dp[target]
            
    ```

### 121. Best Time to Buy and Sell Stock

*   ```python
    class Solution:
        def maxProfit(self, prices: List[int]) -> int:
            MAX_VAL = 10000
            day_num = len(prices)
            dp_0 = 0
            dp_1 = -MAX_VAL
            for day in range(1, day_num + 1):
                dp_0 = max(dp_0, dp_1 + prices[day-1])
                dp_1 = max(dp_1, -prices[day-1])
            return dp_0
        
        # def maxProfit(self, prices: List[int]) -> int:
        #     MAX_VAL = 10000
        #     day_num = len(prices)
        #     dp = [[0] * 2 for i in range(day_num + 1)]
        #     dp[0][1] = -MAX_VAL
        #     for day in range(1, day_num + 1):
        #         dp[day][0] = max(dp[day-1][0], dp[day-1][1] + prices[day-1])
        #         dp[day][1] = max(dp[day-1][1], -prices[day-1])
        #     return dp[day_num][0]
    
    ```

### 122. Best Time to Buy and Sell Stock II

*   ```python
    # same
    class Solution:
        def maxProfit(self, prices: List[int]) -> int:
            day_num = len(prices)
            dp_0, dp_1 = 0, -10000
            for day in range(1, day_num + 1):
                temp_0, temp_1 = dp_0, dp_1
                dp_0 = max(temp_0, temp_1 + prices[day-1])
                dp_1 = max(temp_1, temp_0 - prices[day-1])
            return dp_0
            
    ```



### !!! 123. Best Time to Buy and Sell Stock III

*   Note that `tran` is the maximun transaction num

*   ```python
    class Solution:
        def maxProfit(self, prices: List[int]) -> int:
            day_num = len(prices)
            # dp[day][tran][has?]
            dp = [[[0] * 2 for j in range(3)] for i in range(day_num + 1)]
            for j in range(3):
                dp[0][j][1] = -100000
            for day in range(1, day_num + 1):
                dp[day][0][0] = 0
                for tran in range(2, 0, -1):
                    dp[day][tran][0] = max(dp[day-1][tran][0], dp[day-1][tran][1] + prices[day-1])
                    dp[day][tran][1] = max(dp[day-1][tran][1], dp[day-1][tran-1][0] - prices[day-1])
            return dp[day_num][2][0]
        
    ```



## 06.22

### 188. Best Time to Buy and Sell Stock IV

*   ```python
    # state 1: day; state 2: k; state 3: has?
    # base case
    class Solution:
        def maxProfit(self, k: int, prices: List[int]) -> int:
            day_num = len(prices)
            dp = [[[0] * 2 for j in range(k + 1)] for i in range(day_num + 1)]
            INFINITY = 1000
            for j in range(k + 1):
                dp[0][j][1] = -INFINITY
            
            for i in range(1, day_num + 1):
                for j in range(k, 0, -1):
                    dp[i][j][0] = max(dp[i-1][j][0], dp[i-1][j][1] + prices[i-1])
                    dp[i][j][1] = max(dp[i-1][j][1], dp[i-1][j-1][0] - prices[i-1])
            return dp[day_num][k][0]
            
    ```

*   **the loop order for k is not matter since `dp[i][k][]` won't depend on `dp[i][k-1][]`**



### !!! 309. Best Time to Buy and Sell Stock with Cooldown

*   ```python
    class Solution:
        def maxProfit(self, prices: List[int]) -> int:
            day_num = len(prices)
            dp_0, dp_1, dp_pre_0 = 0, -1000, 0
            for day in range(day_num):
                if day + 1 == 1:
                    dp_0 = 0
                    dp_1 = -prices[day]
                elif day + 1 == 2:
                    dp_0 = max(dp_0, dp_1 + prices[day])
                    dp_1 = max(dp_1, -prices[day])
                else:
                    temp = dp_0
                    dp_0 = max(dp_0, dp_1 + prices[day])
                    dp_1 = max(dp_1, dp_pre_0 - prices[day])
                    dp_pre_0 = temp
            return dp_0
        
    #     def maxProfit(self, prices: List[int]) -> int:
    #         day_num = len(prices)
    #         dp = [[0] * 2 for i in range(day_num)]
            
    #         for day in range(day_num):
    #             if day + 1 == 1:
    #                 dp[day][1] = -prices[day]
    #             elif day + 1 == 2:
    #                 dp[day][0] = max(dp[day-1][0], dp[day-1][1] + prices[day])
    #                 dp[day][1] = max(dp[day-1][1], -prices[day])
    #             else:
    #                 dp[day][0] = max(dp[day-1][0], dp[day-1][1] + prices[day])
    #                 dp[day][1] = max(dp[day-1][1], dp[day-2][0] - prices[day])
    #         return dp[day_num-1][0]
    
    ```

*   



### 714. Best Time to Buy and Sell Stock with Transaction Fee

*   ```python
    class Solution:
        def maxProfit(self, prices: List[int], fee: int) -> int:
            day_num = len(prices)
            dp_0 = 0
            dp_1 = -50000
            for day in range(1, day_num + 1):
                temp = dp_0
                dp_0 = max(dp_0, dp_1 + prices[day-1])
                dp_1 = max(dp_1, temp - prices[day-1] - fee)
            return dp_0
        # def maxProfit(self, prices: List[int], fee: int) -> int:
        #     day_num = len(prices)
        #     dp = [[0] * 2 for day in range(day_num + 1)]
        #     dp[0][1] = -50000
        #     for day in range(1, day_num + 1):
        #         dp[day][0] = max(dp[day-1][0], dp[day-1][1] + prices[day-1])
        #         dp[day][1] = max(dp[day-1][1], dp[day-1][0] - prices[day-1] - fee)
        #     return dp[day_num][0]
    ```



### 198. House Robber

*   Analysis:

    *   State: location in nums;
    *   Selection: Rob its or not

*   ```python
    class Solution:
        def rob(self, nums: List[int]) -> int:
            dp_0 = 0
            dp_1 = nums[0]
            for i in range(1, len(nums)):
                temp = dp_0
                dp_0 = max(dp_0, dp_1)
                dp_1 = temp + nums[i]
            return max(dp_0, dp_1)
            
    ```



### !!! 213. House Robber II

*   There are 2 cases:

    *   Exclude the last one
    *   Exclude the first one

*   ```python
    class Solution:
        def selected_rob(self, nums, start, end):
            dp_0 = 0
            dp_1 = nums[start]
            for i in range(start + 1, end):
                temp = dp_0
                dp_0 = max(dp_0, dp_1)
                dp_1 = temp + nums[i]
            return max(dp_0, dp_1)
        
        def rob(self, nums: List[int]) -> int:
            if len(nums) == 1:
                return nums[0]
            return max(self.selected_rob(nums, 0, len(nums)-1), self.selected_rob(nums, 1, len(nums)))
            
            
    ```



### 337. House Robber III

>   Traverse a binary tree & DP

*   Traverse: what should we do at each root? Give the max value of this subtree containing the root or not. -> `max_0` & `max_1`

    *   We need to know subtrees first -> postorder traverse

*   ```python
    # Definition for a binary tree node.
    # class TreeNode:
    #     def __init__(self, val=0, left=None, right=None):
    #         self.val = val
    #         self.left = left
    #         self.right = right
    class Solution:
        def traverse(self, root):
            if not root:
                return 0, -10000
            max_left_0, max_left_1 = self.traverse(root.left)
            max_right_0, max_right_1 = self.traverse(root.right)
            # dp here
            max_0 = max(max_left_0, max_left_1) + max(max_right_0, max_right_1)
            max_1 = root.val + max_left_0 + max_right_0
            return max_0, max_1
            
        def rob(self, root: Optional[TreeNode]) -> int:
            return max(self.traverse(root))
            
    ```



### !!! 877. Stone Game

*   (bad) Top-down recursive

    *   Time & Space: $O(n^2)$

    *   ```python
        class Solution:
            def stoneGame(self, piles: List[int]) -> bool:
                @lru_cache(None)
                def dp(start, end):
                    if start > end:
                        return 0, 0
                    start_first, start_then = dp(start + 1, end)
                    end_first, end_then= dp(start, end - 1)
                    # start_then is Alice's next selection
                    if start_then + piles[start] > end_then + piles[end]:
                        return start_then + piles[start], start_first
                    return end_then + piles[end], end_first
                alice, bob = dp(0, len(piles) - 1)
                return True if alice > bob else False
                
        ```

    *   Time Limit Exceeded without `@lru_cache(None)`

*   continue this tomorrow



## 06.23

### 877. Stone Game

*   The loop order depends on the **State Equation**

*   Bottom-up --- Top-down with dp table --- Top-down

*   ```python
    # Bottom-up
    class Solution:
        def stoneGame(self, piles: List[int]) -> bool:
            size = len(piles)
            dp = [[[0] * 2 for j in range(size)] for i in range(size)]
            for end in range(1, size):
                for start in range(end - 1, -1, -1):
                    if piles[start] + dp[start + 1][end][1] > piles[end] + dp[start][end - 1][1]:
                        first = piles[start] + dp[start + 1][end][1]
                        second = dp[start + 1][end][0]
                    else:
                        first = piles[end] + dp[start][end - 1][1]
                        second = dp[start][end - 1][0]
                    dp[start][end] = [first, end]
            alice, bob = dp[0][size-1]
            return True if alice > bob else False
        
    #################
    # Recursive with dp table
    # class Solution:
    #     def __init__(self):
    #         self.memo = [[[]]]
        
    #     def dp(self, piles, start, end):
    #         if start > end:
    #             return 0, 0
    #         if self.memo[start][end][0] != -1:
    #             return self.memo[start][end]
    #         start_first, start_second = self.dp(piles, start + 1, end)
    #         end_first, end_second = self.dp(piles, start, end - 1)
    #         if start_second + piles[start] > end_second + piles[end]:
    #             self.memo[start][end] = [start_second + piles[start], start_first]
    #             return start_second + piles[start], start_first
    #         else:
    #             self.memo[start][end] = [end_second + piles[end], end_first]
    #             return end_second + piles[end], end_first
            
    #     def stoneGame(self, piles: List[int]) -> bool:
    #         size = len(piles)
    #         self.memo = [[[-1] * 2 for j in range(size)] for i in range(size)]
    #         for i in range(size):
    #             self.memo[i][i] = [0, 0]
    #         alice, bob = self.dp(piles, 0, size - 1)
    #         return True if alice > bob else False
    
    #################
    # Recursive
    # class Solution:
    #     def stoneGame(self, piles: List[int]) -> bool:
    #         @lru_cache(None)
    #         def dp(start, end):
    #             if start > end:
    #                 return 0, 0
    #             start_first, start_then = dp(start + 1, end)
    #             end_first, end_then= dp(start, end - 1)
    #             # start_then is Alice's next selection
    #             if start_then + piles[start] > end_then + piles[end]:
    #                 return start_then + piles[start], start_first
    #             return end_then + piles[end], end_first
    #         alice, bob = dp(0, len(piles) - 1)
    #         return True if alice > bob else False
            
    ```



### 64. Minimum Path Sum

*   ```python
    class Solution:
        def minPathSum(self, grid: List[List[int]]) -> int:
            row_size = len(grid)
            col_size = len(grid[0])
            dp = [[0] * col_size for i in range(row_size)]
            dp[0][0] = grid[0][0]
            for row in range(1, row_size):
                dp[row][0] = grid[row][0] + dp[row-1][0]
            for col in range(1, col_size):
                dp[0][col] = grid[0][col] + dp[0][col-1]
            for row in range(1, row_size):
                for col in range(1, col_size):
                    dp[row][col] = grid[row][col] + min(dp[row-1][col], dp[row][col-1])
            return dp[row_size-1][col_size-1]
        
    ```



### !!! /// 887. Super Egg Drop

*   Time Limit Exceeded

*   Time: $O(k*n^2)$

*   Space: $O(k*n)$

*   ```python
    # dp[n][k]
    class Solution:
        def superEggDrop(self, k: int, n: int) -> int:
            memo = [[-1] * (n + 1) for i in range(k + 1)]
            @lru_cache(None)
            def dp(k, n):
                if k == 1:
                    return n
                if n == 0:
                    return 0
                if memo[k][n] != -1:
                    return memo[k][n]
                result = 10000
    
                for i in range(1, n + 1):
                    # in least egg we take in the worst situation.
                    # (last selection, (egg broke in ith, not broke) + 1)
                    result = min(result, max(dp(k-1, i-1), dp(k, n-i)) + 1)
                memo[k][n] = result
                return result
            return dp(k, n)
        
    ```



### 787. Cheapest Flights Within K Stops

*   dp

    *   Time: $O(E * n * k)$?

    *   ```python
        # graph; bfs -> dp
        # state: start, most flights
        # equation: dp[start][k] = min(dp[i][k-1] + price)
        class Solution:
            def findCheapestPrice(self, n: int, flights: List[List[int]], src: int, dst: int, k: int) -> int:
                # bfs -> we build the graph -> has weight -> adjacency matrix
                # k becomes maximum step
                graph = [[] for i in range(n)]
                for start, end, price in flights:
                    graph[start].append((end, price))
                memo = [[-2] * (k + 1) for i in range(n)]
                MAX_VALUE = 10000 * (k + 1)
                @lru_cache(None)
                def dp(start, k):
                    if start == dst:
                        return 0
                    if k == -1:
                        return -1
                    if memo[start][k] != -2:
                        return memo[start][k]
                    cost = MAX_VALUE
                    for stop, price in graph[start]:
                        next_cost = dp(stop, k - 1)
                        if next_cost == -1:
                            continue
                        cost = min(cost, next_cost + price)
                    memo[start][k] = -1 if cost == MAX_VALUE else cost
                    return memo[start][k]
                
                return dp(src, k)
                
        ```

    *   

*   /// Dijkstra

### ///204. Count Primes

>   Math

*   Create a table

*   /// Time: $O(N * \log \log N)$

*   ```python
    class Solution:
        def countPrimes(self, n: int) -> int:
            if n <= 1:
                return 0
            # Note it is "less than n"
            table = [1] * n
            table[0] = 0
            table[1] = 0
            left = 2
            while left * left < n:
                if table[left] == 1:
                    right = left * left
                    while right < n:
                        table[right] = 0
                        right += left
                left += 1
            return sum(table)
    
    ```



### +++ 172. Factorial Trailing Zeroes

>   math

*   ```python
    class Solution:
        # sum_2 always > sum_5
        def trailingZeroes(self, n: int) -> int:
            sum_5 = 0
            dividor = 5
            while dividor <= n:
                sum_5 += n // dividor
                dividor = dividor * 5
            return sum_5
            
    ```



### /// 793. Preimage Size of Factorial Zeroes Function



## 06.24

### !!! +++ 382. Linked List Random Node

*   p = 1/i

*   ```python
    # Definition for singly-linked list.
    # class ListNode:
    #     def __init__(self, val=0, next=None):
    #         self.val = val
    #         self.next = next
    import random
    
    class Solution:
    
        def __init__(self, head: Optional[ListNode]):
            self.head = ListNode()
            self.head.next = head
            
        def getRandom(self) -> int:
            loc = 0
            head = self.head
            result = -1
            while head.next:
                loc += 1
                head = head.next
                r = random.randint(1, loc)
                if loc == r:
                    result = head.val
            return result
    
    
    # Your Solution object will be instantiated and called as such:
    # obj = Solution(head)
    # param_1 = obj.getRandom()
    
    ```



### +++!!!134. Gas Station

>   Greedy

*   Time: $O(n)$

*   ```python
    class Solution:
        def travel(self, cost, gas):
            rest = gas - cost
            return -1 if rest < 0 else cost
        
        def canCompleteCircuit(self, gas: List[int], cost: List[int]) -> int:
            rest = [gas[i] - cost[i] for i in range(len(gas))]
            sum_val = sum(rest)
            if sum_val < 0:
                return -1
            tank = 0
            start = 0
            for i in range(len(rest)):
                tank += rest[i]
                if tank < 0:
                    start = i + 1
                    tank = 0
            return start if start != len(rest) else 0
            
    ```

## 06.27

### /// 645. Set Mismatch

### !!! 15. 3Sum

*   Time: $O(n^2)$ (`sort()` $O(n\log n)$)

*   ```python
    class Solution:
        def two_sum(self, nums, start, target):
            left = start
            right = len(nums) - 1
            result = []
            while left < right:
                left_val = nums[left]
                right_val = nums[right]
                sum_val = left_val + right_val
                if  sum_val == target:
                    result.append([-target, left_val, right_val])
                    while left < right and nums[left] == left_val:
                        left += 1
                    while left < right and nums[right] == right_val:
                        right -= 1
                elif sum_val < target:
                    while left < right and nums[left] == left_val:
                        left += 1
                elif sum_val > target:
                    while left < right and nums[right] == right_val:
                        right -= 1
            return result
            
        def threeSum(self, nums: List[int]) -> List[List[int]]:
            result = []
            size = len(nums)
            nums.sort()
            i = 0
            while i < size:
                cur_i = nums[i]
                temp_result = self.two_sum(nums, i + 1, -cur_i) 
                result += temp_result
                while i < size and nums[i] == cur_i:
                    i += 1
            return result
            
    ```



### +++ 1288. Remove Covered Intervals

*   ```python
    class Solution:
        def removeCoveredIntervals(self, intervals: List[List[int]]) -> int:
            # sort key_1: start ascending; key_2: end descending
            intervals.sort(key=lambda x: (x[0], -x[1]))
            start, end = intervals[0]
            size = len(intervals)
            count = 0
            for i in range(1, size):
                left, right = intervals[i]
                # cover
                if start <= left and end >= right:
                    count += 1
                # intersect
                elif left <= end and right >= end:
                    end = right
                # not itersect
                elif left > end:
                    start = left
                    end = right
            return size - count
                
    ```



### 56. Merge Intervals

*   From last problem:

    *   ```python
        class Solution:
            def merge(self, intervals: List[List[int]]) -> List[List[int]]:
                intervals.sort(key=lambda x: (x[0], -x[1]))
                start_mark, end_mark = intervals[0]
                result = []
                start, end = 0, 0
                for i in range(1, len(intervals)):
                    start, end = intervals[i]
                    # cover
                    if start_mark <= start and end <= end_mark:
                        continue
                    # intersect
                    elif start <= end_mark <= end:
                        end_mark = end
                    # not intersect
                    elif end_mark < start:
                        result.append([start_mark, end_mark])
                        start_mark, end_mark = start, end
                result.append([start_mark, end_mark])
                return result
            
        ```

    *   

*   ```python
    class Solution:
        def merge(self, intervals: List[List[int]]) -> List[List[int]]:
            intervals.sort(key=lambda x: x[0])
            result = []
            start_mark, end_mark = intervals[0]
            for i in range(1, len(intervals)):
                start, end = intervals[i]
                if start > end_mark:
                    result.append([start_mark, end_mark])
                    start_mark, end_mark = start, end
                else:
                    end_mark = max(end, end_mark)
            result.append([start_mark, end_mark])
            return result
            
    ```

*   



### ///1024. Video Stitching







