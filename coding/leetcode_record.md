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

### 5. Longest Palindromic Substring

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

    *   <img src="/Users/rraymondy/Desktop/mle-preparation/coding/leetcode_record.assets/image-20220603123344908.png" alt="image-20220603123344908" style="zoom: 25%;" />

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

    *   <img src="/Users/rraymondy/Desktop/mle-preparation/coding/leetcode_record.assets/image-20220603130232556.png" alt="image-20220603130232556" style="zoom:25%;" />

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



## 06.04

















