# Array & Linked List

*   Two Pointers
*   Binary search
    *   Open Interval / Closed Interval decides the condition ($<$ or $\leq$) in `while` and `right = len()` or `right = len()-1`
    *   `while` determines when it ends.
    *   `left = mid + 1 or mid` (`right = mid - 1 or mid`) depend on the search interval.
    *   `mid = left + ((right - left) >> 1)`: prevent int overflow.
*   PreSum
*   *Monotonic Stack* (increasing or decreasing)
*   Monotonic Queue

    *   ```python
        class MonotonicQueue:
            def push():
            def pop():
            def max/min():
        ```



# Binary Tree

*   Outline

    >   What we do for each node, and when?

    *   Traverse -- Backtracking
    *   Sub-problem -- DP
        *   Create trees.

*   Traverse:

    *   Preorder: Know info from father nodes
    *   Postorder: Know info from child nodes
        *   To get info from sub-tree,  we need postorder one.

