"""
Prompt constants for code generation tasks.
"""

from lcb_runner.benchmarks.code_generation import CodeGenerationProblem, Platform


class PromptConstants:
    SYSTEM_MESSAGE_GENERIC = "You are an expert Python programmer. You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests."

    SYSTEM_MESSAGE_GEMINI = "You are an expert Python programmer. You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests. Do NOT use system calls like `exit` in the generated program. Ensure that the first code block contains the solution."

    SYSTEM_MESSAGE_GEMINITHINK = "You are an expert Python programmer. You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests."

    SYSTEM_MESSAGE_DEEPSEEK = "You are an AI programming assistant, utilizing the DeepSeek Coder model, developed by DeepSeek Company, and you answer questions related to computer science."

    SYSTEM_MESSAGE_CODEQWEN = (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user"
    )

    SYSTEM_MESSAGE_QWEN_QWQ = "<|im_start|>system\nYou are a helpful and harmless assistant. You are Qwen developed by Alibaba. You should think step-by-step.<|im_end|>\n<|im_start|>user"

    SYSTEM_MESSAGE_DEEPSEEK_R1 = (
        "<｜begin▁of▁sentence｜>A conversation between User and Assistant. "
        "The user asks a question, and the Assistant solves it. "
        "The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. "
        "The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.<｜User｜>"
    )

    FORMATTING_MESSAGE_WITH_STARTER_CODE = "You will use the following starter code to write the solution to the problem and enclose your code within delimiters."

    FORMATTING_WITHOUT_STARTER_CODE = "Read the inputs from stdin solve the problem and write the answer to stdout (do not directly test on the sample inputs). Enclose your code within delimiters as follows. Ensure that when the python program runs, it reads the inputs, runs the algorithm and writes output to STDOUT."

    COF_LEETCODE = """
### Code Organization Guidelines:

1. **Function Organization**:
   - Implement solution inside the `Solution` class method
   - Place helper methods within the `Solution` class

2. **Write Docstrings in Each Function** (Chain of Thought):
   - **CRITICAL**: Add a docstring using triple quotes (''') at the beginning of each function
   - **Main methods**: Explain approach, algorithm choice, strategy, and key steps
   - **Helper methods**: Explain specific logic and purpose

3. **Helper Methods**: Extract logic into separate methods when the operation is repeated or complex calculations can be isolated

**Important**: The example shows CODE STRUCTURE and DOCSTRING STYLE only. Solve the actual problem, NOT the example algorithm.

Example demonstrating format:
```python
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        '''
        Strategy: Use sliding window with a hash set to track unique characters.
        Maintain window [left, right] and expand by moving right pointer.
        When duplicate found, shrink from left until duplicate removed.
        '''
        if not s:
            return 0

        char_set = set()
        left = 0
        max_length = 0

        for right in range(len(s)):
            while s[right] in char_set:
                char_set.remove(s[left])
                left += 1

            char_set.add(s[right])
            max_length = max(max_length, right - left + 1)

        return max_length

    def lengthOfLongestSubstringAlternative(self, s: str) -> int:
        '''
        Alternative: Use hash map to store last seen index of each character.
        Jump left pointer directly to position after last occurrence instead of
        incrementing one by one.
        '''
        char_index = {}
        left = 0
        max_length = 0

        for right, char in enumerate(s):
            if char in char_index and char_index[char] >= left:
                left = char_index[char] + 1

            char_index[char] = right
            max_length = max(max_length, right - left + 1)

        return max_length
```
"""

    COF_GENERAL = """
### Code Organization Guidelines:

1. **Function Organization**:
   - Place high-level solving functions first (e.g., `main()`, primary algorithm)
   - Follow with helper functions implementing specific subtasks
   - Avoid nested functions; keep all functions at top level

2. **Write Docstrings in Each Function** (Chain of Thought):
   - **CRITICAL**: Add a docstring using triple quotes (''') at the beginning of each function
   - **High-level functions**: Explain approach, algorithm choice, strategy, and key steps
   - **Helper functions**: Explain specific logic and purpose

3. **Helper Functions**: Extract logic into separate functions when the operation is repeated or complex calculations can be isolated

**Important**: The example shows CODE STRUCTURE and DOCSTRING STYLE only. Solve the actual problem, NOT the example algorithm.

Example demonstrating format:
```python
def main():
    '''
    Strategy: Use Dijkstra's algorithm to find shortest path in weighted graph.
    Read graph from stdin (n nodes, m edges, followed by edge list with weights),
    compute shortest path from node 0 to n-1 using priority queue.
    Output -1 if no path exists.
    '''
    n, m = map(int, input().split())
    graph = build_graph(n, m)

    result = dijkstra(graph, 0, n - 1)

    print(result if result != float('inf') else -1)

def dijkstra(graph, start, end):
    '''
    Implements Dijkstra's algorithm using min-heap priority queue.
    Maintain distance dictionary tracking shortest known distance to each node.
    Explore nodes in order of their distance from start to find optimal path.
    '''
    import heapq
    dist = {node: float('inf') for node in graph}
    dist[start] = 0

    pq = [(0, start)]

    while pq:
        curr_dist, node = heapq.heappop(pq)

        if curr_dist > dist[node]:
            continue

        if node == end:
            return curr_dist

        for neighbor, weight in graph[node]:
            new_dist = curr_dist + weight
            if new_dist < dist[neighbor]:
                dist[neighbor] = new_dist
                heapq.heappush(pq, (new_dist, neighbor))

    return dist[end]

def build_graph(n, m):
    '''
    Build adjacency list from stdin input.
    Read m edges with format: u v w (edge from u to v with weight w).
    Create undirected graph by adding edges in both directions.
    '''
    graph = {i: [] for i in range(n)}
    for _ in range(m):
        u, v, w = map(int, input().split())
        graph[u].append((v, w))
        graph[v].append((u, w))
    return graph

if __name__ == '__main__':
    main()
```
"""

    @classmethod
    def get_cof_for_problem(cls, question: CodeGenerationProblem) -> str:
        """
        Get the appropriate COF prompt based on the problem's platform.

        Args:
            question: The code generation problem

        Returns:
            The appropriate COF prompt string
        """
        if question.platform == Platform.LEETCODE:
            return cls.COF_LEETCODE
        else:
            return cls.COF_GENERAL
