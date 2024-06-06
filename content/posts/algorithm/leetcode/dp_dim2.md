---
title: "Dynamic Programming: Dim-2"
date: 2024-05-28T09:47:15Z
lastmod: 2024-05-28
draft: false
description: ""
tags: ["leetcode", "dynamic programming"]
# series: ["LeetCode"]
# series_order: 2
# layout: "simple"
showDate: true
---
## 二维动态规划的基本思路

基本思路与一维 DP 类似：
1. 首先完成递归解法，2维递归的时间复杂度\\(O(n^3)\))，空间复杂度\\(O((log_n)^2)\\)，时间复杂度高主要原因是存在重复计算
    - 递归时，要确定 base case，即判断递归到达底部边界，及其返回值
    - 二维递归的特点是f(i,j)的结果可能依赖于前面的多种情况，比如f(i-1,j-1), f(i-1,j), f(i, j-1)等
2. 递归转换为带有2维 DP 缓存表的、递归版的、自顶向底的 DP
    - 顶指的是最终目标答案，底指的是base case
    - 递归版 DP 基本只是在递归解法的基础上增加了 DP 表，其他逻辑照抄
    - 以空间换时间，dp[i][j]存f(i,j)的结果，递归时，如果 dp[i][j] 已存在则直接返回结果，否则继续递归，并将结果存入dp[i][j]，这样就减少了重复计算
    - 时间复杂度\\(O(n)\))，空间复杂度\\(O(n)\\)
3. 将递归版的DP转化为严格位置依赖的、迭代版的、自底向顶的DP
    - dp[i] 依赖于若干个 dp[<i] 的值，最终结果是 dp[n]
    - 时间复杂度\\(O(n)\))，空间复杂度\\(O(n)\\)
4. 继续优化DP，空间压缩
    - 用有限个变量的滚动更新代替dp数组，比如 cur = last + lastlast
    - 时间复杂度\\(O(n)\))，空间复杂度\\(O(1)\\)

## leetcode 64.最小路径和[中等]

[64.最小路径和](https://leetcode.cn/problems/minimum-path-sum/submissions/535334532/)

给定一个包含非负整数的 m x n 网格 grid ，请找出一条从左上角到右下角的路径，使得路径上的数字总和为最小。

说明：每次只能向下或者向右移动一步。

```c++
class Solution {
public:
    int minPathSum(vector<vector<int>>& grid) {
        // return f1(grid, grid.size()-1, grid[0].size()-1);
        // return f2(grid);
        // return f3(grid);
        return f4(grid);
    }
    //递归：只能想下或向右移动, f(i,j) = max(f(i-1,j), f(i,j-1))+grid[i,j]
    int f1(vector<vector<int>>& grid, int i, int j){
        if(i==0 && j==0){
            return grid[0][0];
        }
        int up=INT_MAX;
        int left=INT_MAX;
        if(i-1>=0){
            up = f1(grid,i-1,j);
        }
        if(j-1>=0){
            left = f1(grid,i,j-1);
        }
        return min(up, left)+grid[i][j];
    }
    //DP
    int f2(vector<vector<int>>& grid){
        vector<vector<int>> dp(grid.size(), vector<int>(grid[0].size(), -1));//存放(i,j)位置的最短路径
        dp[0][0] = grid[0][0];
        return f2_dp1(grid, grid.size()-1, grid[0].size()-1, dp);
    }
    //dp_1，递归版、自顶向底的DP
    int f2_dp1(vector<vector<int>>& grid, int i, int j, vector<vector<int>>& dp){
        if(dp[i][j] != -1) return dp[i][j];
        int up=INT_MAX;
        int left=INT_MAX;
        if(i-1>=0){
            up = f2_dp1(grid,i-1,j,dp);
        }
        if(j-1>=0){
            left = f2_dp1(grid,i,j-1,dp);
        }
        dp[i][j]= min(up, left)+grid[i][j];
        return dp[i][j];
    }
    // 优化DP， 递归转迭代, 严格位置依赖的动态规划
    // 从左往右，从上往下，一行一行的计算dp[i][j]
    int f3(vector<vector<int>>& grid){
        vector<vector<int>> dp(grid.size(), vector<int>(grid[0].size(), -1));//存放(i,j)位置的最短路径
        dp[0][0] = grid[0][0];
        for(int i=1; i<grid.size(); i++){
            dp[i][0] = dp[i-1][0]+grid[i][0];
        }
        for(int j=1; j<grid[0].size(); j++){
            dp[0][j] = dp[0][j-1]+grid[0][j];
        }
        for(int i=1; i<grid.size(); i++){
            for(int j=1; j<grid[0].size(); j++){
                  dp[i][j] = min(dp[i-1][j], dp[i][j-1]) + grid[i][j];
            }
        }
        return dp[grid.size()-1][grid[0].size()-1];
    }
    // 继续优化DP，减小dp数组，二位dp缓存表 先转成2个一维数组，再优化就是一个以为数组，grid每一行复用
    int f4(vector<vector<int>>& grid){
        vector<int> dp(grid[0].size());
        dp[0] = grid[0][0];
        for(int j=1; j<grid[0].size(); j++){
            dp[j] = dp[j-1]+grid[0][j];
        }
        for(int i=1; i<grid.size(); i++){
            // i = 1，dp表变成想象中二维表的第1行的数据
            // ...
            // i = n-1，dp表变成想象中二维表的第n-1行的数据

            //先更新一下这一行左侧第一个值，相当与原来的dp[i][j=0]
            dp[0] += grid[i][0];
            for(int j=1; j<grid[0].size(); j++){
                // dp[j]代表了原来的up=dp[i][j-1]的值，dp[j-1]原来的left=dp[i-1][j]的值
                dp[j] = min(dp[j], dp[j-1]) + grid[i][j];
            }
        }
        return dp[grid[0].size()-1];
    }
};
```

## leetcode 1147. 单词搜索[中等]

[1147. 单词搜索](https://leetcode.cn/problems/word-search/submissions/535380947/)

给定一个 m x n 二维字符网格 board 和一个字符串单词 word 。如果 word 存在于网格中，返回 true ；否则，返回 false 。

单词必须按照字母顺序，通过相邻的单元格内的字母构成，其中“相邻”单元格是那些水平相邻或垂直相邻的单元格。同一个单元格内的字母不允许被重复使用。

```c++
class Solution {
public:
    bool exist(vector<vector<char>>& board, string word) {
        // 带路径的递归
        // 双循环遍历每一个位置，从该位置作起点，用递归法搜索相邻位置，看是够能够完全匹配当w
        // 递归时，退出条件是:1.匹配到完整的w，返回true；2.无法匹配，返回false。如果当前字符匹配，则继续通过相邻位置去匹配下一个字符。走过的位置用mask掩盖下，防止后续搜索中重复过去的位置
        int is = board.size(), js = board[0].size();
        for(int r = 0; r < is; r++){
            for(int c = 0; c < js; c++){
                if(f1(board, word, 0, r, c)){
                    return true;
                }
            }
        }
        return false;
    }
    // k代表word需要匹配的位置
    bool f1(vector<vector<char>>& board, string& word, int k, int i, int j){
        if(k == word.size()){
            return true;
        }
        if(i < 0 || j < 0 || i >= board.size() || j >= board[0].size() || board[i][j] != word[k]){
            return false;
        }
        // 加入mask，但要保留字符用于递归结束的回填
        auto tmp = board[i][j];
        board[i][j] = '0';// 这时如果下面的递归中再来到这个位置，一定有b[i][j] != w[k]
        bool result = f1(board, word, k + 1, i - 1, j);
        result = result || f1(board, word, k + 1, i + 1, j);
        result = result || f1(board, word, k + 1, i, j - 1);
        result = result || f1(board, word, k + 1, i, j + 1);
        board[i][j] = tmp;
        return result;
    }
};
```

## leetcode 1143. 最长公共子序列[中等]

[1143. 最长公共子序列](https://leetcode.cn/problems/longest-common-subsequence/submissions/535642129/)

给定两个字符串 text1 和 text2，返回这两个字符串的最长 公共子序列 的长度。如果不存在 公共子序列 ，返回 0 。

一个字符串的 子序列 是指这样一个新的字符串：它是由原字符串在不改变字符的相对顺序的情况下删除某些字符（也可以不删除任何字符）后组成的新字符串。

例如，"ace" 是 "abcde" 的子序列，但 "aec" 不是 "abcde" 的子序列。
两个字符串的 公共子序列 是这两个字符串所共同拥有的子序列。

 

示例 1：

输入：text1 = "abcde", text2 = "ace" 
输出：3  
解释：最长公共子序列是 "ace" ，它的长度为 3 。
示例 2：

输入：text1 = "abc", text2 = "abc"
输出：3
解释：最长公共子序列是 "abc" ，它的长度为 3 。
示例 3：

输入：text1 = "abc", text2 = "def"
输出：0
解释：两个字符串没有公共子序列，返回 0 。

```c++
class Solution {
public:
    int longestCommonSubsequence(string text1, string text2) {
        return f3(text1,text2);
    }
    // 递归法(超时)
    // a[i],b[j]的结果依赖于三种情况
    int f1(string text1, string text2){
        return f1_help(text1,text2,text1.size(),text2.size());
    }
    int f1_help(string a, string b, int len1, int len2){
        if(len1==0 || len2==0) return 0;
        int ans;
        if(a[len1-1]==b[len2-1]){
            ans = 1+f1_help(a,b,len1-1,len2-1);
        }else{
            ans = max(f1_help(a,b,len1-1,len2), f1_help(a,b,len1,len2-1));//还有一种可能f1_help(a,b,i-1,j-1)，必然小于这两种可能，所以直接舍弃
        }
        return ans;
    }
    //DP，递归直接转dp表，自顶向底的DP
    int f2(string text1, string text2){
        int n=text1.size();
        int m=text2.size();
        vector<vector<int>> dp(n+1, vector<int>(m+1,-1));
        return f2_help(text1,text2,n,m,dp);//[0...n] [0...m]
    }
    int f2_help(string a, string b, int len1, int len2, vector<vector<int>> &dp){
        //二维dp表的第一行和第一列为0
        if(len1==0 || len2==0){
            return 0;
        }
        if(dp[len1][len2]!=-1) {
            return dp[len1][len2];
        }
        int ans;
        if(a[len1-1]==b[len2-1]){
            ans = 1+f2_help(a,b,len1-1,len2-1,dp);
        }else{
            ans = max(f2_help(a,b,len1-1,len2,dp), f2_help(a,b,len1,len2-1,dp));//还有一种可能f1_help(a,b,i-1,j-1)，必然小于这两种可能，所以直接舍弃
        }
        dp[len1][len2]=ans;
        return ans;
    }
    //DP-2:严格位置依赖的DP（递归转迭代，自底向顶的DP）
    int f3(string text1, string text2){
        int n=text1.size();
        int m=text2.size();
        vector<vector<int>> dp(n+1, vector<int>(m+1,0));
        //二维dp表的第一行和第一列为0，但这两个循环可以省略
        // for(int i=0; i<=m;i++){
        //     dp[0][m]=0;
        // }
        // for(int i=0; i<=n;i++){
        //     dp[n][0]=0;
        // }

        //开始更新二维dp表，从上向下，从左到右，按行更新，循环的i、j是len1、len2
        //dp[i][j]依赖三个位置，左上，上，左
        for(int i=1;i<=n;i++){
            for(int j=1;j<=m;j++){
                int leftup=dp[i-1][j-1];
                int left=dp[i][j-1];
                int up=dp[i-1][j];
                if(text1[i-1]==text2[j-1]){
                    dp[i][j] = leftup+1;
                }else{
                    dp[i][j] = max(left, up);
                }
            }
        }
        return dp[n][m];
    }
    //DP-3：对DP2的二维dp表压缩为数组（一行），位置依赖中的up和left很容易，但是leftup需要把前一个的备份一下
    // Todo
};
```

## leetcode 516. 最长回文子序列[中等]

[516. 最长回文子序列](https://leetcode.cn/problems/longest-palindromic-subsequence/submissions/535669641/)

给你一个字符串 s ，找出其中最长的回文子序列，并返回该序列的长度。

子序列定义为：不改变剩余字符顺序的情况下，删除某些字符或者不删除任何字符形成的一个序列。

示例 1：

输入：s = "bbbab"
输出：4
解释：一个可能的最长回文子序列为 "bbbb" 。
示例 2：

输入：s = "cbbd"
输出：2
解释：一个可能的最长回文子序列为 "bb" 。


```c++
class Solution {
public:
    int longestPalindromeSubseq(string s) {
        return f3(s);
    }
    // 暴力：搞一个逆转的string，两个比较
    // 递归(超时)：left,right两个指针，left->...<-right
    // s[left...right]的最长回文子序列，依赖于四种情况
    // 1. s[left]==s[right]时，结果为 2 + s[left+1...right-1]的最长回文子序列
    // 2, s[left]!=s[right]时，有三种情况,取最大值:
    // s[left+1...right], s[left...right-1], s[left+1...right-1]，其中欧冠你最后一种可以省略
    int f1(string s){
        if(s.size()==0) return 0;
        int l = 0;
        int r = s.size()-1;
        return f1_help(s, l, r);
    }
    int f1_help(string s, int l, int r){
        if(l==r) return 1;//如 a
        if(l+1==r) return s[l]==s[r] ? 2: 1;//如 aa 或 ab
        if(s[l]==s[r]){
            return 2 + f1_help(s, l+1, r-1);
        }else{
            return max(f1_help(s, l+1,r),f1_help(s, l,r-1));
        }
    }
    //DP(内存超限): 二维dp表，自顶向底的DP（递归版DP）
    int f2(string s){
        if(s.size()==0) return 0;
        int l = 0;
        int r = s.size()-1;
        vector<vector<int>> dp(s.size(), vector<int>(s.size()));
        return f2_help(s, l, r, dp);
    }
    int f2_help(string s, int l, int r, vector<vector<int>> &dp){
        if(l==r) return 1;//如 a
        if(l+1==r) return s[l]==s[r] ? 2: 1;//如 aa 或 ab
        if(dp[l][r]!=0) return dp[l][r];
        int ans;
        if(s[l]==s[r]){
            ans = 2 + f2_help(s, l+1, r-1, dp);
        }else{
            ans = max(f2_help(s, l+1,r,dp),f2_help(s, l,r-1,dp));
        }
        dp[l][r] = ans;
        return ans;
    }
    //DP: 严格位置依赖的DP，迭代版
    // dp表的对角线上的值dp[i][i]=1,该值的右侧dp[i][i+1]=s[i]==s[i+1] ? 2: 1
    // dp表对角线左下的元素都是l<r，无意义
    // dp表中其他的值依赖于三个位置：左侧、下侧、左下侧
    // 注意：最终结果是表的右上角，dp表要从下往上，从左往右去更新
    int f3(string s){
        if(s.size()==0) return 0;
        int n = s.size();
        vector<vector<int>> dp(n, vector<int>(n));
        for(int i=n-1; i>=0; i--){
            dp[i][i] = 1;
            if(i+1<n){
                dp[i][i+1]=s[i]==s[i+1] ? 2: 1;
            }
            for(int j=i+2; j<n; j++){
                if(s[i]==s[j]){
                    dp[i][j] = 2 + dp[i+1][j-1];//左下侧
                }else{
                    dp[i][j] = max(dp[i+1][j], dp[i][j-1]);//左和右侧
                }
            }
        }
        return dp[0][n-1];
    }
    //DP继续优化，可以压缩空间，用数组代替而二维表，注意加一个变量缓存迁移个位置的左下侧结果

};
```

Reference:

[左程云算法](https://www.bilibili.com/video/BV1WQ4y1W7d1/?spm_id_from=333.999.0.0&vd_source=4020b2f7b29d7059ea21212421b541f1)