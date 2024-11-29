bool isPalindrome(const string &s) {
    int len = s.size();
    for (int i = 0; i < len / 2; ++i) {
        if (s[i] != s[len - i - 1]) {
            return false;
        }
    }
    return true;
}