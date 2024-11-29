int fibonacci(int n) {
    static std::unordered_map<int, int> cache;

    if (n <= 1) return n;
    if (cache.find(n) != cache.end()) return cache[n];

    cache[n] = fibonacci(n - 1) + fibonacci(n - 2);
    return cache[n];
}