int find_max(const std::vector<int>& numbers) {
    if (numbers.empty()) return 0;  // Return 0 for an empty list
    return *std::max_element(numbers.begin(), numbers.end());
}