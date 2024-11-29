double average(const std::vector<int>& numbers) {
    if (numbers.empty()) return 0.0;
    int sum = 0;
    for (int num : numbers) {
        sum += num;
    }
    return static_cast<double>(sum) / numbers.size();
}