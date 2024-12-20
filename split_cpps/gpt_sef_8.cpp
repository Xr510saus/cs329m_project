double median(std::vector<int> numbers) {
    if (numbers.empty()) return 0.0;
    std::sort(numbers.begin(), numbers.end());
    int n = numbers.size();
    if (n % 2 == 0) {
        return (numbers[n / 2 - 1] + numbers[n / 2]) / 2.0;
    } else {
        return numbers[n / 2];
    }
}